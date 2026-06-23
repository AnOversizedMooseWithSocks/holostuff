"""Denoising as manifold projection, and the Plug-and-Play / RED restoration loop.

WHY THIS EXISTS
---------------
Milanfar's thesis: a denoiser is a MAP OF THE MANIFOLD that clean signals live on. holostuff
already owns two such maps -- `cleanup` (snap to the codebook manifold) and `consolidation` (the
low-rank SVD subspace real states occupy). This module exposes them as a callable denoiser and
wraps the standard Plug-and-Play / Regularization-by-Denoising loop (Venkatakrishnan et al. 2013;
Romano, Elad, Milanfar 2017) around them, so the SAME map that denoises also solves any inverse
problem (inpainting, deblurring, reconstruction-under-erasure).

MEASURED (on real SOL price windows; manifold = the consolidation/SVD subspace of clean windows)
  * Projection denoising WINS increasingly as noise grows: +1.7 dB SNR at sigma=0.5, +3.85 dB at
    sigma=0.8 -- the off-manifold directions become mostly noise.
  * KEPT NEGATIVE: it HURTS at low noise (-1.4 dB at sigma=0.3) by discarding real signal detail
    -- projecting onto a fixed low rank over-smooths a clean signal. The method needs the noise
    level (or an adaptive rank/threshold) to be applied well -- exactly the Donoho/Milanfar
    threshold-selection problem.
  * HONEST CONTROL: on random data with no low-rank manifold, projection DESTROYS signal (-5 dB).
    The map only helps where real structure exists. `manifold_denoise` is therefore only a
    denoiser to the extent the fitted basis captures the signal.

DESIGN NOTES
  * `fit_manifold` is the same operation as the engine's consolidation (SVD), kept standalone here
    so denoising does not depend on a creature/brain being present.
  * `pnp_restore` accepts ANY denoiser callable -- pass `manifold_denoise` (projection) or the
    modern-Hopfield `dense_cleanup` (codebook) -- so the loop is agnostic to which map you use.
  * Pure NumPy, deterministic.
"""

import numpy as np


def fit_manifold(samples, rank=8):
    """Learn a signal manifold from clean `samples` (rows) as a rank-`rank` SVD subspace.

    Returns (basis, mean) where basis is (rank x dim) orthonormal -- this IS the consolidation
    step, used here as the map a denoiser projects onto."""
    X = np.asarray(samples, float)
    mean = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    rank = int(min(rank, Vt.shape[0]))
    return Vt[:rank], mean


def manifold_denoise(x, basis, mean):
    """Denoise `x` by projecting onto the affine manifold (mean + span(basis)).

    The holostuff denoiser: keep only the components that lie in the signal subspace, drop the
    rest as noise. Only as good as the basis -- see the kept negative in the module docstring."""
    x = np.asarray(x, float)
    return mean + (x - mean) @ basis.T @ basis


def fit_manifold_full(samples, rank=None):
    """Like fit_manifold but keeps a GENEROUS basis AND its singular values, so a per-signal noise
    threshold (adaptive_manifold_denoise) can decide how many components to keep at denoise time.
    rank=None keeps every component. Returns (basis, sv, mean)."""
    X = np.asarray(samples, float)
    mean = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    if rank is not None:
        Vt, S = Vt[:int(rank)], S[:int(rank)]
    return Vt, S, mean


def estimate_sigma(x):
    """Donoho's robust noise estimate: the MAD of the finest detail (successive differences),
    rescaled. Parameter-free; good when the clean signal is smoother than the noise."""
    d = np.diff(np.asarray(x, float))
    if d.size == 0:
        return 0.0
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2.0))


def adaptive_manifold_denoise(x, basis, mean, sigma=None, kappa=1.0):
    """Adaptive denoiser: project x onto a GENEROUS manifold basis, then HARD-THRESHOLD the projection
    coefficients at a NOISE-DRIVEN level (Donoho-Johnstone shrinkage in the manifold basis). With an
    orthonormal basis each coefficient carries noise of std ~sigma, so dropping |c| <= kappa*sigma*sqrt(
    2 ln r) removes noise-dominated directions and keeps signal-bearing ones.

    This cashes the fixed-rank denoiser's kept negative -- at LOW noise the threshold is tiny so nearly
    all detail survives (no over-smoothing), while a fixed rank-k always truncates and discards real
    detail there; at HIGH noise only strong signal components survive (full denoising). sigma is
    estimated from x if not given (the Donoho/Milanfar threshold-selection step)."""
    x = np.asarray(x, float)
    r = basis.shape[0]
    if sigma is None:
        sigma = estimate_sigma(x)
    thr = kappa * sigma * np.sqrt(2.0 * np.log(max(r, 2)))   # universal threshold, scaled
    c = (x - mean) @ basis.T                                 # projection coefficients
    c = np.where(np.abs(c) > thr, c, 0.0)                    # keep only signal-bearing directions
    return mean + c @ basis


def codebook_denoise(x, codebook, beta=25.0, steps=3):
    """Denoise `x` by snapping toward the codebook manifold via the modern-Hopfield update.
    Thin re-export of holographic_hopfield.dense_cleanup so callers can pick a manifold (subspace)
    or a codebook denoiser without importing two modules."""
    from holographic_hopfield import dense_cleanup
    return dense_cleanup(x, codebook, beta=beta, steps=steps)


def pnp_restore(y, forward, adjoint, denoiser, mu=0.5, steps=30, x0=None):
    """Plug-and-Play / RED restoration: recover x from a degraded measurement y = forward(x)+noise
    by alternating a data-fidelity gradient step with a denoise step.

        x <- x - mu * adjoint(forward(x) - y)     # pull toward agreement with the measurement
        x <- denoiser(x)                          # pull toward the signal manifold (the prior)

    `forward`/`adjoint` are callables for the degradation operator A and its transpose A^T (for
    inpainting, A is a binary mask and A^T == A; for plain denoising, both are identity). Any
    denoiser callable works. Returns the restored vector. Deterministic given a deterministic
    denoiser and x0."""
    y = np.asarray(y, float)
    x = np.asarray(x0, float).copy() if x0 is not None else adjoint(y).astype(float)
    for _ in range(steps):
        x = x - mu * adjoint(forward(x) - y)       # data-fidelity descent
        x = np.asarray(denoiser(x), float)         # the prior, applied as a denoiser
    return x


def nlm_denoise(patches, k=12, h=0.5, use_forest=True):
    """Non-local-means denoising (Buades, Coll, Morel 2005; BM3D, Dabov et al. 2007) running on
    holostuff's OWN content-addressable recall.

    "Find the patches that look like this one and average them" -- averaging k near-duplicate
    patches cancels the independent noise in each (a ~1/sqrt(k) reduction). The neighbour search
    is exactly recall, so it runs sub-linearly through `HoloForest.recall_k`; with use_forest=False
    it falls back to exact cosine kNN (handy for small sets and for a deterministic reference).

    COMPLEMENTARY to manifold_denoise, not a replacement -- measured:
      * self-similar signals (repeated motifs): NLM wins big (averages the duplicates) -- e.g. on
        real SOL motif-windows, ~11 dB vs ~7 dB for rank-8 projection.
      * low-rank but NOT self-similar (every patch unique): projection wins, NLM has nothing to
        average -- ~2.8 dB vs ~0.5 dB. KEPT NEGATIVE: NLM only helps where near-duplicates exist.

    `patches` is (N, D). Returns the denoised (N, D). Weights are softmax(cosine / h) over the
    k nearest (including self), so a closer neighbour counts more; smaller h = more selective."""
    X = np.asarray(patches, float)
    N = len(X)
    k = min(k, N)
    out = np.empty_like(X)

    if use_forest:
        from holographic_tree import HoloForest
        forest = HoloForest(X.shape[1], n_trees=4, leaf_size=max(8, N // 16), seed=0).build(X)
        for i in range(N):
            idx, sims = forest.recall_k(X[i], k=k)
            if len(idx) == 0:                       # nothing routed -> keep the patch as-is
                out[i] = X[i]; continue
            w = np.exp(sims / h); w /= w.sum()
            out[i] = w @ X[idx]
        return out

    # exact reference path: all-pairs cosine, top-k per patch
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = U @ U.T
    for i in range(N):
        idx = np.argsort(S[i])[::-1][:k]
        w = np.exp(S[i, idx] / h); w /= w.sum()
        out[i] = w @ X[idx]
    return out
