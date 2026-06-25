"""Holographic Gaussian splatting -- a scene/field as a SUPERPOSITION of Gaussian primitives.

WHY THIS EXISTS
---------------
3D Gaussian Splatting (Kerbl, Kopanas, Leimkuehler, Drettakis 2023) represents a scene as an
explicit SUM of parameterised Gaussians fit to data. holostuff's `bundle` IS superposition -- so a
splat scene is, structurally, a bundle of primitives. This module makes that concrete for 2-D
fields/images: fit K Gaussian splats by matching pursuit (greedy superposition), render the sum,
and -- because a small set of smooth Gaussians cannot represent high-frequency noise -- use the
fit itself as a denoiser.

MEASURED (on a real (log-return, log-volume) density from the SOL market data)
  * ~20 superposed Gaussians reconstruct the density at ~31 dB PSNR using ~3.5% of the pixel
    budget (4 numbers per splat). It plateaus by ~50 splats -- a compact, adaptive code.
  * Fitting 20 splats to a NOISY density (22.6 dB to clean) recovers it to ~27 dB -- denoising by
    splatting, because the representation has no capacity for noise.
  * The bridge to the rest of the engine: holostuff's RBF `ScalarEncoder` already places a
    Gaussian bump in similarity space, i.e. it is Gaussian splatting in the hypervector domain.
    The splat <-> kernel <-> FHRR-phasor chain is one object.

DESIGN NOTES
  * Isotropic splats and a small fixed scale set keep the fit a clean, deterministic matching
    pursuit. KEPT NEGATIVE / SCOPE: anisotropic covariances and gradient refinement (full 3DGS)
    are deliberately out of scope here -- isotropic matching pursuit is the honest baseline, and
    real images plateau in quality once the smooth structure is captured (noise is, correctly,
    not fit).
  * Pure NumPy, deterministic: the greedy order is fixed by the residual, no RNG.
"""

import numpy as np


def _gaussian(shape, cy, cx, sigma):
    """A unit-L2-norm isotropic Gaussian centred at (cy, cx) on a `shape` grid."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    g = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma * sigma))
    return g / (np.sqrt((g * g).sum()) + 1e-12)


def splat_fit(target, K, scales=(1.0, 2.0, 3.5, 6.0)):
    """Fit `target` (a 2-D array) with K Gaussian splats by matching pursuit.

    Each step places a splat at the current residual's peak, picks the scale (from `scales`) that
    explains the most residual energy, fits its amplitude by projection, and subtracts it. Returns
    a list of (cy, cx, amplitude, sigma) -- the scene as an explicit superposition of primitives."""
    target = np.asarray(target, float)
    R = target.copy()
    splats = []
    for _ in range(K):
        cy, cx = np.unravel_index(np.abs(R).argmax(), R.shape)
        best = None                                   # (energy, amp, sigma, g)
        for s in scales:
            g = _gaussian(R.shape, cy, cx, s)
            amp = float((R * g).sum())                # least-squares amplitude (g is unit norm)
            energy = amp * amp
            if best is None or energy > best[0]:
                best = (energy, amp, s, g)
        _, amp, s, g = best
        R = R - amp * g
        splats.append((int(cy), int(cx), amp, s))
    return splats


def splat_render(splats, shape):
    """Render a splat list back to a 2-D array -- the superposition (sum) of its primitives."""
    out = np.zeros(shape, float)
    for cy, cx, amp, s in splats:
        out += amp * _gaussian(shape, cy, cx, s)
    return out


def splat_denoise(noisy, K, scales=(1.0, 2.0, 3.5, 6.0)):
    """Denoise a 2-D field by fitting K splats and rendering them: the smooth Gaussian basis
    captures structure but not high-frequency noise, so the fit is a denoiser."""
    return splat_render(splat_fit(noisy, K, scales), np.asarray(noisy).shape)


def psnr(a, b, peak=1.0):
    """Peak-signal-to-noise ratio in dB between two arrays (99.0 if identical)."""
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    return 99.0 if mse == 0.0 else float(10.0 * np.log10(peak * peak / mse))


# --- the HOLOGRAPHIC layer: a splat scene AS a bundle, queryable by region ----------------------
# "a splat scene is a bundle" made literal: bundle a per-region descriptor of the splats, each bound
# to a region role, into ONE hypervector, and read a region back by unbinding its role. This is the
# content-addressable "what's roughly HERE" query the archive's exact splat-list lookup complements.

def splat_bundle(splats, shape, dim=4096, grid=8, levels=5, seed=0):
    """Encode a splat scene as ONE hypervector: partition `shape` into grid x grid regions, quantise each
    region's PEAK occupancy to one of `levels` near-orthogonal level atoms, and bundle bind(region_role,
    level_atom) over all regions. Returns (scene_hv, ctx); ctx carries the role + level codebooks + grid so
    recall_region can read a region back. The bundle IS a superposition -- the engine's bundle over the
    scene's own primitives. (Quantised levels with ORTHOGONAL atoms, not a continuous RBF value, so the
    per-region readback survives the bundle crosstalk -- the readout is robust, the value is coarse.)"""
    from holographic_ai import bind, bundle, Vocabulary
    H, W = shape[0], shape[1]
    rendered = splat_render(splats, (H, W))
    roles = Vocabulary(dim, seed=seed)
    lvl = Vocabulary(dim, seed=seed + 1)                  # `levels` near-orthogonal occupancy atoms
    peak = float(np.abs(rendered).max()) + 1e-12
    parts, desc = [], {}
    for gy in range(grid):
        for gx in range(grid):
            ys, ye = gy * H // grid, (gy + 1) * H // grid
            xs, xe = gx * W // grid, (gx + 1) * W // grid
            energy = float(np.clip(np.abs(rendered[ys:ye, xs:xe]).max() / peak, 0.0, 1.0))
            q = int(round(energy * (levels - 1)))         # quantise PEAK occupancy to a level index
            desc[(gy, gx)] = q / (levels - 1)
            parts.append(bind(roles.get(f"cell:{gy}:{gx}"), lvl.get(f"lvl:{q}")))
    ctx = {"roles": roles, "lvl": lvl, "levels": levels, "grid": grid, "desc": desc}
    return (bundle(parts) if parts else np.zeros(dim)), ctx


def recall_region(scene_hv, cell, ctx):
    """Read a region's quantised occupancy back out of a splat-bundle by unbinding its role and cleaning
    up against the orthogonal level atoms -- content-addressable region lookup. `cell` is (gy, gx). Returns
    the recovered occupancy in [0, 1]. COARSE by design (quantised levels); for exact per-splat region
    content use SplatArchive.region, the precise complement."""
    from holographic_ai import unbind
    roles, lvl, L = ctx["roles"], ctx["lvl"], ctx["levels"]
    noisy = unbind(np.asarray(scene_hv, float), roles.get(f"cell:{cell[0]}:{cell[1]}"))
    best_q, best_s = 0, -2.0
    for q in range(L):
        a = lvl.get(f"lvl:{q}")
        s = float(noisy @ a / (np.linalg.norm(noisy) * np.linalg.norm(a) + 1e-12))
        if s > best_s:
            best_q, best_s = q, s
    return best_q / (L - 1)


# --- anisotropic splats: full-covariance Gaussians fit by gradient descent (the real 3DGS primitive) ------
# Each splat is (center, amplitude, L) where L is an n*n lower-triangular Cholesky factor of the INVERSE
# covariance, so the Gaussian is amp * exp(-0.5 * ||L^T (x - center)||^2). L lower-triangular keeps the
# precision positive-definite for free. Works in any dimension -- 2-D fields and 3-D volumes share one fit.

def _coords(shape):
    """All voxel coordinates of an n-D array as an (npix, n) float array (row-major), built once per fit."""
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1).astype(float)


def _iso_pursuit(target, K, scales=(1.0, 2.0, 3.5, 6.0)):
    """Isotropic matching pursuit in n-D -- the warm start for the anisotropic fit. Returns a list of
    (center (n,), peak_amplitude, sigma); render is amp * exp(-0.5 |x-center|^2 / sigma^2)."""
    R = np.asarray(target, float).copy()
    shape = R.shape
    C = _coords(shape)
    out = []
    for _ in range(K):
        ctr = np.array(np.unravel_index(np.abs(R).argmax(), shape), float)
        d2 = ((C - ctr) ** 2).sum(1)
        best = None
        for s in scales:
            g = np.exp(-0.5 * d2 / s ** 2)                      # peak 1
            amp = float((R.ravel() @ g) / (g @ g + 1e-12))      # least-squares peak amplitude
            energy = amp * amp * float(g @ g)
            if best is None or energy > best[0]:
                best = (energy, amp, s, g)
        _, amp, s, g = best
        R = (R.ravel() - amp * g).reshape(shape)
        out.append((ctr, amp, s))
    return out


def aniso_render(splats, shape):
    """Render anisotropic splats (center, amp, L) back to an n-D array -- the superposition of full-covariance
    Gaussians, exactly what aniso_fit optimises."""
    C = _coords(shape)
    out = np.zeros(C.shape[0])
    for ctr, amp, L in splats:
        u = (C - ctr) @ L
        out += amp * np.exp(-0.5 * (u * u).sum(1))
    return out.reshape(shape)


def aniso_fit(target, K, steps=200, lr=0.15, scales=(1.0, 2.0, 3.5, 6.0)):
    """Fit `target` (any n-D array) with K ANISOTROPIC Gaussian splats by gradient descent on the
    reconstruction MSE -- the 3D-Gaussian-Splatting primitive (oriented, elliptical Gaussians), in NumPy with
    analytical gradients and a small built-in Adam (no autodiff framework). Warm-started from the isotropic
    matching pursuit so the covariances only have to specialise. Each splat is (center, amplitude, L), L the
    lower-triangular Cholesky factor of the inverse covariance. Returns (splats, rendered).

    Anisotropy is decisive where structure is oriented/elongated -- one aligned splat replaces many circular
    ones. KEPT NEGATIVE: the loss is non-convex, so this finds a LOCAL optimum -- more splats do not help
    monotonically (a good K=4 fit can beat a messier K=8 one), and the result depends on the warm start. This
    is the honest from-scratch core of 3DGS, without its tile rasteriser, spherical-harmonic view-dependent
    colour, or GPU speed."""
    target = np.asarray(target, float)
    shape = target.shape
    n = target.ndim
    C = _coords(shape)
    t = target.ravel()
    iso = _iso_pursuit(target, K, scales)
    centers = np.array([c for c, _, _ in iso])
    amps = np.array([a for _, a, _ in iso])
    Ls = np.array([np.eye(n) / max(sg, 0.5) for _, _, sg in iso])      # L = (1/sigma) I  (isotropic init)
    tril = np.tril_indices(n)
    state = {key: (np.zeros_like(v), np.zeros_like(v)) for key, v in (("a", amps), ("c", centers), ("L", Ls))}
    b1, b2, eps = 0.9, 0.999, 1e-8

    def render(ce, am, Ls_):
        m = np.zeros(len(t))
        for k in range(K):
            u = (C - ce[k]) @ Ls_[k]
            m += am[k] * np.exp(-0.5 * (u * u).sum(1))
        return m

    for step in range(1, steps + 1):
        r = render(centers, amps, Ls) - t                                # residual
        ga = np.zeros_like(amps); gc = np.zeros_like(centers); gL = np.zeros_like(Ls)
        for k in range(K):
            d = C - centers[k]
            u = d @ Ls[k]                                                # u = L^T d  (per pixel)
            ex = np.exp(-0.5 * (u * u).sum(1))
            g = amps[k] * ex
            ga[k] = float((r * ex).sum())                                # dE/d amp
            Pd = u @ Ls[k].T                                             # (L L^T) d = precision * d
            gc[k] = ((r * g)[:, None] * Pd).sum(0)                       # dE/d center = sum r g (P d)
            for a_, b_ in zip(*tril):                                    # dE/d L_ab = sum r (-g) d_a u_b
                gL[k][a_, b_] = float((r * (-g) * d[:, a_] * u[:, b_]).sum())
        for key, par, grad in (("a", amps, ga), ("c", centers, gc), ("L", Ls, gL)):
            m, v = state[key]
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            par -= lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
            state[key] = (m, v)
    splats = [(centers[k].copy(), float(amps[k]), Ls[k].copy()) for k in range(K)]
    return splats, render(centers, amps, Ls).reshape(shape)
