"""Denoising as manifold projection + Plug-and-Play restoration (B7)."""
import numpy as np
from holographic_denoise import (fit_manifold, manifold_denoise, pnp_restore,
                                 codebook_denoise, nlm_denoise)


def _low_rank_signals(n=400, dim=32, rank=5, seed=0):
    rng = np.random.default_rng(seed)
    B = np.linalg.svd(rng.standard_normal((rank, dim)), full_matrices=False)[2]
    coeffs = rng.standard_normal((n, rank))
    return coeffs @ B                          # genuinely rank-`rank` signals


def _snr(clean, est):
    return 10 * np.log10(np.sum(clean ** 2) / (np.sum((clean - est) ** 2) + 1e-12))


def test_manifold_projection_denoises_high_noise_low_rank_signal():
    rng = np.random.default_rng(1)
    X = _low_rank_signals()
    basis, mean = fit_manifold(X[:300], rank=6)
    raw = proj = 0.0
    for x in X[300:380]:
        noisy = x + 0.7 * rng.standard_normal(x.shape[0])
        raw += _snr(x, noisy); proj += _snr(x, manifold_denoise(noisy, basis, mean))
    assert proj > raw                          # projection helps when noise dominates off-manifold


def test_manifold_projection_does_not_help_random_data():
    # honest control: no low-rank manifold -> projecting onto a spurious subspace cannot denoise.
    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 32))
    basis, mean = fit_manifold(X[:300], rank=6)
    raw = proj = 0.0
    for x in X[300:380]:
        noisy = x + 0.7 * rng.standard_normal(32)
        raw += _snr(x, noisy); proj += _snr(x, manifold_denoise(noisy, basis, mean))
    assert proj < raw + 0.5                     # no real gain (and typically a loss)


def test_pnp_restore_recovers_an_inpainting_problem():
    # use the manifold denoiser as the prior to fill erased entries (A = a binary mask, A^T == A).
    rng = np.random.default_rng(3)
    X = _low_rank_signals(dim=40, rank=5)
    basis, mean = fit_manifold(X[:300], rank=6)
    x = X[350]
    mask = (rng.random(40) > 0.4).astype(float)        # keep ~60% of entries
    A = lambda v: mask * v
    y = A(x)
    den = lambda v: manifold_denoise(v, basis, mean)
    rec = pnp_restore(y, A, A, den, mu=0.8, steps=60)
    assert _snr(x, rec) > _snr(x, y)                   # restoration beats the masked measurement


def _motif_signal(M, R, D=24, sigma=0.6, seed=0):
    rng = np.random.default_rng(seed)
    motifs = rng.standard_normal((M, D))
    motifs /= np.linalg.norm(motifs, axis=1, keepdims=True)
    clean = np.repeat(motifs, R, axis=0)
    return clean, clean + sigma * rng.standard_normal(clean.shape)


def test_nlm_beats_projection_on_self_similar_signal():
    # repeated motifs -> NLM averages the near-duplicates and cancels noise; projection cannot.
    clean, noisy = _motif_signal(M=20, R=8)
    basis, mean = fit_manifold(noisy, rank=8)
    proj = np.stack([manifold_denoise(x, basis, mean) for x in noisy])
    nlm = nlm_denoise(noisy, k=8, use_forest=True)
    s = lambda A: np.mean([_snr(clean[i], A[i]) for i in range(len(clean))])
    assert s(nlm) > s(proj) and s(nlm) > s(noisy) + 3.0


def test_projection_beats_nlm_without_self_similarity():
    # KEPT NEGATIVE / complementarity: low-rank but every patch unique -> NLM has no duplicates to
    # average, projection captures the subspace and wins. The two denoisers cover different worlds.
    rng = np.random.default_rng(3)
    X = _low_rank_signals(n=400, dim=32, rank=5)        # all-unique low-rank patches
    noisy = X + 0.6 * rng.standard_normal(X.shape)
    basis, mean = fit_manifold(noisy, rank=6)
    proj = np.stack([manifold_denoise(x, basis, mean) for x in noisy])
    nlm = nlm_denoise(noisy, k=8, use_forest=True)
    s = lambda A: np.mean([_snr(X[i], A[i]) for i in range(len(X))])
    assert s(proj) > s(nlm)


def test_forest_recall_k_finds_near_duplicates():
    # the recall step: a duplicated patch's k-nearest should be dominated by its other copies.
    from holographic_tree import HoloForest
    clean, noisy = _motif_signal(M=10, R=8, sigma=0.2, seed=5)
    f = HoloForest(noisy.shape[1], n_trees=4, leaf_size=8, seed=0).build(noisy)
    idx, sims = f.recall_k(noisy[0], k=6)
    assert len(idx) >= 1 and sims[0] >= sims[-1]        # ranked descending
    # the closest neighbours should come from the same motif block (the first 8 rows)
    assert np.mean(idx[:4] < 8) >= 0.5
