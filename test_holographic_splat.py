"""Holographic Gaussian splatting (B8): a scene as a superposition of Gaussian primitives."""
import numpy as np
from holographic_splat import splat_fit, splat_render, splat_denoise, psnr


def _target(G=48, seed=0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:G, 0:G]
    T = np.zeros((G, G))
    for _ in range(4):                         # a smooth few-blob target splats can represent
        cy, cx, s, a = rng.uniform(8, G - 8, 2).tolist() + [rng.uniform(3, 7), rng.uniform(0.5, 1)]
        T += a * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * s * s))
    return T / T.max()


def test_more_splats_reconstruct_better_and_compactly():
    T = _target()
    q8 = psnr(T, splat_render(splat_fit(T, 8), T.shape))
    q40 = psnr(T, splat_render(splat_fit(T, 40), T.shape))
    assert q40 > q8 and q40 > 25.0             # superposition of primitives reconstructs the field


def test_splatting_denoises():
    # fitting few smooth Gaussians to noisy data recovers the clean field (no capacity for noise).
    rng = np.random.default_rng(2)
    T = _target()
    noisy = T + 0.10 * rng.standard_normal(T.shape)
    assert psnr(T, splat_denoise(noisy, 30)) > psnr(T, noisy) + 1.0


def test_rbf_encoder_is_a_gaussian_splat_in_hv_space():
    # the bridge: holostuff's RBF scalar encoder already places a Gaussian bump in similarity space.
    from holographic_encoders import ScalarEncoder
    enc = ScalarEncoder(512, lo=0.0, hi=1.0, kernel="rbf", seed=0)
    c = enc.encode(0.5)
    vals = np.linspace(0, 1, 21)
    sims = np.array([float(np.dot(c, enc.encode(v)) /
                     (np.linalg.norm(c) * np.linalg.norm(enc.encode(v)) + 1e-12)) for v in vals])
    assert abs(vals[int(sims.argmax())] - 0.5) < 0.06   # peaks at the encoded value
    assert sims.max() > 0.95 and sims.min() < sims.max() - 0.2   # smooth Gaussian-like falloff
