"""Integration tests: the recent modules are FACULTIES of UnifiedMind, proven end to end.

The integration plan's hardest lesson (its section 6): naive cross-module chaining REGRESSED once --
a denoiser fed a recall output dropped cosine, because a shared *kernel* is not a shared *manifold*.
So wiring is not proven by an import check; it is proven by running a cross-faculty pipeline THROUGH
UnifiedMind and asserting it actually works, with each hop's prior matched to its input.

This file covers the Tier 1 faculties -- decompose_signal / denoise / fit_function -- the DECOMPOSE /
DENOISE / FIT half of the loop. Each tier of wiring lands with at least one pipeline test here.
"""
import os
import tempfile

import numpy as np
import pytest

from holographic_unified import UnifiedMind
from holographic_symbolic import Formula


def _rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---- the faculties are actually on the mind (not a separate drawer of experiments) --------------
def test_unified_exposes_the_tier1_faculties():
    m = UnifiedMind(dim=256, seed=0)
    for name in ("decompose_signal", "denoise", "fit_function"):
        assert callable(getattr(m, name)), f"UnifiedMind is missing faculty {name!r}"


# ---- decompose_signal: foreign signal -> a savable, realizable generating law -------------------
def test_decompose_signal_recovers_periodic_law_and_seed_roundtrips():
    m = UnifiedMind(dim=256, seed=0)
    x = np.linspace(0, 4 * np.pi, 240)
    y = np.sin(x) + 0.3 * np.cos(2 * x)                 # a ring (periodic) law

    f, info = m.decompose_signal(x, y)
    assert info["topology"] == "ring"
    assert info["mode"] == "additive"
    assert info["n_terms"] <= 4 and info["resid_rms"] < 1e-2     # recovered the law tightly
    assert info["compression_ratio"] > 1.0                       # the seed is smaller than the samples

    # the Formula IS a realizable seed: save -> load -> generate must be BIT-identical
    seedpath = os.path.join(tempfile.mkdtemp(), "seed.json")
    f.save(seedpath)
    assert np.max(np.abs(Formula.load(seedpath).generate(x) - f.generate(x))) < 1e-12

    # and it extrapolates PERIODICALLY (bounded) instead of diverging the way a polynomial would
    xe = np.linspace(4 * np.pi, 6 * np.pi, 120)
    assert np.max(np.abs(f.generate(xe))) < 3.0


def test_decompose_signal_auto_selects_multiplicative_for_a_power_law():
    m = UnifiedMind(dim=256, seed=0)
    x = np.linspace(1.0, 6.0, 150)
    y = 2.0 * x ** 1.5                                  # a multiplicative law on a LINE domain (y > 0)

    f, info = m.decompose_signal(x, y)
    assert info["topology"] == "line" and info["mode"] == "multiplicative"
    assert info["resid_rms"] < 1e-6                     # log transform turns the power law additive

    # single-array shorthand: decompose_signal(y) fits the signal on a unit index grid and still runs
    f2, info2 = m.decompose_signal(y)
    assert set(info2) >= {"topology", "mode", "resid_rms", "compression_ratio"}
    assert np.isfinite(f2.generate(np.arange(len(y), dtype=float))).all()


# ---- THE PIPELINE the plan asks for: decompose -> save -> realize -> denoise, through ONE mind ---
def test_end_to_end_decompose_save_realize_denoise_pipeline():
    """detect topology -> decompose_signal -> seed.save -> realize (reload + generate) -> denoise.
    The denoise hop runs on a manifold that CONTAINS the regenerated signal (compatible prior), so the
    chain is honest, and the end result must IMPROVE on the noisy input -- no silent regression."""
    m = UnifiedMind(dim=256, seed=0)
    x = np.linspace(0, 4 * np.pi, 240)
    clean = np.sin(x) + 0.4 * np.sin(3 * x)             # only ODD harmonics -> an antiperiodic (mobius) law

    # decompose into a law and persist the seed. (This signal is purely odd-harmonic, so the topology
    # detector correctly calls it mobius -- it decomposes onto the ODD-harmonic basis. Either periodic
    # class extrapolates on its manifold; the pipeline is what we are testing, not the class.)
    f, info = m.decompose_signal(x, clean)
    assert info["topology"] in ("ring", "mobius") and info["resid_rms"] < 1e-2
    seedpath = os.path.join(tempfile.mkdtemp(), "law.json")
    f.save(seedpath)

    # realize: reload the seed and regenerate the signal it encodes
    regenerated = Formula.load(seedpath).generate(x)
    assert _rms(regenerated, clean) < 1e-2              # the law reproduces the signal

    # denoise a noisy copy on the signal's OWN harmonic family (a manifold that contains it)
    fam = np.stack([np.sin(x + p) + 0.4 * np.sin(3 * (x + p))
                    for p in np.linspace(0, 2 * np.pi, 64)])
    rng = np.random.default_rng(0)
    noisy = regenerated + 0.8 * rng.standard_normal(len(x))
    den = m.denoise(noisy, method="adaptive", samples=fam)

    assert _rms(den, clean) < _rms(noisy, clean) - 0.2  # the pipeline END materially helps (no regress)


# ---- denoise: routing, the honest "needs a prior" refusal, and the codebook map -----------------
def test_denoise_routes_and_requires_a_prior():
    m = UnifiedMind(dim=256, seed=0)

    # auto with no prior is an honest refusal -- a denoiser is a map of a manifold; a lone vector
    # has none, so there is no free lunch to silently take
    with pytest.raises(ValueError):
        m.denoise(np.zeros(32))

    # codebook route: a noisy atom should move TOWARD its codebook entry
    rng = np.random.default_rng(0)
    cb = rng.standard_normal((8, 64))
    cb /= np.linalg.norm(cb, axis=1, keepdims=True)
    noisy_atom = cb[3] + 0.5 * rng.standard_normal(64)
    cleaned = m.denoise(noisy_atom, method="codebook", codebook=cb)
    assert _cos(cleaned, cb[3]) > _cos(noisy_atom, cb[3])


def test_denoise_helps_at_high_noise_on_real_low_rank_data():
    """Grounding on real SOL price windows -- the same low-rank manifold the denoise module measured.
    At high noise, projecting onto the manifold removes off-manifold noise (the measured win)."""
    px = np.load("data/sol_5min.npz")["px"].astype(float)
    W, step = 64, 16
    wins = np.stack([px[i:i + W] for i in range(0, len(px) - W, step)])
    wins = (wins - wins.mean(1, keepdims=True)) / (wins.std(1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0); rng.shuffle(wins)
    train, test = wins[:600], wins[600:750]

    m = UnifiedMind(dim=256, seed=0)
    snr = lambda c, e: 10 * np.log10(np.var(c) / (np.mean((c - e) ** 2) + 1e-12))
    rng2 = np.random.default_rng(1)
    gains = []
    for c in test:
        n = c + 0.8 * rng2.standard_normal(len(c))
        d = m.denoise(n, method="adaptive", samples=train)
        gains.append(snr(c, d) - snr(c, n))
    assert np.mean(gains) > 1.5                          # solid denoising at high noise, on real data


# ---- fit_function: interpretable additive fit, with its interaction boundary kept on record ------
def test_fit_function_recovers_additive_parts_and_shows_interaction_limit():
    m = UnifiedMind(dim=512, seed=0)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (1400, 2))
    g1 = lambda t: np.sin(2 * np.pi * t)
    g2 = lambda t: 4 * (t - 0.5) ** 2
    y = g1(X[:, 0]) + g2(X[:, 1]) + 0.02 * rng.standard_normal(1400)

    k = m.fit_function(X[:1000], y[:1000])
    r2 = 1 - np.sum((y[1000:] - k.predict(X[1000:])) ** 2) / np.sum((y[1000:] - y[1000:].mean()) ** 2)
    assert r2 > 0.98
    ts = np.linspace(0.05, 0.95, 40)
    assert abs(np.corrcoef(k.feature_function(0, ts), g1(ts))[0, 1]) > 0.95   # psi_1 recovers sin

    # KEPT NEGATIVE (surfaced, not hidden): the additive form cannot fit an interaction
    yprod = (2 * X[:, 0] - 1) * (2 * X[:, 1] - 1) + 0.02 * rng.standard_normal(1400)
    kp = m.fit_function(X[:1000], yprod[:1000])
    r2p = 1 - np.sum((yprod[1000:] - kp.predict(X[1000:])) ** 2) / np.sum(
        (yprod[1000:] - yprod[1000:].mean()) ** 2)
    assert r2p < 0.3                                     # the boundary, shown

    # 1-D feature shorthand: a lone feature vector is accepted as one column
    k1 = m.fit_function(X[:500, 0], y[:500])
    assert k1.predict(X[:5, 0].reshape(-1, 1)).shape == (5,)


# ============================ Tier 2 -- the factor_composite de-siloing ============================
# The integration plan's "real de-siloing": the higher-capacity SBC resonator gets a first-class mind
# faculty (decompose_structure), and factor_composite becomes ONE entry point that delegates to it for
# SBC problems while keeping -- and deprecating -- the legacy dense MAP path (a different algebra the SBC
# resonator cannot factor, so it is delegated-past, not deleted).

def test_decompose_structure_faculty_factors_and_verifies():
    from holographic_sbc import sbc_codebook, sbc_reconstruct
    m = UnifiedMind(dim=256, seed=0)
    B, L = 16, 16
    cbs = [sbc_codebook(B, L, 10, seed=k) for k in range(3)]
    true = (2, 5, 8)
    P = sbc_reconstruct(true, cbs, L)

    out = m.decompose_structure(P, cbs, L)
    assert tuple(out["picks"]) == true and out["verified"]
    assert np.array_equal(sbc_reconstruct(out["picks"], cbs, L), P)   # the recipe rebuilds the structure
    assert out["present"] == [True, True, True]                       # all three factors are present


def test_factor_composite_routes_to_sbc_and_matches_decompose_structure():
    from holographic_sbc import sbc_codebook, sbc_reconstruct, sbc_identity
    m = UnifiedMind(dim=256, seed=0)
    B, L = 16, 16
    cbs = [sbc_codebook(B, L, 10, seed=k) for k in range(3)]
    true = (2, 5, 8)
    P = sbc_reconstruct(true, cbs, L)

    # one entry point, given an L, takes the SBC path and agrees with the canonical faculty
    fc = m.factor_composite(P, cbs, L=L)
    assert fc["backend"] == "sbc"
    assert fc["solved"] and fc["factors"] == true
    assert fc["factors"] == tuple(m.decompose_structure(P, cbs, L)["picks"])

    # presence detection survives the routing: an identity factor is reported ABSENT
    cbs2 = [list(cbs[0]) + [sbc_identity(B)], cbs[1], cbs[2]]
    P_absent = sbc_reconstruct((len(cbs2[0]) - 1, 4, 6), cbs2, L)      # factor 0 = identity
    fa = m.factor_composite(P_absent, cbs2, L=L)
    assert fa["present"][0] is False and fa["present"][1] and fa["present"][2]


def test_factor_composite_dense_path_is_backward_compatible_and_deprecated():
    """The pinned legacy contract: a dense MAP/bipolar composite still factors through factor_composite
    (the SBC resonator cannot do this algebra, so the path is kept) -- and it now warns, steering new
    code to the SBC factorizer."""
    import warnings
    from holographic_resonator import map_codebook, map_bind
    m = UnifiedMind(dim=256, seed=0)
    books = [map_codebook(40, 1500, s) for s in range(3)]
    rng = np.random.default_rng(4)
    true = [int(rng.integers(40)) for _ in range(3)]
    c = map_bind(*[books[f][true[f]] for f in range(3)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = m.factor_composite(c, books, restarts=30)
        assert any(issubclass(x.category, DeprecationWarning) for x in caught)  # deprecation steers callers
    assert r["solved"] and r["factors"] == tuple(true) and r["backend"] == "dense"


# =================== Tier 2 (items 4 & 6) -- decode_structure (peel) + energy cleanup ===============
# Item 4: the B8 per-peel decode exposed as the inverse of the B7 chain typed structure, run through
# the mind (build -> realize -> decode). Item 6: the B1 dense-Hopfield cleanup wired as an opt-in flag
# on the core cleanup, pinned bit-for-bit to argmax at high beta.

def test_decode_structure_round_trips_a_chain_through_the_mind():
    m = UnifiedMind(dim=512, seed=1)
    recipe, nodes = m.chain_structure(16)
    M = m.realize(recipe)                                  # B7 forward: realize the chain memory
    n_correct = lambda seq: sum(1 for h, i in enumerate(seq) if i == h + 1)

    hard = m.decode_structure(M, nodes, cleanup="hard")
    soft = m.decode_structure(M, nodes, cleanup="soft")
    raw = m.decode_structure(M, nodes, cleanup=None)

    assert n_correct(hard) == 15                           # per-peel cleanup decodes every hop
    assert n_correct(soft) == 15                           # soft ties hard on discrete pointers (B1 negative)
    assert n_correct(raw) <= 3                              # KEPT NEGATIVE: raw decode craters (noise compounds)
    assert hard == list(range(1, 16))                      # the exact recovered sequence


def test_energy_cleanup_is_opt_in_and_matches_argmax_at_high_beta():
    from holographic_ai import Vocabulary, random_vector
    v = Vocabulary(512, seed=2)
    for nm in ("alpha", "beta", "gamma", "delta", "epsilon"):
        v.get(nm)
    rng = np.random.default_rng(0)
    noisy = v.get("gamma") + 0.7 * random_vector(512, rng)

    plain = v.cleanup(noisy)                               # default off -- the existing decision
    energy_hi = v.cleanup(noisy, energy=True, beta=1e6)    # beta->inf softmax is one-hot == argmax
    assert energy_hi[0] == plain[0]                        # bit-for-bit the same identity (B1 guarantee)

    # the flag is genuinely opt-in: a clean atom is recovered with or without it
    assert v.cleanup(v.get("delta"))[0] == "delta"
    assert v.cleanup(v.get("delta"), energy=True)[0] == "delta"


# ============================== Tier 3 -- search & dynamics faculties ===============================
# Min-cost search (a maze; a fragment assembly) and learned linear dynamics, as faculties of the mind.
# The assembly result comes back as a B7 typed structure the mind can realize; dynamics is a bind.

def test_solve_maze_faculty_finds_the_optimal_path_deterministically():
    from holographic_creature import GridWorld
    m = UnifiedMind(dim=256, seed=0)
    w = GridWorld(16, 16, maze=True, fixed_seed=7, braid=1.0)
    path, info = m.solve_maze(w)
    assert info["reached"] and info["deterministic"] is True
    assert info["extracted_len"] == info["optimal"]       # flow collapses onto the shortest tube
    # deterministic: a second solve gives the identical path
    path2, _ = m.solve_maze(GridWorld(16, 16, maze=True, fixed_seed=7, braid=1.0))
    assert path == path2


def test_assemble_faculty_is_optimal_and_a_realizable_typed_structure():
    from holographic_assembly import assemble_optimal_energy
    from holographic_typed import op_kinds
    m = UnifiedMind(dim=256, seed=0)
    target = "ABCABCABCA"
    full = sorted({target[p:p + 2] for p in range(len(target) - 1)})

    out = m.assemble(target, full)
    assert out["assembled"] == target and out["energy"] == 0

    # the result IS a B7 typed structure: only the typed op-kinds, and the mind can realize it
    assert op_kinds(out["recipe"]) <= {"atom", "bind", "bundle", "superpose", "permute", "raw", "normalize"}
    assert m.realize(out["recipe"]).shape == (256,)        # realizes to a hypervector at the mind's dim

    # forced to mismatch, the flow assembly still attains the GLOBAL optimum (matches the exact DP)
    lib = sorted((set(full) - {"CA"}) | {"AA", "BB", "CC"})
    o2 = m.assemble(target, lib)
    assert o2["energy"] == assemble_optimal_energy(target, lib, frag_len=2) and o2["energy"] > 0


def test_learn_dynamics_faculty_predicts_bind_shaped_and_round_trips():
    from holographic_ai import bind, cosine, random_vector
    rng = np.random.default_rng(0)
    U = random_vector(256, rng)
    s = random_vector(256, rng)
    traj = [s]
    for _ in range(400):
        s = bind(U, s) + 0.01 * rng.standard_normal(256)
        s /= np.linalg.norm(s)
        traj.append(s)
    traj = np.array(traj)

    m = UnifiedMind(dim=256, seed=0)
    prop = m.learn_dynamics(traj[:300])

    # step is LITERALLY a bind with the learned operator, and it predicts bind-shaped dynamics
    assert np.allclose(prop.step(traj[310]), bind(prop.U, traj[310]))
    pred = np.mean([cosine(prop.step(traj[300 + i]), traj[301 + i]) for i in range(80)])
    persist = np.mean([cosine(traj[300 + i], traj[301 + i]) for i in range(80)])
    assert pred > 0.9 and pred > persist + 0.2

    # the durable win: the trajectory is content-addressable -- forward k then back k returns the start
    x = traj[350]
    assert cosine(x, prop.recall_at(prop.rollout(x, 4)[-1], 4)) > 0.99


# =================== Tier 4 -- persistence (rd) + the generative faculties ==========================
# Item 9: the mind is persistable via the kernel save (so quant='rd' applies), round-tripping its
# LEARNED generalization (classify + decide identical). Item 10: vector generation as denoising-from-
# noise (B10), and a 2-D field as a superposition of Gaussian splats (B8).

def test_unified_mind_save_load_round_trips_classify_and_decide():
    import os
    import tempfile
    m = UnifiedMind(dim=256, seed=0, maintain="manual")
    rng = np.random.default_rng(0)
    for _ in range(20):                                    # two clean numeric classes
        m.learn(round(float(rng.uniform(0, 1)), 3), "small", modality="number")
        m.learn(round(float(rng.uniform(5, 6)), 3), "big", modality="number")
    m.actions(["left", "right"])                           # and a decision brain
    for _ in range(30):
        s = round(float(rng.uniform(0, 1)), 3)
        m.reinforce(s, "left" if s < 0.5 else "right", 1.0, modality="number")

    probes = [round(float(rng.uniform(0, 6)), 3) for _ in range(15)]
    before_cls = [m.classify(p, modality="number")[0] for p in probes]
    before_dec = [m.decide(p, modality="number") for p in probes]

    path = os.path.join(tempfile.mkdtemp(), "mind")
    m.save(path, quant="rd")                               # the B5 rate-distortion save level, via the mind
    m2 = UnifiedMind.load(path)

    assert [m2.classify(p, modality="number")[0] for p in probes] == before_cls  # classify identical
    assert [m2.decide(p, modality="number") for p in probes] == before_dec       # decide identical

    # documented boundary: the verbatim recall index of individuals is NOT persisted (re-learn for it)
    import pytest
    with pytest.raises(RuntimeError):
        m2.recall(0.5, modality="number")


def test_generate_vector_faculty_lands_on_the_manifold():
    from holographic_ai import random_vector, cosine
    m = UnifiedMind(dim=256, seed=0)
    rng = np.random.default_rng(0)
    codebook = np.stack([random_vector(256, rng) for _ in range(8)])
    g = m.generate_vector(codebook, seed=3)
    # denoising from pure noise walks onto a stored pattern (B10); over a bare codebook that is a
    # stored atom -- the kept-negative degenerate regime, which is exactly what we assert here
    assert max(cosine(g, codebook[i]) for i in range(8)) > 0.99
    # deterministic in the seed
    assert np.allclose(g, m.generate_vector(codebook, seed=3))


def test_splat_field_faculty_reconstructs_and_denoises():
    from holographic_splat import psnr
    G = 48
    ys, xs = np.mgrid[0:G, 0:G]
    rng = np.random.default_rng(0)
    T = np.zeros((G, G))
    for _ in range(4):
        cy, cx, s, a = rng.uniform(8, G - 8, 2).tolist() + [rng.uniform(3, 7), rng.uniform(0.5, 1)]
        T += a * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * s * s))
    T /= T.max()

    splats, rendered = UnifiedMind(dim=128, seed=0).splat_field(T, k=40)
    assert len(splats) == 40 and psnr(T, rendered) > 25.0   # superposition of primitives reconstructs

    noisy = T + 0.10 * rng.standard_normal(T.shape)
    clean = UnifiedMind(dim=128, seed=0).splat_field(noisy, k=30, denoise=True)
    assert psnr(T, clean) > psnr(T, noisy) + 1.0            # the splat fit denoises (no capacity for noise)


# ============== Wiring-check follow-ups: axial perception + the splat-bundle archive ===============
# Boundary 1: holographic_mobius's AxialEncoder wired as the "axial" modality (theta == theta+pi).
# Boundary 2: a splat-bundle image archive beside the WHT plates, + the addendum's splat_bundle/recall_region.

def test_axial_modality_treats_theta_and_theta_plus_pi_as_the_same():
    import math
    m = UnifiedMind(dim=512, seed=0)
    t = 0.7
    assert m.axial_similarity(t, t + math.pi) > 0.99          # a pi flip is the SAME orientation
    assert m.axial_similarity(t, t + math.pi / 2) < 0.5       # an orthogonal orientation is dissimilar
    # decode is mod pi: theta and theta+pi both read back as the same angle in [0, pi)
    assert abs(m.decode_axial(m.perceive(1.2, "axial")) - 1.2) < 0.05
    assert abs(m.decode_axial(m.perceive(1.2 + math.pi, "axial")) - 1.2) < 0.05

    # learn / classify over orientations: an A reported as a pi flip still classifies as A
    rng = np.random.default_rng(0)
    for _ in range(15):
        m.learn(float(rng.uniform(0.0, 0.4)), "A", modality="axial")     # cluster near 0.2
        m.learn(float(rng.uniform(1.2, 1.6)), "B", modality="axial")     # cluster near 1.4
    assert m.classify(0.2 + math.pi, modality="axial")[0] == "A"         # flip-invariant classification


def test_splat_archive_reconstructs_refines_and_region_queries():
    from holographic_archive import _gallery
    from holographic_splat import psnr
    imgs = _gallery(48)
    K = 120
    m = UnifiedMind(dim=128, seed=0)
    arch = m.splat_archive((48, 48, 3), keep=K)
    for im in imgs:
        arch.add(im)

    # reconstruction is reasonable, and a PREFIX is a coarser-but-valid preview (progressive refinement)
    full = np.mean([psnr(imgs[i], arch.recover(i)) for i in range(arch.n)])
    quarter = np.mean([psnr(imgs[i], arch.recover(i, k=K // 4)) for i in range(arch.n)])
    assert full > 15.0 and full > quarter + 1.0               # quality rises with k (importance order)

    # EXACT region query: every returned splat's centre lies inside the requested box
    box = (0, 24, 0, 24)
    here, _patch = arch.region(0, box)
    for chan in here:
        assert all(0 <= cy < 24 and 0 <= cx < 24 for (cy, cx, _a, _s) in chan)

    # content recall on a noisy copy lands on the right image
    noisy = imgs[3] + 0.05 * np.random.default_rng(0).standard_normal(imgs[3].shape)
    assert arch.recall(noisy)[0] == 3


def test_splat_bundle_is_a_superposition_carrying_region_signal():
    from holographic_splat import splat_fit, splat_bundle, recall_region
    # a single blob in the TOP-LEFT, empty bottom-right
    G = 48
    ys, xs = np.mgrid[0:G, 0:G]
    T = np.exp(-((ys - 8) ** 2 + (xs - 8) ** 2) / (2 * 5.0 ** 2))
    T = T / T.max()
    splats = splat_fit(T, 10)

    scene, ctx = splat_bundle(splats, T.shape, dim=4096, grid=4)
    assert scene.shape == (4096,)
    # content-addressable region read: the OCCUPIED region recalls more energy than an EMPTY one.
    # (Only a coarse signal -- exact per-splat recovery is the archive's job; this is the VSA cliff.)
    occupied = recall_region(scene, (0, 0), ctx)              # top-left: has the blob
    empty = recall_region(scene, (3, 3), ctx)                 # bottom-right: empty
    assert occupied > empty


# ============== Honesty layer woven into recognition (RecallNull / SPRT / bh_fdr as core) ===========
# RecallNull/SPRT/bh_fdr were a standalone measurement harness; these prove they are now part of how the
# mind RECOGNISES -- calibrated confidence, honest abstention, sequential decision, FDR-controlled batch.

def _animal_mind():
    m = UnifiedMind(dim=512, seed=0)
    for w in ["dog", "wolf", "puppy", "hound"]:
        m.learn(w, "canine")
    for w in ["cat", "lion", "kitten", "tiger"]:
        m.learn(w, "feline")
    for w in ["oak", "pine", "maple", "birch"]:
        m.learn(w, "tree")
    return m


def test_recognize_is_calibrated_and_classify_can_abstain():
    m = _animal_mind()
    # a learned member: low false-alarm p; gibberish: clearly higher p
    _lab, _sim, p_real = m.recognize("dog")
    _lab2, _sim2, p_noise = m.recognize("qz xkqv zzpf")
    assert p_real < 0.05 and p_noise > p_real + 0.2
    # default classify ALWAYS names a nearest label (backward compatible, returns a (label, score) tuple)
    lab, score = m.classify("qz xkqv zzpf")
    assert lab is not None and isinstance(score, float)
    # with abstain: None label on noise, real label on a member -- both keep the (label, score) shape
    none_lab, _ = m.classify("qz xkqv zzpf", abstain=0.05)
    real_lab, _ = m.classify("dog", abstain=0.05)
    assert none_lab is None and real_lab == "canine"


def test_stream_recognize_decides_match_for_real_and_reject_for_noise():
    m = _animal_mind()
    dec_real, lab_real, _n1 = m.stream_recognize(["dog", "hound", "puppy", "wolf"])
    dec_noise, _lab, _n2 = m.stream_recognize(["qz1 zz", "zxq ww", "vbn qq", "wqp kk"])
    assert dec_real == "MATCH" and lab_real == "canine"
    assert dec_noise == "REJECT"


def test_recognize_batch_controls_false_discovery():
    m = _animal_mind()
    out = m.recognize_batch(["dog", "tiger", "oak", "zzqx vvbn wqlk"], alpha=0.1)
    sig = {r["label"]: r["significant"] for r in out[:3]}        # first three are real members
    assert all(sig.values())                                     # learned members survive FDR
    assert out[3]["significant"] is False                        # the gibberish does not


def test_recall_can_abstain_on_unseen_inputs():
    m = _animal_mind()
    # a stored individual recalls with a tiny false-alarm p; gibberish sits above the abstain threshold
    _pay, _sim, p_real = m.recall_calibrated("dog")
    _pay2, _sim2, p_noise = m.recall_calibrated("zzqx vvbn wqlk")
    assert p_real < 0.05 < p_noise                               # they straddle the abstention level
    # default recall is unchanged (returns (payload, score)); abstain returns the payload or None
    assert m.recall("dog") is not None
    assert m.recall("dog", abstain=0.05) is not None
    assert m.recall("zzqx vvbn wqlk", abstain=0.05) is None


# ============== Coherence-gated maintenance (the measured win from the calibrated-novelty study) ======
# Calibrated NOVELTY (the originally-flagged idea) was a NEGATIVE -- novelty detects "matches nothing",
# but reorganization's value is fixing INCOHERENCE, which novelty cannot see. The study that disproved it
# found COHERENCE-gated reorganization instead: reorganize only when the store is actually incoherent, not
# on a fixed clock. This test pins the win at the mind level -- fewer reorganize passes at comparable
# accuracy -- and that the default (coherence_floor=None) still reorganizes on the fixed schedule.

def _shift_stream(seed=0):
    """Antipodal-bimodal classes (a single prototype is useless, only a SPLIT classifies), with two NEW
    classes arriving mid-stream and then a long STABLE coherent tail where a fixed schedule keeps paying
    for reorganize passes the gate can skip."""
    import numpy as np
    rng = np.random.default_rng(seed)
    L, NC, MODES = 24, 4, 2
    ang = np.linspace(0, 2 * np.pi, NC * MODES, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1) @ rng.standard_normal((2, L))
    csub = {c: [c + NC * m for m in range(MODES)] for c in range(NC)}
    samp = lambda c: dirs[csub[c][rng.integers(MODES)]] * 3 + 0.5 * rng.standard_normal(L)
    rows = []
    for _ in range(90):
        c = int(rng.integers(2)); rows.append((samp(c), c))          # phase 1: 2 classes
    for _ in range(90):
        c = int(rng.integers(NC)); rows.append((samp(c), c))         # shift: 2 new classes join
    for _ in range(150):
        c = int(rng.integers(NC)); rows.append((samp(c), c))         # stable, coherent tail
    return rows


def _run_maintained(coherence_floor, maintain="auto", check_every=30):
    import numpy as np
    rows = _shift_stream(0)
    m = UnifiedMind(dim=384, seed=0, check_every=check_every,
                    coherence_floor=coherence_floor, maintain=maintain)
    correct = []
    for i, (x, c) in enumerate(rows):
        pred = m.classify(x, modality="vector")[0] if m.memory.live.size() else None
        correct.append(pred == c)
        m.learn(x, c, modality="vector")
    return float(np.mean(correct)), len(m.journal), len(rows)        # journal length == reorganize passes


def test_coherence_gate_reorganizes_less_than_schedule_at_comparable_accuracy():
    sched_acc, sched_passes, total = _run_maintained(None)           # default: fixed schedule
    gated_acc, gated_passes, _ = _run_maintained(0.65)               # opt-in coherence gate
    floor_acc, _, _ = _run_maintained(None, maintain="manual")       # never reorganize (the floor)
    # the gate reorganizes FEWER times than the schedule (but does reorganize) ...
    assert 2 <= gated_passes < sched_passes
    # ... at accuracy comparable to the schedule ...
    assert gated_acc >= sched_acc - 0.15
    # ... and well above the never-reorganize floor (so it reorganizes USEFULLY, not idly) ...
    assert gated_acc >= floor_acc + 0.12
    # ... while the DEFAULT (coherence_floor=None) reorganizes on the fixed schedule, unchanged.
    assert sched_passes == total // 30


def test_coherence_floor_survives_save_and_reload():
    import tempfile, os
    m = UnifiedMind(dim=256, seed=0, coherence_floor=0.6)
    m.learn("dog", "canine"); m.learn("cat", "feline")
    p = os.path.join(tempfile.mkdtemp(), "mind")
    m.save(p)
    assert UnifiedMind.load(p).coherence_floor == 0.6                 # config round-trips


# ============== Tier-0 panel fixes: sublinear+calibrated recall, coverage, rd-in-auto ================
# Pharr (sublinear recall_calibrated), Cranmer (calibration coverage), Duda (auto save uses B5 rd).

def test_recall_calibrated_uses_the_same_path_as_recall_and_can_abstain():
    m = UnifiedMind(dim=512, seed=0)
    for w in ["dog", "wolf", "puppy", "hound", "cat", "lion", "kitten", "tiger", "oak", "pine", "maple"]:
        m.learn(w, "animal" if w not in ("oak", "pine", "maple") else "tree")
    # recall_calibrated now routes the winner through recall() itself (the forest on a big store, the exact
    # scan on a small one) instead of its own exact scan -- so the two agree on the winner and the score.
    pay_r, sim_r = m.recall("dog")
    pay_c, sim_c, p_real = m.recall_calibrated("dog")
    assert pay_r == pay_c and abs(sim_r - sim_c) < 1e-9
    _pay, _sim, p_noise = m.recall_calibrated("zzqx vvbn wqlk")
    assert p_real < 0.05 < p_noise                                # stored vs noise straddle the threshold
    assert m.recall("dog", abstain=0.05) is not None
    assert m.recall("zzqx vvbn wqlk", abstain=0.05) is None


def test_recognition_p_values_are_calibrated_on_noise():
    m = UnifiedMind(dim=512, seed=0)
    for w in ["dog", "wolf", "puppy", "hound", "cat", "lion", "kitten", "tiger",
              "oak", "pine", "maple", "birch", "ash", "elm", "fox", "bear"]:
        m.learn(w, "animal" if w not in ("oak", "pine", "maple", "birch", "ash", "elm") else "tree")
    rep = m.calibration_report(n=3000)
    # a calibrated detector fires on pure noise at ~= alpha, on BOTH the prototype and individual paths.
    for path in ("prototype_false_alarm", "individual_false_alarm"):
        assert 0.02 <= rep[path][0.05] <= 0.10                    # ~5% false-alarm at alpha=0.05
        assert 0.05 <= rep[path][0.1] <= 0.17                     # ~10% at alpha=0.10


def test_auto_save_uses_rate_distortion_for_large_low_rank_arrays():
    import numpy as np
    from holographic_ratedistortion import (geometry_preserving_code, pack_code, unpack_code,
                                             reconstruct, bits_per_vector)
    rng = np.random.default_rng(0)
    rows, cols, rank = 512, 256, 8
    A = rng.standard_normal((rows, rank)) @ rng.standard_normal((rank, cols))   # large + genuinely low-rank
    code = geometry_preserving_code(A, target_cos=0.9999)
    # this is the decision auto now makes: take rd only when it beats int8 (8 bits/value)
    assert bits_per_vector(code) < 8 * cols
    B = reconstruct(unpack_code(pack_code(code)))                 # full pack -> unpack -> reconstruct
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    assert float(np.sum(An * Bn, axis=1).min()) >= 0.998         # decision-safe (cosines preserved)


def test_default_auto_save_round_trips_a_mind_identically():
    import tempfile, os
    import numpy as np
    m = UnifiedMind(dim=256, seed=0, maintain="manual")
    rng = np.random.default_rng(0)
    for _ in range(20):
        m.learn(round(float(rng.uniform(0, 1)), 3), "small", modality="number")
        m.learn(round(float(rng.uniform(5, 6)), 3), "big", modality="number")
    probe = [round(float(rng.uniform(0, 6)), 3) for _ in range(12)]
    before = [m.classify(p, modality="number")[0] for p in probe]
    p = os.path.join(tempfile.mkdtemp(), "mind")
    m.save(p)                                                     # default quant='auto', now rd-aware
    after = [UnifiedMind.load(p).classify(p2, modality="number")[0] for p2 in probe]
    assert before == after


# ============== Tier-1: calibrated decide -- honesty from perception to ACTION (Togelius) ============

def _taught_action_mind(dim=512, seed=0):
    import numpy as np
    import holographic_ai as A
    m = UnifiedMind(dim=dim, seed=seed)
    m.actions(["N", "S", "E", "W"])
    rng = np.random.default_rng(seed)
    archetypes = [A.random_vector(dim, rng) for _ in range(4)]
    best = ["N", "E", "S", "W"]
    for _ in range(40):
        for k, base in enumerate(archetypes):
            s = base + 0.25 * A.random_vector(dim, rng); s /= np.linalg.norm(s)
            m.reinforce(s, best[k], 1.0)                   # this action paid off in situations like this
            m.reinforce(s, best[(k + 1) % 4], -0.5)        # a bad alternative, so values differ
    return m, archetypes, best, rng


def test_decide_confidence_is_low_for_familiar_states_and_high_for_novel_ones():
    import numpy as np
    import holographic_ai as A
    m, archetypes, best, rng = _taught_action_mind()
    fam = archetypes[0] + 0.25 * A.random_vector(512, rng); fam /= np.linalg.norm(fam)
    nov = A.random_vector(512, rng)
    act_f, p_f = m.decide_confidence(fam)
    act_n, p_n = m.decide_confidence(nov)
    assert act_f == best[0]                                 # familiar -> the learned-good action
    assert p_f < 0.1 < p_n                                  # familiar vs novel straddle the threshold


def test_brain_recognition_null_is_calibrated_on_noise():
    import numpy as np
    m, _arch, _best, _rng = _taught_action_mind()
    null = m._brain_null()
    rng = np.random.default_rng(7)
    sup = np.array([max(m._brain.value(s, a)[1] for a in range(4))
                    for s in (rng.standard_normal((3000, 512)) /
                              np.linalg.norm(rng.standard_normal((3000, 512)), axis=1, keepdims=True))])
    ps = np.array([null.pvalue(x) for x in sup])
    assert 0.02 <= float(np.mean(ps <= 0.05)) <= 0.10      # the action-side detector tracks alpha too
    assert 0.05 <= float(np.mean(ps <= 0.10)) <= 0.17


def test_explore_if_unrecognized_guesses_randomly_on_novel_states_and_commits_on_familiar():
    import numpy as np
    import holographic_ai as A
    m, archetypes, best, rng = _taught_action_mind()
    fam = archetypes[0] + 0.25 * A.random_vector(512, rng); fam /= np.linalg.norm(fam)
    nov = A.random_vector(512, rng)
    novel_actions = {m.decide(nov, explore_if_unrecognized=0.1) for _ in range(100)}
    fam_actions = {m.decide(fam, explore_if_unrecognized=0.1) for _ in range(100)}
    assert len(novel_actions) >= 3                          # unrecognized -> safe random move (guessing)
    assert fam_actions == {best[0]}                         # recognized -> commits to the trusted action


# ============== Tier-0: the SPRT earns its keep on OVERLAPPING densities (Wald sample-savings) ==========

def test_sprt_spends_more_samples_as_densities_overlap_and_beats_fixed_n():
    import numpy as np
    from holographic_honesty import SPRTRecall
    def avg_n_err(mu0, sd0, mu1, sd1, trials=1500, cap=60):
        null = np.random.default_rng(1).normal(mu0, sd0, 3000)
        match = np.random.default_rng(2).normal(mu1, sd1, 3000)
        g = np.random.default_rng(5); ns, ok = [], 0
        for t in range(trials):
            ms = (t % 2 == 0); mu, sd = (mu1, sd1) if ms else (mu0, sd0)
            d, n = SPRTRecall(null, match, alpha=0.05, beta=0.05).decide(g.normal(mu, sd, cap), cap=cap)
            ns.append(n); ok += (d == ("MATCH" if ms else "REJECT"))
        return float(np.mean(ns)), 1.0 - ok / trials
    sep_n, _ = avg_n_err(0.10, 0.04, 0.45, 0.14)          # well-separated -> decisive in ~1 sample
    ovl_n, ovl_err = avg_n_err(0.35, 0.13, 0.52, 0.13)    # heavy overlap -> several samples
    assert sep_n < 1.5 < ovl_n                            # the SPRT spends more samples when densities overlap
    # at matched error, the smallest fixed-window rule uses MORE samples than the SPRT's average
    thresh = (0.35 + 0.52) / 2.0
    def fixedN_err(N, trials=1500):
        h = np.random.default_rng(11); bad = 0
        for t in range(trials):
            ms = (t % 2 == 0); mu, sd = (0.52, 0.13) if ms else (0.35, 0.13)
            bad += ((float(np.mean(h.normal(mu, sd, N))) >= thresh) != ms)
        return bad / trials
    fixedN = next((N for N in range(1, 40) if fixedN_err(N) <= ovl_err + 0.01), None)
    assert fixedN is not None and fixedN > ovl_n          # Wald uses fewer samples for the same error


# ============== Tier-0: auto-calibrated coherence floor -- relative drop, no absolute threshold ==========

def test_auto_coherence_floor_matches_the_hand_set_floor_without_an_absolute_threshold():
    sched_acc, sched_passes, total = _run_maintained(None)           # fixed schedule
    auto_acc, auto_passes, _ = _run_maintained('auto')               # auto relative-drop floor (90% of peak)
    floor_acc, _, _ = _run_maintained(None, maintain="manual")       # never reorganize (the floor)
    assert 2 <= auto_passes < sched_passes                           # reorganizes less than the schedule, but does
    assert auto_acc >= sched_acc - 0.15                              # at accuracy comparable to the schedule
    assert auto_acc >= floor_acc + 0.12                              # usefully above never-reorganizing
    # the 'auto' sentinel round-trips through save/load exactly like a numeric floor
    import tempfile, os
    m = UnifiedMind(dim=256, seed=0, coherence_floor='auto')
    m.learn("dog", "canine"); m.learn("cat", "feline")
    p = os.path.join(tempfile.mkdtemp(), "mind"); m.save(p)
    assert UnifiedMind.load(p).coherence_floor == 'auto'
