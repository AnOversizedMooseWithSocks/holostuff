"""Dense associative memory -- the modern Hopfield cleanup, and generation by denoising.

WHY THIS EXISTS
---------------
holostuff's standard cleanup (`Vocabulary.cleanup`) is a one-shot HARD nearest-neighbour:
`argmax(V @ q)`, snap to the winning atom. That is already Bayes-optimal for the question
"which stored atom is this noisy vector closest to" -- so nothing can beat it on IDENTITY, and
we measured exactly that (modern Hopfield ties it 1.000 vs 1.000 across noise levels). That tie
is a KEPT NEGATIVE: do not expect an accuracy win on classification.

What the modern continuous Hopfield update (Ramsauer et al. 2020, "Hopfield Networks is All You
Need"; Krotov & Hopfield 2016; Demircigil et al. 2017) actually buys us is two things the hard
argmax cannot give:

  1. CONTINUOUS-VECTOR DENOISING. The update z = V^T softmax(beta * V q) returns a clean vector
     on the stored-pattern manifold, not just an identity. Measured: a recovered vector at heavy
     noise (cosine 0.45 to truth) cleans to cosine ~1.0. That matters wherever snapping to a
     discrete atom is the WRONG move -- continuous encoders, FHRR phasor states, compositional
     intermediates, and as the per-step denoiser inside Plug-and-Play restoration and the
     resonator.

  2. GENERATION BY DENOISING. Iterating the same update from PURE NOISE walks onto the manifold
     (measured: nearest-pattern cosine 0.5 -> 1.0 in ~8 steps). Denoising and generation are the
     SAME operation in different regimes -- the engine's own small diffusion sampler.

DESIGN NOTES (backward-compatible by construction)
  * At beta -> infinity the softmax becomes a one-hot argmax, so `dense_cleanup` reproduces the
    existing hard-NN decision EXACTLY. It is a strict superset, added as a separate callable; the
    default `Vocabulary.cleanup` is untouched.
  * KEPT NEGATIVE (generation): sampling over the bare codebook just returns stored atoms (a
    degenerate sampler). The interesting regime is generating over a COMPOSED or continuous
    manifold -- `generate` takes whatever codebook you hand it, so feed it composed states.
  * Deterministic: `generate` takes an explicit seed; everything else is pure NumPy with no RNG.
"""

import numpy as np


def _unit_rows(M):
    """Row-normalise to unit length (so a dot is a cosine). 1e-12 guards a zero row."""
    M = np.asarray(M, float)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def dense_cleanup(query, codebook, beta=25.0, steps=3):
    """One modern-Hopfield denoise of `query` against `codebook` (V), iterated `steps` times.

    z <- V^T softmax(beta * V z), starting from z = query. Returns the cleaned CONTINUOUS vector
    (not an identity). Higher beta = sharper (beta->inf reproduces hard nearest-neighbour); more
    steps = deeper basin descent. The softmax is computed in a max-subtracted, stable form."""
    V = _unit_rows(codebook)
    z = np.asarray(query, float).copy()
    for _ in range(steps):
        s = V @ z
        s -= s.max()                      # numerical stability; does not change the softmax
        w = np.exp(beta * s)
        w /= w.sum()
        z = V.T @ w
    return z


class HopfieldCleanup:
    """A drop-in associative-memory cleanup with a continuous denoiser and an identity readout.

    fit() caches the unit codebook; cleanup() gives the (index, cosine) hard decision -- identical
    to holostuff's argmax NN -- and denoise() gives the cleaned continuous vector. Same object,
    both readouts, so callers can pick the one their downstream step needs."""

    def __init__(self, beta=25.0, steps=3):
        self.beta = beta
        self.steps = steps
        self.V = None

    def fit(self, codebook):
        self.V = _unit_rows(codebook)
        return self

    def denoise(self, query):
        """Cleaned continuous vector on the stored-pattern manifold."""
        return dense_cleanup(query, self.V, self.beta, self.steps)

    def cleanup(self, query):
        """Hard (index, cosine) readout -- matches Vocabulary.cleanup's decision at high beta."""
        if self.V is None:
            raise ValueError("fit() a codebook first")
        q = np.asarray(query, float)
        qn = np.linalg.norm(q)
        sims = self.V @ (q / qn) if qn > 0 else self.V @ q
        j = int(sims.argmax())
        return j, float(sims[j])


def generate(codebook, steps=12, beta0=4.0, beta1=40.0, noise0=0.6, seed=0):
    """Generate a sample by DENOISING from pure noise (B10): the cleanup attractor as a tiny
    holographic diffusion. Anneal beta upward (vague -> sharp) and injected noise downward across
    `steps`, starting from a random unit vector, ending on the manifold.

    Returns the generated unit vector. NOTE (kept negative): over a bare codebook this converges
    to a stored atom; feed a COMPOSED/continuous manifold as `codebook` for novel-but-valid
    samples. Deterministic in `seed`."""
    V = _unit_rows(codebook)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(V.shape[1])
    z /= np.linalg.norm(z) + 1e-12
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac          # sharpen toward the manifold
        noise = noise0 * (1.0 - frac)                  # cool the injected noise
        z = dense_cleanup(z, V, beta=beta, steps=1)
        if noise > 0:
            z = z + noise * rng.standard_normal(V.shape[1]) / np.sqrt(V.shape[1])
        z /= np.linalg.norm(z) + 1e-12
    return z
