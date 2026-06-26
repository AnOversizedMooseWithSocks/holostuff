"""FHRR phase-domain morph -- interpolate in the phase domain (phase shift = motion), not by blending amplitudes.

WHY THIS EXISTS (PHASE-1)
-------------------------
Phase-based frame interpolation interpolates the PHASE (which encodes where a feature is) rather than blending pixel
amplitudes, because amplitude blending under large motion produces a ghost (two faint copies fading through each
other) while a phase shift MOVES the feature. FHRR is already the engine's phase domain: every atom is a vector of
complex UNIT PHASORS e^{i*theta}. So the engine gets phase-domain interpolation for free -- morph by shifting each
component's phase along the shortest arc, staying on the unit-phasor manifold, instead of blending the complex
vectors (the current amplitude-domain morph).

MEASURED (see `_selftest`, an FHRR-encoded position = a feature moving a large distance between two states):
  * UNIFORM MOTION (the "phase shift = motion" win): the phase morph moves the decoded feature at CONSTANT velocity
    -- it tracks the ideal uniform trajectory exactly (deviation ~0). The amplitude blend instead STALLS near each
    endpoint and rushes through the middle (an eased, non-physical S-curve, deviation ~0.06), because the phase of a
    weighted complex sum is biased toward the heavier endpoint.
  * ENERGY / VALIDITY: the phase morph is a valid unit phasor at EVERY t (magnitude 1). The amplitude blend
    COLLAPSES where components fall out of phase -- at the midpoint of a large change its mean magnitude drops to
    ~0.75 (and toward 0.64 for independent states), so it is not even a valid FHRR vector without renormalising.

THE HONEST NEGATIVE (the scope, kept loud): phase-domain morphing is NOT a free win under arbitrarily large change.
The morph uses the SHORTEST ARC per component, which WRAPS once a component's phase difference exceeds pi -- past
that it takes the wrong way round and stops tracking the true intermediate (measured: at a separation where
per-component phase diffs reach ~1.6*pi the decoded trajectory deviates by ~0.98, i.e. completely lost). And
near-orthogonal endpoints have no well-defined intermediate for ANY method. So the win holds while the change keeps
per-component phase differences under pi; beyond that it degrades, gracefully on energy (still unit phasors) but not
on tracking.
"""

import numpy as np


def phase_morph(a, b, t):
    """Morph between two FHRR phasor vectors by interpolating each component's PHASE along the SHORTEST ARC --
    phase shift = motion. Stays on the unit-phasor manifold (every output component has magnitude 1), giving uniform
    feature motion and full energy. `t` in [0, 1]. NOTE the scope: the shortest arc wraps once a component's phase
    difference exceeds pi, so under extreme change (near-orthogonal states) it stops tracking the true intermediate
    -- a kept negative, not a bug."""
    a = np.asarray(a)
    b = np.asarray(b)
    da = np.angle(b) - np.angle(a)
    da = (da + np.pi) % (2 * np.pi) - np.pi                  # shortest arc: the shorter way round the circle
    return np.exp(1j * (np.angle(a) + t * da))


def amplitude_morph(a, b, t):
    """The amplitude-domain baseline: a linear blend of the complex vectors. Collapses in magnitude where the
    components fall out of phase (an invalid, sub-unit phasor), and moves the feature non-uniformly."""
    return (1 - t) * np.asarray(a) + t * np.asarray(b)


def _selftest():
    """CI-fast: an FHRR-encoded position is morphed across a large change. The phase morph moves the decoded
    feature at constant velocity (tracks the ideal trajectory; the amplitude blend eases/stalls) and stays a valid
    unit phasor (the blend's magnitude collapses). The kept negative: under extreme change the shortest-arc morph
    wraps and loses tracking."""
    from holographic_fhrr import phasor_atom, fhrr_sim
    rng = np.random.default_rng(0)
    D = 2048
    base = phasor_atom(D, rng)
    phi = np.angle(base)
    def encode(x):
        return np.exp(1j * x * phi)                          # FHRR fractional-power position encoding
    def decode_x(q, grid):
        s = np.array([fhrr_sim(q, encode(x)) for x in grid])
        return float(grid[np.argmax(s)])

    # WIN regime: a feature moving a large distance, per-component phase diffs still under pi
    xA, xB = 0.1, 0.9
    a, b = encode(xA), encode(xB)
    grid = np.linspace(-0.1, 1.1, 481)
    ts = np.linspace(0.1, 0.9, 9)
    dev_p = max(abs(decode_x(phase_morph(a, b, t), grid) - ((1 - t) * xA + t * xB)) for t in ts)
    dev_a = max(abs(decode_x(amplitude_morph(a, b, t), grid) - ((1 - t) * xA + t * xB)) for t in ts)
    assert dev_p < 0.02 and dev_p < dev_a * 0.5, (dev_p, dev_a)   # phase = uniform motion, amplitude = eased

    # energy / validity: the phase morph is a unit phasor everywhere; the amplitude blend collapses
    pm, am = phase_morph(a, b, 0.5), amplitude_morph(a, b, 0.5)
    assert np.allclose(np.abs(pm), 1.0) and np.mean(np.abs(am)) < 0.8, np.mean(np.abs(am))

    # HONEST NEGATIVE: under extreme change the shortest arc wraps and loses tracking (not a free win)
    xA2, xB2 = -0.3, 1.3
    a2, b2 = encode(xA2), encode(xB2)
    grid2 = np.linspace(-0.5, 1.5, 601)
    dev_extreme = max(abs(decode_x(phase_morph(a2, b2, t), grid2) - ((1 - t) * xA2 + t * xB2)) for t in ts)
    assert dev_extreme > 0.2, dev_extreme


if __name__ == "__main__":
    _selftest()
    print("holographic_phasemorph selftest passed")
