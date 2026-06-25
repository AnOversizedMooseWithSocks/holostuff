# holostuff — re-opening the single-model negatives under the distributed premise

*A working list. The session established a structural fact: many of the engine's hard walls were
not properties of a problem but properties of trying to fit the problem in **one vector / one
model**, and that constraint is now lifted — the store (and, as of the latest result, the
**computation**) is distributed across a federation of vectors coordinated by a thin layer. So
every kept negative recorded under "this doesn't fit in one vector" is worth re-opening. This
document (1) preserves the proven win so it is not lost, (2) lists candidates to investigate, in
honest buckets, and (3) names — equally honestly — the negatives distribution will NOT fix, so we
don't chase them. As always: each result carries its kept negative, procedure-matched nulls, and a
measurement bar; nothing claimed without honest measurement.*

---

## 0. The pattern that makes a candidate

A candidate qualifies if its limit is **crosstalk capacity in a single vector** — the
~0.1×D-items-per-vector budget, or its tighter ~0.02×D continuous-compute cousin. Those walls move
when you **federate** (more vectors = more total dimensions = more capacity) with a **thin
coordination layer** (a directory, a router, a parity shard). The conservation law is the
invariant at every level: *faithful capacity scales with total dimensions*; adding units adds
capacity, partitioning a fixed D conserves it but buys operational wins (fault isolation, parallel
local ops, RAID, routing).

A limit does **not** qualify if it is about *expressiveness, depth/nonlinearity, search difficulty,
or the data simply lacking structure*. More vectors do not add feature interactions, do not deepen
a shallow map, and do not create linear structure in a signal that has none. Those need a different
fix (depth, learning, a better search), and pretending distribution solves them would be the kind
of overclaim the project exists to avoid.

---

## 1. THE PROVEN WIN — distributed superposed forward pass (do not lose this)

**Status: measured, holds, files in `/mnt/user-data/outputs/`.**
`experiment_distributed_forward_pass.py` · `distributed_forward_pass.png`

The flagship Path-D target was "run a neural-network forward pass entirely in superposition and find
the wall." The original result (`experiment_superposed_forward_pass.py`) found a hard one: a forward
pass read out of **one** superposed weight-memory stays faithful only to **C ≈ 0.02×D** classes —
about **6× tighter** than plain storage recall — because continuous compute has no cleanup step to
absorb crosstalk, and was **~2.2× slower** than a plain matmul on CPU (FLOPs don't vanish). That was
a kept negative.

Re-opened under the distributed premise, the wall moves almost linearly with shard count. Measured
at **D = 1024**, federating the weight-memory across K shards (class *c* lives in shard *c* mod K):

| weight-memory | faithful layer width (logit-fidelity ≥ 0.90) | vs single |
|---|---|---|
| 1 shard (the blocked version) | C = 16  (~0.016×D) | — |
| 2 shards | C = 32 | 2× |
| 4 shards | C = 48 | 3× |
| 8 shards | C = 96 | ~6× |

The single-vector classifier collapses past ~32 classes; the **8-shard classifier tracks the exact
matmul out past 160 classes** before gently declining.

**What this establishes (the load-bearing statement to keep):** the 0.02×D ceiling was never a
property of the forward pass — it was the budget of *one vector*. **The conservation law governs
compute capacity exactly as it governs storage: faithful forward-pass width scales with total
dimensions across the federation (≈ K×D).** "As above, so below" now spans the whole stack —
storage, lookup, resilience, factorization, and the forward pass itself — all the same
federate-with-a-thin-layer move.

**Honest accounting, kept on the record:**
- The negative is **relocated, not refuted**: each shard still caps at ~0.02×D, which is exactly
  why K of them buy ~K×. The fix is architectural (more vectors), not a cleverer encoding.
- **No extra compute here**: it is still C row-recoveries total, regrouped into K lower-crosstalk
  bundles; the cost paid is **K×D storage** for the weight-memory, not extra FLOPs. (Storage-side
  RAID *did* add parity overhead; this doesn't.)
- The K shards are **independent → parallelizable**, which is the literal "GPU things without a GPU"
  point: the parallelism is across the federation, native to the in-memory/neuromorphic substrate
  the Path-D charter pointed at.

---

## 1.5 THE BUCKET-A SWEEP — measured outcomes

**Status: all six run on the real kernel. Figure `bucket_A_sweep.png`; scripts
`exp_batch_234.py` (A2/A3/A4), `exp_batch_B.py` (A2/A5/A6), `exp_A1.py` (A1).** D = 1024 throughout.

| item | wall on one vector | under federation | verdict |
|---|---|---|---|
| **A3** hypothesis selection | pick-the-best to H = 64 | H = 256 at K = 8 | **WIN** (~K×) |
| **A4** sequence memory | 90%-recall to T = 64 | T = 256 at K = 8 | **WIN** (~K×) |
| **A6** residue integer range | 10³⁴ (24 moduli) | 10³⁹¹ (160 moduli) at K = 8 | **WIN** (~K× moduli) |
| **A5** federated archive | exact to dim/keep; quality falls as N grows | *identical* quality, capacity ×K | **CONSERVED** + operational win |
| **A2** matmul | lossy-bundle encoding: noise at any size | **RNS-phasor reformulation: EXACT (error 0)**, range federates | **RESOLVED** — the formula needed adapting, not abandoning |
| **A1** deep forward pass | lossy readout: 0.98 → 0.42 → 0.15 by depth | **exact RNS per layer: 1.00 at every depth** | **RESOLVED for inference** — same disease/cure as A2; residual is quantization (clean ≥4 bits) |

**The reading.** The pattern from the win generalizes cleanly **wherever the readout ends in a
discrete cleanup** — argmax (A3 selection), codebook snap (A4 sequence), or CRT (A6 integer range)
all federate the single-vector wall roughly linearly in shard count, exactly like storage and the
forward pass. The archive (A5) behaves like block-federation: at fixed total dimensions, partitioning
**conserves** recovery quality exactly (the monolithic and federated curves lie on top of each other),
and capacity scales by adding dimensions (more shards), with the operational wins (parallel recovery,
fault isolation, the directory) coming for free.

The two that are **not** wins are the most useful results, because they mark the honest edge of the
premise. **A2 (dense continuous matmul)** stays low-fidelity at any size, single or federated — a
continuous product has no cleanup to absorb crosstalk, so this is a *precision* wall, not a capacity
wall, and federation only nudges it. **A1 (depth)** is the genuine frontier: federation handles
per-layer *width*, but noise *compounds across layers* (a single hidden layer recovers to near-exact
with cleanup; by three layers most of the signal is gone), and inter-layer cleanup helps every time
without fully stopping the bleed. Both confirm the §0 principle from the other side: **distribution is
the width/capacity lever. It is not a depth lever and not a precision lever** — A1 needs better
inter-layer cleanup (or learned attractors), A2 needs a discrete readout to anchor it (at which point
it *is* A3). These belong beside the Bucket-C honesty, now measured rather than predicted.

---

## 1.6 A2 RESOLVED — adapt the formula to the substrate (RNS-phasor matmul)

**Status: built and measured. Figure `a2_resolved_rns.png`; script `exp_A2_rns.py`.**

A2 first read as a boundary. It is not. The wall belonged to *one encoding* — store the matrix as
superposed rows, read them back, dot with the input, no cleanup — which bundles many things into one
vector so crosstalk wins. The matmul *formula* (designed for silicon: store densely, multiply) was
being forced onto the substrate unchanged. Decomposed into the substrate's exact primitives and
recomposed, it works:

- **decompose** — cast W, x to integers; take residues mod each coprime modulus *m_k* (one per channel/shard).
- **adapt** — multiply-accumulate mod *m_k* in each channel, where the **accumulation is FHRR phasor
  binding**: `∏ exp(2πi r/m) = exp(2πi (Σr)/m)`, so the sum-mod-*m* is carried *exactly* in the phase,
  for any number of terms — no superposition, no crosstalk. (Measured: 0 errors summing 5,000 integers.)
- **recompose** — CRT-combine the per-channel residues into the exact integer result.

**Measured:** integer matmul at M = 8, 64, 256 — the exact sizes the bundle degraded on — returns
**max error 0**, while the lossy bundle sat at ~0 fidelity (noise). The dynamic range **federates over
moduli channels exactly like A6**: 4 / 8 / 16 / 32 channels → 10⁸ / 10¹⁶ / 10³⁴ / 10⁷¹ exact range.

**The principle this establishes (worth more than the one result):** a wall found with the *naive*
encoding is a question about the encoding, not a law of the substrate. The right move is
*decompose → adapt to the native primitives → recompose* — and the native primitive that matters here
is **exact phasor binding for arithmetic**, never lossy superposition for it. This is the published
**Residue Hyperdimensional Computing** line (Kymn, Mazelet, Ng, Kleyko, Olshausen, Sommer, 2025) over
FHRR phasors (Plate), and it's already in the project's orbit (A6; leOS's `residue_arithmetic.py`).

**Honest scope, kept on the record:** exact for **integer / fixed-point within range** (a float
matmul is quantized first — a precision *choice*, the same trade A6 makes, and a good one because you
size range and precision by choosing moduli). FLOPs are real — the parallelism is per-modulus and
per-output, native on phasor / RNS / neuromorphic hardware, not free on a CPU. And this does **not**
by itself close A1 (depth): depth's problem is compounding *nonlinearity*, not arithmetic crosstalk —
though making each layer's matmul exact this way removes the per-layer arithmetic noise and isolates
the genuine depth/cleanup question, which is the natural follow-up.

---

## 1.7 A1 FLIPPED — depth was the A2 disease in disguise

**Status: built and measured. Figure `a1_resolved_rns.png`; script `exp_A1_rns.py`.**

A1 first read as the genuine frontier — depth, where federation handled width but accuracy compounded
*downward* (0.98 → 0.42 → 0.15) and cleanup only half-helped. Applying A2's lesson exposed the real
cause: the per-layer noise was the **lossy weight-superposition readout**, the same encoding artifact,
compounding layer over layer. Rebuild each layer's matmul as **exact RNS-phasor** (fixed-point, the A2
cure) and the decay is gone — the substrate forward pass tracks the float MLP at **1.000 at depth 1, 2,
and 3.** There was nothing to compound once each layer was exact.

The one honest residual is **fixed-point quantization** — a real, *separable, controllable* factor, not
a crosstalk wall. Measured at depth 3: 0.986 at 3-bit, **1.000 from 4-bit up.** Bit-depth (more moduli)
is the knob, exactly as quantized inference always has.

**Boundary, stated plainly so this isn't overclaimed:** this is exact deep **inference** of an
*already-trained* net — depth is not a wall for *running* a known network in the substrate. It does
**not** address **learning** a deep net there (gradients/training — Gap 1 in the frontier audit), which
remains the genuine open frontier. What flipped is the inference-depth wall; the learning-depth question
is untouched and separate.

**The pattern now firm across A1 and A2:** both walls were one disease — forcing the silicon formula
(lossy superposition for arithmetic, continuous readout, no cleanup) onto the substrate — and both take
one cure: **exact phasor-binding arithmetic (RNS) is the substrate's native way to compute, never lossy
superposition.** The engine now has two distinct, composable levers: **distribution** moves
capacity/width walls; **RNS-phasor** removes arithmetic-crosstalk walls in compute (matmul, and any
depth of it). What is left genuinely hard is no longer "compute in the substrate" — it is **learning**
(gradients) and the data-structure-limited cases of Bucket C. A much smaller, sharper frontier than the
sweep first suggested.

---

## 2. CANDIDATES TO INVESTIGATE

### Bucket A — strong fit (a clean single-vector capacity wall; distribution should push it)

**A1 — Deep / multi-layer forward pass with cleanup between layers.**
*Seats: Plate (capacity), Olshausen (Hopfield cleanup as the nonlinearity), Cranmer (measurement).*
A1 is the direct sequel to the win: stack layers. Per-layer width is now federatable (§1), but a
deep net feeds each layer's noisy output into the next, so crosstalk **compounds with depth** —
the same compounding we saw in tree routing and iterated decode. The likely fix is a **cleanup
(dense-Hopfield) step between layers** to snap the activation back onto the manifold and stop the
compounding — the discrete-cleanup move the single continuous pass lacked. *Bar:* a 2–4 layer
federated classifier whose accuracy holds with depth, vs the same net with no inter-layer cleanup
(the kept negative: how far depth goes before noise wins without cleanup). *Risk:* this is partly a
**depth** problem (Gap 2 from the frontier audit), not purely capacity — distribution handles the
width, cleanup must handle the depth.

**A2 — General matmul / dense linear algebra in superposition.**
*Seat: Stoudenmire (tensor networks — the bridge from binding to matmul-class compute).*
The forward pass is one matrix-vector product; the charter flagged **dense linear algebra at scale**
as the hardest "GPU thing." Federate the matrix across shards (rows or blocks) and measure whether
faithful matmul size scales with K, for W@x (many inputs) and W₁@W₂. *Bar:* fidelity of a superposed
W@X vs the exact product as matrix size grows, single-vector vs federated; report the FLOPs honestly
(this likely *does* cost real compute, unlike A0). *Risk:* exact high-precision dense LA is where the
substrate is weakest (Wall 3, precision-for-parallelism) — the honest expectation is "scales the
approximate/low-rank case, loses to a GPU on exact dense at scale." Tensor-train structure may help.

**A3 — Distributed hypothesis evaluation (`superposed_compute`, generalized).**
*Seats: the Path-D compute frame; leOS `superposed_compute.py` ("one processor, many states").*
Evaluating many hypotheses bundled into one vector hits the same 0.02×D continuous-readout wall as
the forward pass — A3 is literally the win's sibling. Federate the hypotheses across shards →
more candidates evaluated faithfully in parallel. *Bar:* number of hypotheses scored within a
fidelity tolerance, single-vector vs K-shard. *Risk:* low — this is the same mechanism as §1, just
a different application; expect the same ~linear scaling.

**A4 — Sequence / temporal memory length.**
*Seat: Puckette (sequences, temporal structure); the recurrent/reservoir modules.*
A sequence stored in one vector via permutation-binding has a length budget before crosstalk; long
sequences were length-capped. Federate the sequence across vectors (a windowed/chained store with a
thin "which window" index) to hold longer sequences at fidelity. *Bar:* recoverable sequence length
vs shard count, vs the single-vector length cliff. *Risk:* moderate — sequence recall couples order
(permutation) with capacity; verify the federation doesn't break order recovery.

**A5 — Sharded content archive (more items at fidelity).**
*Seats: Ozcan (reconstruction under degradation), Pharr (content-addressable cache).*
`HolographicArchive` (WHT plates) and `splat_archive` have a per-archive item budget. This is the
storage win applied to the archive specifically: a **federated archive** (shards + the §1-style
directory/router, plus per-shard parity) to hold many more images/items at recall fidelity, with
graceful degradation. *Bar:* images stored at fixed reconstruction PSNR/SSIM, single archive vs
federated, with a recall route. *Risk:* low — this is the array pattern on a concrete payload; mostly
integration. (Note: the general "directory-routed storage is effectively unbounded" result already
holds; A5 is the specific, useful instantiation.)

**A6 — Residue integer-arithmetic range (leOS-side).**
*Seat: residue-HDC (Kymn et al.); leOS `residue_arithmetic.py`.*
Integer math via CRT-residue binding is range-bounded by the moduli product and per-vector capacity.
Federate the residue channels / moduli across vectors to extend the faithful integer range. *Bar:*
largest integer range with exact round-trip, single-vector vs federated. *Risk:* moderate — residue
arithmetic is exact-by-construction within range; the question is whether federation extends range
without breaking the CRT reconstruction. Lives in leOS; port a minimal version to holostuff to test.

### Bucket B — already investigated this session (don't redo; extend)

**B1 — Factorization capacity.** *Done:* `experiment_factor_wall.py` / `factor_wall.png`. Dense
resonator cliffs at F=4 (0.67) / F=5 (0.29); block-distributed SBC pushes ~1 factor (0.92 / 0.58);
both die at F=6; **adding dimension shifts the wall right** (F=4 solved at D≥2048; F=6 cracks at
D=4096). *Open extension:* a **hierarchical / federated resonator** (a tree of small resonators over
shards) to push the joint-search wall further — the recursive version of the same move.

**B2 — Compositional / nesting depth.** *Largely addressed:* the pivot tree
(`experiment_pivot_tree.py`) routes as accurately as exhaustive at every depth (86× fewer
comparisons at depth 4; beam-5 truth-in-set 99.9%), and `holographic_peel` decodes to depth 15.
*Open extension:* a formal **federated deep-structure encoder** — a deep recipe split across vectors
with pointer links — measured for depth scaling against the single-vector inception law (~8).

**B3 — Sublinear / routed cleanup at scale.** *Largely done:* the sketch router + recursive
pivot tree give sublinear, depth-surviving routing, and `recall_calibrated` runs through the forest.
*Open:* minor — confirm the routed cleanup's abstention (calibrated p) and the forest's cross-tree
agreement agree at scale.

**B4 — Storage capacity (the canonical win).** *Resolved:* the aligned array
(`holographic_array.py`, `experiment_array_*.py`) scales 90%-recall linearly with shards
(48/96/192/384 for K=1/2/4/8), directory-routed storage is effectively unbounded (46,080 items at
0.97 recall, O(1) query), with RAID parity and self-upgrade. This is the parent result the rest
descend from.

### Bucket C — NOT distribution candidates (honest; listed so we don't waste effort)

These are real kept negatives, but their cause is **not** single-vector capacity, so federating will
not help. They need depth, learning, a better search, or are simply facts about the data.

| Negative on the record | Why it's NOT a distribution candidate |
|---|---|
| `learn_dynamics` ties a mean predictor on market returns | The data has ~no linear structure to exploit — a regime fact, not a capacity wall. Distribution adds nothing. |
| `spectral_encode` fails on pure tones / non-integer-cycle freqs; Burgers shock-forming flow | Representational / nonlinearity-regime limits, not crosstalk. |
| Multiplicative symbolic regression expressiveness | A search/expressiveness limit (the log-transform was the fix); more vectors don't widen the hypothesis class. |
| `fit_function` (KAN) is single-layer, no feature interactions | A **depth/nonlinearity** gap — needs interaction terms or depth, not width. (Weakly related to A1: cleanup-between-layers, not federation, is the lever.) |
| No gradient learning (SVD/ridge/DMD/RL only) | A **learning** gap (Gap 1) — the differentiable-backend question, entirely separate from distribution. |
| Binary quantization distorts pairwise-similarity geometry | A **precision** limit (rate-distortion), addressed by `quant='rd'`, not by more vectors. |

---

## 3. Suggested sequencing (value over effort)

**Done (the sweep above):** A2, A3, A4, A5, A6, plus the proven forward-pass win in §1. What remains
is not new Bucket-A items but the two **follow-ups the sweep surfaced**, where the honest edge is:

1. **A1 depth — the real open problem.** Federation solved width; depth did not fall. The lever to try
   next is a *better* inter-layer cleanup: hard nearest-prototype or class-conditional attractors
   instead of the soft codebook used here, an iterated (multi-step Hopfield) settle between layers, or
   a *learned* cleanup — and measure whether usable depth extends past 2. Keep the no-cleanup decay
   curve as the kept negative. This is where "the substrate does deep nets" is genuinely decided.
2. **A2 with a readout — close the loop.** Dense continuous matmul is a boundary, but the moment its
   output is consumed by a discrete cleanup it becomes A3. Worth a short demo: a two-stage
   matmul→select pipeline, to mark exactly where in a computation the cleanup has to sit for the
   substrate to stay faithful.

Lower priority, when the above are banked: **B1-extension** (hierarchical resonator) and **A6 in leOS**
(port the residue-range result into the parent system's `residue_arithmetic.py`).

---

## 4. The discipline (so the list stays honest)

Every item lands under the engine's standing rules: **procedure-matched nulls** (not naive ones);
**kept negatives travel with the code** (docstring carries validated regimes *and* failure modes —
e.g. "faithful to C≈0.02×D **per shard**"); **dual-criterion claims** where a selector is involved;
**no precomputed shortcuts** in any demo; and the **close-out ritual** on each result that clears its
bar — append to `tour.py` (§15), `NOTES_concepts.md`, and `README.md` (test count + integration
list), then rebuild and verify the zip from a clean extraction. The conservation law is the through
-line to state once and reuse: *capacity — storage or compute — tracks total dimensions; you escape a
single vector's budget by federating, not by encoding harder.*

---

### Files referenced
- **Win:** `experiment_distributed_forward_pass.py`, `distributed_forward_pass.png`; original negative `experiment_superposed_forward_pass.py`, `capacity_cliff.png`.
- **This session's banked-pending results:** `holographic_array.py`, `holographic_superposed.py`, `experiment_array_resilience.py`, `experiment_array_scale.py`, `experiment_array_router.py`, `experiment_pivot_tree.py` (`pivot_tree.png`), `experiment_below_federation.py`, `experiment_factor_wall.py` (`factor_wall.png`).
- **Charter context:** `holostuff_path_d_holographic_compute.md`, `holostuff_frontier_catchup_plan.md`.
