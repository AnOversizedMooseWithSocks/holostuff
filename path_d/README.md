# holostuff — Path D work bundle (the "as above, so below" arc)

This bundle is the complete record of the Path D investigation: **computing and storing inside the
holographic (VSA/HRR) space**, and the discovery that one conservation law governs the whole stack.
Everything here was prototyped and **measured on the real holostuff kernel** before being written
down, with negatives kept on the record — the engine's own method.

> **The through-line.** A single D-dimensional vector holds only ~0.1×D items faithfully (and only
> ~0.02×D for *continuous* compute). That budget is **conserved**. You don't beat it by encoding
> harder — you **federate**: more vectors = more total dimensions = more capacity, coordinated by a
> thin layer (a directory, a router, a parity shard, a CRT recompose). The same move recurs at every
> scale — storage, lookup, resilience, factorization, and the neural-network forward pass itself.

> **Two levers emerged, and they compose.**
> 1. **Distribution** moves *capacity / width* walls (the storage cliff, the forward-pass width cliff).
> 2. **RNS-phasor arithmetic** removes *arithmetic-crosstalk* walls in compute — by carrying numbers as
>    residues over exact FHRR phasor binding instead of lossy superposition. This flipped two walls
>    (matmul, and depth) that first looked fundamental.
>
> What is left genuinely hard is no longer "can the substrate compute." It is **learning** (gradient
> training — not addressed here) and the handful of cases where the data simply lacks structure.

---

## Headline results (all measured, D = 1024 unless noted)

| area | result | file(s) |
|---|---|---|
| **Storage array** | 90%-recall scales **linearly** with shards (48/96/192/384 for K=1/2/4/8); directory-routed storage effectively **unbounded** (46,080 items @ 0.97 recall, O(1) query); RAID-5/6 parity reconstructs exactly; self-upgrades under pressure | `experiment_vsa_array.py`, `experiment_array_*.py`, `holographic_array.py` |
| **Recursive pivot-tree index** | greedy top-1 routing = exhaustive at **every** depth (0.882 = 0.882); **86× fewer** comparisons at depth 4 (28 vs 2401); beam-5 lands the true shard 99.9%; naive summary-of-summaries index = 0.23 (kept negative) | `experiment_pivot_tree.py` |
| **Downward block-federation** | partitioning a fixed D **conserves** capacity (48/48/48/48/64 for B=1/2/4/8/16, fixed codebook); RAID is scale-invariant (block d=128 ≈ array d=1024) | `experiment_below_federation.py` |
| **THE WIN — distributed forward pass** | single weight-vector faithful to **16 classes** (~0.02×D); federating → 32 / 48 / **96** for K=2/4/8 (~6×); 8-shard classifier tracks the exact matmul **past 160 classes** | `experiment_distributed_forward_pass.py` |
| **Factorization wall** | dense resonator cliffs F=4 (0.67) / F=5 (0.29) / F=6 (0.04); SBC pushes ~1 factor (F=4 0.92 / F=5 0.58); both die at F=6; **adding dimension** shifts the wall right (F=4 solved at D≥2048; F=6 cracks 0.19 at D=4096) | `experiment_factor_wall.py` |

### Bucket A sweep (six single-vector walls re-opened under distribution)

| item | one vector | federated | verdict | file |
|---|---|---|---|---|
| **A3** hypothesis selection | pick-best to H=64 | H=256 (K=8) | **WIN** ~K× | `exp_batch_234.py` |
| **A4** sequence memory | 90%-recall to T=64 | T=256 (K=8) | **WIN** ~K× | `exp_batch_234.py` |
| **A6** residue integer range | 10³⁴ (24 moduli) | 10³⁹¹ (160 moduli, K=8) | **WIN** ~K× moduli | `exp_batch_B.py` |
| **A5** federated archive | exact to dim/keep; quality falls as N grows | *identical* quality at fixed total dim; capacity ×K | **CONSERVED** + operational win | `exp_batch_B.py` |
| **A2** matmul | lossy-bundle encoding → noise at any size | **RNS-phasor → EXACT (error 0)**; range federates 10⁸→10⁷¹ (4→32 ch) | **RESOLVED** by adapting the formula | `exp_A2_rns.py` |
| **A1** deep forward pass | lossy readout 0.98 → 0.42 → 0.15 by depth | **exact RNS per layer → 1.00 at every depth**; residual is quantization (clean ≥4 bits) | **RESOLVED for inference** | `exp_A1_rns.py` |

The original **A1/A2 first reads** (a depth gap and a precision boundary) are also kept, as the
encoding-artifact negatives that the RNS reformulation later flipped: `exp_A1.py` (lossy deep
forward pass + inter-layer cleanup), `experiment_superposed_forward_pass.py` (the original ~0.02×D
forward-pass cliff), `experiment_depth_vs_width.py`.

---

## The kept negatives (these travel with the work)

- A **single** vector's continuous-compute budget is ~0.02×D — ~6× tighter than discrete storage
  recall, and the superposed forward pass is ~2.2× slower than a matmul on CPU. The win **relocates**
  this (each shard still caps at ~0.02×D; K shards buy ~K×), it does not refute it.
- Naive **summary-of-summaries** routing crashes (the sketch re-blows the budget one level up); tree
  routing needs *structured* data (random keys → use a hash).
- Factorization is a **joint** combinatorial search → distribution *pushes* the wall but does not
  demolish it (a steeper exchange rate than independent storage).
- **A2/A1 are exact only for integer / fixed-point within range** — a float matmul is quantized first
  (a precision *choice*; size range/precision by choosing moduli). FLOPs are real; the parallelism is
  per-modulus / per-output, native on phasor/RNS/neuromorphic hardware, not free on a CPU.
- The A1 flip is exact **inference** of an *already-trained* net. It does **not** address **learning**
  a deep net in the substrate (gradients) — that remains the genuine open frontier.
- Dense continuous matmul with the *lossy* encoding stays noise at any size (the boundary the RNS
  reformulation exists to escape) — kept in `exp_batch_B.py` / `exp_A2_rns.py` as the head-to-head.

---

## Directory map

```
docs/        the design + research documents
  holostuff_distribution_candidates.md     <- START HERE: the candidate list with all sweep results,
                                               the A2 (§1.6) and A1 (§1.7) flips, and the verdict table
  holostuff_path_d_holographic_compute.md   the Path D charter (what's real, the walls, where it's new)
  holostuff_frontier_catchup_plan.md        the "behind on what, exactly?" audit (learning/depth/scale)
  holostuff_frontier_program.md             the honest frontier-gap assessment (A/B/C buckets)
  holostuff_dataset_benchmark_program.md    the 19-seat dataset benchmark program

modules/     new / ported engine code
  holographic_array.py        the aligned VSA storage array: linear-scaling recall, directory routing,
                              RAID-5/6 parity shards, self-upgrade on sensed pressure
  holographic_superposed.py   the WIDTH primitive (pack / recover_all / score_all) ported from leOS

experiments/ the measured experiments (each is the record of what was run)
plots/       the figure-generator + helper scripts that produced figures/
figures/     all 12 figures (PNG)
```

## How to run an experiment

Each experiment grounds itself on the **real holostuff kernel** via
`sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repo"))`. To reproduce, place a checkout
of the holostuff repository (github.com/AnOversizedMooseWithSocks/holostuff) at
`experiments/repo/` (and `plots/repo/` for the plot scripts), then e.g.:

```
cd experiments && python3 exp_A1_rns.py        # the depth flip
python3 exp_A2_rns.py                           # the matmul flip
python3 experiment_distributed_forward_pass.py  # the win
```

Dependencies: `numpy`, `scikit-learn`, `matplotlib` (plots). Everything is CPU-only and deterministic.

---

## Status

This is the **artifact bundle** — the complete record of what was built and measured. It is *not* yet
integrated into the holostuff repo: the close-out ritual (wire `holographic_array` as a UnifiedMind
faculty, add the conservation-law + two-levers notes to `NOTES_concepts.md`, bump the README test
count and integration list, add an end-to-end integration test, rebuild and verify
`holographic_vsa_complete.zip` from a clean extraction) is a separate step, available on request.
