# holostuff Frontier Program — the panel reconvenes against the real 2026 state of the art

*Moose pushed back on the first benchmark pass: the datasets were too small, and the tests read like
a hobby project rather than people trying to crack problems they are actually stuck on. So the panel
was reconvened with three instructions: (1) point each seat at the problem its field is genuinely
fighting in **June 2026**, not a tidy textbook task; (2) use a dataset big enough to mean something
(**≥10 MB**, still ≤100 MB for the GitHub constraint); and (3) be honest about whether the seat even
**wants** the engine pointed there — because at the real frontier, holostuff loses the accuracy crown
in almost every one of these fields, and pretending otherwise would betray the whole project.*

*As before: every proposal is attributed to a SEAT and that field's REAL published method. No opinion
is invented. Where the field's frontier has moved past what holostuff can touch, that is said plainly
and the seat is either re-scoped to the transferable abstraction or kept as a structural
proof-of-concept with a loud negative.*

---

## The honest finding, stated first

The first pass asked "can holostuff beat each field's mature method on a small dataset?" and answered,
mostly, "no, but it has a complementary property." Pointing the same question at the **2026 frontier**
sharpens the answer rather than softening it. Against the actual state of the art —

- **protein:** deep-learning binder design (BindCraft, RFdiffusion3 atom-level-at-scale), where the
  *open* problem is candidate triage, not generation;
- **fluids/PDEs:** FNO is now the *baseline*; foundation operators (DPOT, OmniArch) and shock/long-
  rollout robustness are the frontier;
- **denoising:** real sensor noise on SIDD/DND, owned by Restormer/NAFNet (~40 dB) and self-supervised
  diffusion, with BM3D the historical floor;
- **splatting:** rate-distortion-optimal compression of dynamic/large scenes (CAT-3DGS, HAC, MPEG-GS,
  the glTF splat standard arriving Q2 2026);
- **retrieval:** billion-scale, filtered ANN for RAG (FAISS, DiskANN, CAGRA), where the live pain is
  *trustworthy* retrieval;
- **knowledge graphs:** the leaderboard (ComplEx, RotatE, AutoSF+) and KG foundation models;
- **audio:** neural codecs (EnCodec, DAC) and LLM/diffusion source separation on MUSDB18-HQ;

— holostuff is **not competitive on the headline metric**, and the program says so up front, per seat.
What survives the reconvening is narrower and more honest: the specific, *current* places where a
deterministic, calibrated, content-addressable, erasure-robust VSA engine earns a seat next to the
tools that outscore it. The ambition Moose asked for is real, but it is the ambition of **measuring
honestly against the real frontier**, not of claiming a win that isn't there.

The reconvening sorts the nineteen seats into three honest buckets.

**Bucket A — genuine complementary capability at the frontier (keep, point at frontier data, concede
the accuracy crown):** the calibration / abstention / honesty seats (Tarter, Siemion, Cranmer, Pharr),
the content-addressable-memory seats (Plate, Puckette-recall), the erasure-robust storage seats
(Ozcan, Duda), the invertible-dynamics seat (Stam), the explainable-agent seat (Togelius). These are
the strongest: the SOTA tools in each field largely *lack* the property holostuff supplies, and the
frontier datasets exercise it.

**Bucket B — frontier is genuinely beyond the engine; scope to the transferable abstraction on
controlled data with a loud negative (keep, but honest about reach):** Baker (the protein frontier is
owned by AF-guided diffusion; holostuff has no protein model — test the fragment-assembly *abstraction*
only), Stoudenmire (tensor-network bind is research, not a benchmark), Macklin (a constraint solver in
bind/unbind is a real idea; learned simulators win the physics).

**Bucket C — structurally adjacent but two big steps away (keep the structural proof-of-concept,
concede the real problem):** Drettakis (a splat scene *is* a bundle, and splat compression *is* a
rate-distortion problem holostuff has a code for — but the engine is isotropic-2D while the frontier is
anisotropic-3D dynamic), Quílez (procedural richness from a tiny kernel is real; neural fields own
detailed geometry), Eno (generation-over-a-composed-manifold is real; modern generative models own
richness), Adamatzky (the flow solver is faithful Physarum; the multi-terminal network-design frontier
needs A9, still unbuilt), Olshausen (the resonator is a real cortical-scene model; it hits its
operational-capacity cliff well before real cluttered scenes).

---

## Dataset mechanics, updated for ≥10 MB

The ≥10 MB requirement collides head-on with the fetch constraint, and it is worth being blunt about
why. Our network allowlist reaches only `github.com`, `raw.githubusercontent.com`, and
`codeload.github.com`. Real frontier datasets in the 10–100 MB range almost always live somewhere we
**cannot** reach — Zenodo (JetNet, MUSDB18-HQ, PDEBench), Stanford SNAP (OGB), the SIDD project server,
research.yandex / Microsoft (the billion-scale ANN sets), Kaggle. And the ones that *are* on GitHub are
usually **git-LFS-backed**, which means `raw.githubusercontent.com` hands back a tiny pointer file, not
the bytes — and the real bytes sit on `media.githubusercontent.com`, off the allowlist. Archive zips
ship the same pointers. A bigger dataset makes this trap *more* likely, not less.

So the rule the first pass discovered holds harder here: **the only reliable ≥10 MB source is data
committed directly as ordinary git blobs.** Two mechanisms cover every seat:

1. **Commit a license-compatible curated subset into the holostuff data repo as ordinary blobs.** This
   is exactly how the project already ships its Binance `.npz` — derived, frozen, versioned, no LFS. For
   each frontier dataset we take the real benchmark, carve a ≥10 MB / ≤100 MB slice that preserves the
   hard part (e.g. 100k JetNet jets ≈ 36 MB out of the full ~316 MB; the SIDD validation patches; a
   GloVe-100d 100k-vector slice ≈ 40 MB; the ogbl-biokg triples ≈ 57 MB as packed int32), and commit it
   with a `SOURCE.md` recording provenance, license, the upstream SHA/DOI, and the exact subsetting
   script so it is reproducible and honest. The `sha256` in the fetcher's `SOURCES` table then frozen-
   pins our committed slice, and an LFS-pointer-instead-of-data surprise simply fails the hash.
2. **Generate locally from a committed deterministic generator** where no honest committed slice exists
   — the Stam exception, now expanded. Stam's 2-D Navier–Stokes field comes from a committed FFT-on-a-
   torus pseudospectral solver (his own method); the same applies anywhere the "dataset" is really a
   simulator output.

The fetcher (`holographic_datasets.py`, sketched in the first program — stdlib-only `urllib` +
`hashlib` + `zipfile`, SHA-pinned `SOURCES`, 100 MB size guard, `sha256` verify, idempotent `ensure()`)
is unchanged; only the `SOURCES` entries change to point at our committed slices. The honest cost is
stated plainly: **we are benchmarking on curated subsets of the real frontier datasets, not the full
sets**, so every result carries "on an N-sample slice" in its claim, and the subset script is in the
repo so anyone can check the slice isn't cherry-picked.

| Seat | Frontier dataset (real home) | How we get ≥10 MB | Tag |
|---|---|---|---|
| Tarter | Breakthrough Listen / GBT filterbank (Berkeley, Zenodo) | committed multi-coarse-channel `.fil` slice | subset |
| Siemion | HTRU2 + a BL candidate slice | committed; HTRU2 direct | subset/direct |
| Baker | PDB structures + a design-benchmark target set (RCSB) | committed fragment library + targets | subset |
| Olshausen | cluttered visual scenes (COCO-things crops / MNIST-superpose) | committed composed-scene set | subset |
| Cranmer | **JetNet / JetClass** particle clouds (Zenodo) | committed 100k-jet slice (~36 MB) | subset |
| Pharr | **GloVe-100d** / SIFT (ANN-Benchmarks) | committed 100k-vector slice (~40 MB) | subset |
| Adamatzky | real road / transport network (OSM extract, DIMACS) | committed city graph (~10–50 MB) | subset |
| Quílez | high-res texture album (DTD / Brodatz) | committed texture slice | subset |
| Togelius | larger level corpus / agent trajectories (VGLC, NetHack) | committed level + trajectory slice | subset |
| Macklin | **AMASS / CMU mocap** (BVH) | committed multi-clip slice | subset |
| Stam | **PDEBench as reference**; field generated locally | committed FFT-torus solver (`[gen]`) | gen |
| Ozcan | real microscopy (BBBC / malaria cells) | committed cell-image slice | subset |
| Puckette | **MUSDB18-HQ** / NSynth (Zenodo) | committed stem/instrument slice | subset |
| Duda | the GloVe / image store (shared with Pharr) | committed (shared) | subset |
| Plate | **ogbl-biokg** biomedical KG (SNAP/OGB) | committed triple slice (~57 MB) | subset |
| Stoudenmire | MNIST / Frey (low-rank structure) | committed | direct/subset |
| Milanfar | **SIDD / DND** real-noise pairs (SIDD server) | committed validation-patch slice | subset |
| Drettakis | a real splat scene + 2-D field (Mip-NeRF360) | committed 2-D field; PLY splat slice | subset |
| Eno | larger MIDI corpus (Lakh / Nottingham) | committed chorale+MIDI slice | subset |

---

## Seat by seat: the real 2026 fight, and what the seat actually wants

### Jill Tarter — radio astronomy · BUCKET A
**The 2026 fight.** Breakthrough Listen's pipelines now scan petabytes and the bottleneck is the same
as ever, scaled up: separate a real narrowband technosignature from machine-generated RFI in near-real
time, and *prove the candidate isn't a pipeline artifact* — the move her career is built on.
**Frontier data.** A committed slice of real GBT/BL filterbank (`.fil`) wide enough to contain RFI
combs and a faint injected narrowband drift — not the single tiny Voyager coarse channel, but a
multi-coarse-channel slice in the tens of MB.
**Does the seat still want it? Yes — this is the engine's home turf.** holostuff's `stream_recognize`
(Wald SPRT) decides in *fewer samples* than a fixed window; `recognize` returns a calibrated
false-alarm p-value; `recognize_batch` (BH-FDR) controls the look-elsewhere effect across channels. The
SOTA ML detectors do not natively give a calibrated decide-when-enough-evidence stop.
**Bar.** On the slice, reach a target (α, β) error pair in fewer expected samples than a matched-filter
fixed-N detector, and abstain on the RFI-only channels.
**Loud negative.** holostuff is *not* a Doppler-drift engine — a drifting cue whose match score wanders
needs a matched-filter bank over candidate drift rates (the field's own fallback), which is not built.
It is a triage layer behind a spectrometer, not a spectrometer.

### Andrew Siemion — SETI · BUCKET A
**The 2026 fight.** "Flag anything that isn't noise" across an astronomical candidate count without
prescribing the signal — and not drowning in false positives when you scan billions of channels.
**Frontier data.** HTRU2 (direct) as the labeled stress test plus a committed BL candidate slice for
the unsupervised-discovery half.
**Does the seat still want it? Yes, and it sharpens into one faculty.** The single genuinely-unbuilt
honesty-arc item is the **`scan` faculty**: SPRT-per-channel *and* FDR-across-channels in one pass —
streaming detection with look-elsewhere control combined. Every part it needs (SPRT, `bh_fdr`, the
coverage check) shipped in Tier 0; it is pure assembly.
**Bar.** Label-free, FDR-controlled triage whose realized false-alarm rate tracks the stated α on
held-out noise, at a candidate count large enough that uncontrolled 2σ luck would swamp it.
**Loud negative.** A purpose-trained, tuned classifier wins raw per-candidate accuracy; holostuff's
contribution is the calibrated, deterministic, air-gappable *control*, not the best raw detector.

### David Baker — protein folding · BUCKET B
**The 2026 fight.** De novo binder design has been reinvented by deep learning: BindCraft (AF2-guided,
10–100% experimental success depending on target), RFdiffusion3 (atom-level design at scale), with
ProteinMPNN having largely displaced Rosetta for sequence design. And the *open* problem is precisely a
ranking one — there is still no reliable metric that correlates with binder affinity, so predicting
which designs will work is the unsolved bottleneck.
**Does the seat still want the engine pointed here? Honestly, no — not at the real problem.** holostuff
has no protein model, no learned affinity predictor; pointing it at de novo binder design would be
theater. What Baker's *real method* — Rosetta fragment assembly, global structure from a library of
local motifs under an energy — recognizes in the engine is the **abstraction**: combinatorial search
over role-bound fragment bundles under a cleanup energy. So the seat is re-scoped to that abstraction
on a controlled problem with a known energy.
**Frontier-adjacent data.** A committed fragment library plus a set of assembly targets where the
optimal placement is computable, so the search can be scored honestly.
**Bar.** holostuff's VSA-native fragment assembly matches a classical combinatorial-search baseline at
equal cost on the controlled targets.
**Loud negative — the loudest in the program.** This does not touch the frontier. BindCraft and
RFdiffusion3 own de novo design; `assemble`'s energy is an explicit stand-in (the pluggable cleanup
energy, A3, is unbuilt); and a real test on real proteins would need affinity-labeled data and a real
fragment library far beyond a 100 MB slice. The honest claim is "a fast, interpretable, deterministic
pre-screen abstraction," nothing more.

### Bruno Olshausen — neuroscience · BUCKET C
**The 2026 fight.** Factor a *real, cluttered* visual scene into its parts — the resonator-network
program ("vision as inference") pushed from clean synthetic products toward natural images.
**Frontier data.** A committed set of composed scenes with real clutter (object crops over varied
backgrounds), harder than clean MNIST products.
**Does the seat still want it? Yes, as a deterministic testbed — with eyes open about the ceiling.**
`decompose_structure` (the B2 SBC resonator) factors object × position × colour × scale; the energy
cleanup is already inlined as its annealed-softmax step; the remaining real work is a **calibrated soft
convergence confidence** (A2) for noisy/approximate inputs, where the current exact `validated`
certificate is brittle.
**Bar.** Factor strictly more (factors × alphabet) at fixed dimension than the dense resonator, with a
calibrated confidence that correctly flags the non-converged cases.
**Loud negative.** The operational-capacity cliff is real: the resonator stalls well before the clutter
and occlusion of an actual natural scene, where convolutional sparse coding + a much larger network is
needed. holostuff is a clean model of the mechanism, not a scene parser for the wild.

### Kyle Cranmer — particle physics · BUCKET A
**The 2026 fight.** Model-independent new-physics searches at the LHC: anomaly detection and
out-of-distribution flagging on jets, where the field is actively building *uncertainty quantification*
into taggers (evidential deep learning, OOD detection on JetNet/JetClass) — because a tagger that can't
say "this is unlike anything I trained on" is dangerous in a discovery search.
**Frontier data.** A committed 100k-jet slice of **JetNet** particle clouds (~36 MB) — light-quark/gluon
as background, top/W/Z as the held-out "signal" the detector must flag without having been told its
shape.
**Does the seat still want it? Emphatically yes — this is the engine's epistemics meeting a live
need.** `recognize` (calibrated p), `recognize_batch` (FDR), `stream_recognize` (SPRT), and
`calibration_report` give exactly the calibrated-OOD-with-look-elsewhere-control the jet-UQ literature
is reaching for, and the new question Tier 0 raised — does the false-alarm rate stay at α as the store
**grows** under capacity pressure? — is the merged capacity-and-calibration-coverage diagnostic (A6).
**Bar.** Calibrated false-alarm control at the stated α on the held-out signal jets, validated to hold
as the prototype store grows, with FDR across a realistic candidate count.
**Loud negative.** A boosted tree or a trained deep tagger wins raw tagging AUC; holostuff's claim is
the *trustworthy* false-alarm rate and the look-elsewhere control, not the best discriminator.

### Matt Pharr — 3D / raytracing / retrieval · BUCKET A
**The 2026 fight.** This seat's acceleration-structure expertise now lives in the RAG era: billion-scale
filtered ANN (FAISS, DiskANN, CAGRA, ScaNN) is mature and fast — and the live pain is **trustworthy
retrieval**, knowing when the nearest neighbour is actually garbage so a downstream LLM doesn't
hallucinate on a bad hit. That is exactly his old grazing-ray question — "did the traversal find
something, or am I guessing?" — at internet scale.
**Frontier data.** A committed 100k-vector GloVe-100d slice (~40 MB), queried both with in-distribution
and deliberately out-of-distribution queries that *have no good match*.
**Does the seat still want it? Yes — for the abstention, not the speed.** `recall` runs through the
sublinear HoloForest; `recall_calibrated` (now routed through that same forest, Tier 0) returns a
calibrated "no good match — abstain," complementing the forest's cross-tree agreement.
**Bar.** On OOD queries, abstain at a calibrated rate (don't return a confident garbage neighbour);
on in-distribution queries, recall correctly at sublinear cost.
**Loud negative.** FAISS / HNSW / DiskANN crush the HoloForest on the speed–recall Pareto by orders of
magnitude. holostuff's *only* real claim here is the calibrated abstention — a complementary signal you
could bolt onto a fast index, not a replacement for it.

### Andrew Adamatzky — unconventional computing · BUCKET C
**The 2026 fight.** Beyond mazes: efficient transport-network design — the problems his slime moulds are
famous for, the Tero *Tokyo rail* multi-source/multi-sink model, fault-tolerant network synthesis.
**Frontier data.** A committed real city/transport graph (OSM extract or a DIMACS road slice, tens of
MB) with multiple terminals.
**Does the seat still want it? Yes, but it depends on unbuilt work.** `solve_maze` is already the
deterministic Tero flow model — a faithful in-silico Physarum. The frontier needs the extension from
single-source/single-sink to **multi-terminal network design** (A9), returning the network as a typed
structure.
**Bar.** Synthesize an MST-cost-competitive network on the real graph with measurably higher fault
tolerance than a minimum spanning tree (the Physarum trade-off Tero measured).
**Loud negative.** A9 is unbuilt; and a dedicated Steiner-tree solver wins on raw total length.
holostuff's pitch is the biological trade-off — robustness for a little extra length — deterministically
reproduced, not the shortest network.

### Iñigo Quílez — demoscene / procedural · BUCKET C
**The 2026 fight.** Representing detailed geometry and texture compactly — the world has moved to neural
fields / implicit neural representations (INRs) and learned SDFs that fit a tiny network to a shape.
**Frontier data.** A committed high-resolution texture slice (DTD or Brodatz), self-similar and
detailed.
**Does the seat still want it? Yes, for the structural aesthetic.** holostuff's procedural generation
(morph / nucleus / compose / nested scenes) and `generate_vector` build scenes-of-scenes from one
bind+superpose primitive; the real next step is a **recursive/fractal scene generator from a single
seed vector** with `fractal_dimension` reported (A12) — maximal richness from a tiny deterministic
kernel, the demoscene creed.
**Bar.** On self-similar textures, higher reconstruction PSNR-per-byte than JPEG at matched bytes;
report `fractal_dimension` on the generated result.
**Loud negative.** Only self-similar content benefits — compression is structure-dependent, and the
engine is honest about it. A trained INR represents arbitrary detailed geometry that holostuff's
fixed algebra cannot.

### Julian Togelius — game AI / PCG · BUCKET A
**The 2026 fight.** Agents you can actually ship and trust, and the open-endedness / world-model push —
NPCs that learn online without catastrophic forgetting and that *know when they don't know*, rather than
a black-box policy that acts confidently off-distribution.
**Frontier data.** A committed larger level corpus plus agent trajectories (VGLC levels + a trajectory
slice) — enough to exercise online learning and out-of-distribution states.
**Does the seat still want it? Yes — explainability + calibrated abstention is rare and valuable.**
`decide` / `reinforce` / `actions` is the deterministic, self-explaining creature brain;
`decide_confidence` (Tier 0) turns its raw support into a false-alarm p-value and `explore_if_
unrecognized=α` makes it explore safely when its state recognition is noise-level — the honesty arc
carried from perception into action.
**Bar.** Reactive parity with a small DQN on the levels, *plus* a calibrated, explainable abstention on
novel states the DQN handles silently (and wrongly).
**Loud negative.** The brain is reactive — it does not plan; a planning agent or a large RL policy beats
it where lookahead matters. Its edge is being debuggable, online, forgetting-free, and honest about
uncertainty, not being the strongest player.

### Miles Macklin — physics simulation · BUCKET B
**The 2026 fight.** Fast, accurate, *differentiable* simulation — learned simulators (graph-network
simulators), neural cloth/fluid/soft-body, and the production reality of robust constraint solving.
**Frontier data.** A committed multi-clip AMASS/CMU mocap (BVH) slice.
**Does the seat still want it? Partly, and honestly scoped.** `learn_dynamics` (prediction as one bind)
is wired and the rigid-shift-as-binding is in the kernel; the genuinely interesting unbuilt idea is a
single **iterate-a-projection faculty** (A4) unifying resonator + denoise + dynamics as "project onto
constraints" — because his alternating constraint projection, the resonator's alternating projection,
and the PnP loop are the same object. Plus the determinism / tie-break audit his `bind_batch` lesson
demands, now covering the new calibrated/null code paths.
**Bar.** VSA-native motion-compensated representation with a bit-exact determinism audit clean
run-to-run; the unified projection faculty demonstrated across all three modules.
**Loud negative.** H.264 wins motion compression; a trained graph-network simulator wins physical
accuracy. holostuff offers an O(1)-advancement representation and a unifying abstraction, not a faster
or more accurate simulator.

### Jos Stam — fluids / PDEs · BUCKET A
**The 2026 fight.** Neural PDE surrogates have exploded — and FNO is now the **baseline**, not the
frontier. The 2026 frontier is foundation operators (DPOT, OmniArch, AMR-Transformer) and the known
failure modes: FNO rings on shocks, neural surrogates struggle with high-amplitude discontinuities, and
long autoregressive rollouts go unstable. A trained FNO already beats persistence *and* mean.
**Frontier data.** PDEBench as the published reference point; the actual field **generated locally** by
a committed FFT-on-a-torus pseudospectral 2-D Navier–Stokes solver — Stam's own method, the `[gen]`
exception.
**Does the seat still want it? Yes — but the claim is not accuracy.** holostuff's learned bind is a
*linear* Koopman/DMD operator in Fourier coordinates — even further from the frontier than FNO. Its
durable, genuine property is the one neural operators do **not** offer: an **exact, invertible
round-trip**, making the trajectory content-addressable — recover the state k steps ago, or query "the
state just before the regime change." That is real and current and unique.
**Bar.** On the generated field, beat persistence *and* mean at multi-step prediction (closing the loop
the market-data negative left open) *and* preserve an exact trajectory round-trip a neural operator
cannot.
**Loud negative.** A trained FNO — let alone a foundation operator — is far more accurate. holostuff is
DMD-in-Fourier with content-addressability, not a competitive surrogate.

### Aydogan Ozcan — medical imaging · BUCKET A
**The 2026 fight.** Computational microscopy reconstruction and virtual staining are deep-learning-
owned; the live engineering reality is reconstructing a clean image from a *degraded* sensor
measurement and being robust when part of the measurement is missing.
**Frontier data.** A committed real cell-microscopy slice (BBBC / malaria cells), larger and messier
than the tiny BCCD set.
**Does the seat still want it? Yes — erasure-robustness and reconstruction-as-inverse-problem.**
`splat_archive` and the WHT archive store and reconstruct under degradation; the real wiring is a
**PnP/RED restoration faculty** (A5, refocused to a measured demo since the `pnp` machinery exists):
use `denoise` as the prior in a loop to inpaint an erased archive plate, and validate the σ estimate.
**Bar.** On a real erased plate, the PnP/RED loop beats a single denoise at reconstruction quality, with
the noise estimate honestly checked.
**Loud negative.** A trained reconstruction network wins raw image quality; manifold denoising
over-smooths at low noise (a kept negative — it needs the noise level). holostuff's claim is graceful
degradation under erasure and reconstruction-as-a-solved-inverse-problem, not the best image.

### Miller Puckette — audio · BUCKET A (recall) / C (codec)
**The 2026 fight.** Neural audio codecs (EnCodec, DAC, RVQ-based) and LLM/diffusion music source
separation on MUSDB18-HQ own the field; the phase vocoder is now the classical substrate underneath.
**Frontier data.** A committed MUSDB18-HQ stem slice or an NSynth instrument slice.
**Does the seat still want it? Yes — for content-addressable sound memory, not codecs.** holostuff's
FHRR (`high_capacity_memory`) is Puckette's representation exactly — complex phasors, unit magnitude,
phase carrying the information — and `bind` is an FFT round-trip, the phase vocoder's home. The honest
test is the capability codecs and separators *don't* provide: encode a spectrum as an FHRR hypervector,
bundle a chord, **recall an instrument by content**; plus `learn_dynamics` on an audio-frame sequence,
the dynamics proving ground B4 explicitly flagged.
**Bar.** Content-addressable spectral recall (recall the right instrument from a degraded cue), and
audio-frame dynamics that beat persistence *and* mean.
**Loud negative.** EnCodec/DAC win compression; Demucs/BandSplit-RoFormer win separation. holostuff
does neither — it offers associative recall over sound, a different and narrower capability.

### Jarek Duda — file compression · BUCKET A
**The 2026 fight.** Neural compression (learned image/video codecs) is the headline; Duda's own ANS is
in Zstandard, and the live question for *distributed representations* is how few bits preserve the
**decision geometry** (the cosines that drive recall) in the embedding stores that now back every RAG
system.
**Frontier data.** The shared GloVe-100d store (with Pharr) — real embeddings whose geometry must
survive compression.
**Does the seat still want it? Yes — geometry-preserving compression of vectors.** The B5 rate-distortion
code (consolidation/KLT → water-filling → rANS) is in the kernel save (`quant='rd'`) and the mind's
`save` now auto-selects it for low-rank arrays (Tier 0). The genuine remaining diagnostic is reporting
**bits/vector at fixed cosine fidelity** as a save-time readout.
**Bar.** Fewer bits per vector at a fixed cosine fidelity than int8 on the embedding store.
**Loud negative.** High-entropy, full-rank vectors don't compress — the win is real only where low-rank
structure exists; and a learned codec wins on images. holostuff's frontier is the narrow, real one of
geometry-preserving distributed-representation coding.

### Tony Plate — VSA / HRR · BUCKET A
**The 2026 fight.** Knowledge-graph embedding has a leaderboard (ComplEx, RotatE, AutoSF+, ConEx) and a
move toward KG foundation models — and a real-stakes application: biomedical link prediction for **drug
repurposing** on large graphs like ogbl-biokg. HolE — Plate's own foundation, HRR for KGs — is now a
classical baseline the leaderboard has passed.
**Frontier data.** A committed ogbl-biokg triple slice (~57 MB packed) — 51 relations, ~94k entities,
millions of training triples, with genuine drug-repurposing stakes.
**Does the seat still want it? Yes — to measure the foundation honestly on a real graph.** holostuff's
bind/cleanup *is* HolE's circular-correlation operation; this test runs the engine's own foundation on a
weighty biomedical graph and adds a **capacity / SNR diagnostic** (A6) — each store's operating point
versus the noise-wins cliff, the "capacity cliff shown, not hidden" made a live readout.
**Bar.** HolE-ballpark MRR on the ogbl-biokg slice, with a live capacity readout grounded in HRR
capacity theory.
**Loud negative.** ComplEx/RotatE/AutoSF+ beat HolE on MRR — holostuff will not top the leaderboard.
The value is an honest measurement of the foundation, on a real graph with real stakes, with the
capacity mathematics Plate wrote exposed as a diagnostic.

### Miles Stoudenmire — tensor networks · BUCKET B
**The 2026 fight.** Tensor networks for ML and for classically simulating quantum systems — MPS/DMRG
truncation as a compression and learning tool.
**Frontier data.** MNIST / Frey faces (committed) — real data with exploitable low-rank structure.
**Does the seat still want it? As research, honestly labelled.** `consolidation` (SVD/KLT) already *is*
a tensor-network truncation by another name, and SBC is wired. The speculative ask is an optional
**tensor-train (MPS) bind mode** (A14) with a capacity comparison against HRR convolution — the
heaviest, least-certain item in the whole program.
**Bar.** Show that the consolidation truncation matches a tensor-network truncation on the real data;
*if* the MPS-bind mode is built, compare its binding capacity to HRR convolution.
**Loud negative.** This is a research direction, not a benchmark holostuff is expected to win. The
MPS-bind mode may not raise capacity at all — it is on the list precisely so the negative can be put on
record if it doesn't.

### Peyman Milanfar — denoising / inverse problems · BUCKET A
**The 2026 fight.** Real-world sensor-noise denoising on SIDD/DND is owned by deep nets (Restormer
~40 dB, NAFNet, MIRNet) and self-supervised / blind-spot / diffusion methods; BM3D is the historical
floor neural nets were first asked to beat in 2012. The frontier is *blind, signal-dependent* noise
without clean pairs.
**Frontier data.** A committed slice of SIDD validation patches — real smartphone sensor noise, the
current benchmark, far harder than synthetic-Gaussian Set12.
**Does the seat still want it? Yes — as the denoiser-as-prior, his actual thesis.** holostuff's
`denoise` (manifold / NLM-via-recall / PnP-RED / codebook) embodies RED's claim that a denoiser is a
manifold map you can plug into any inverse problem. The honest pairing: match classical NLM in the
high-noise regime, and — more to the point — serve as the **prior inside a PnP/RED loop** (the inverse
problem is the contribution, not the raw PSNR).
**Bar.** Match classical NLM/BM3D-lite in the high-noise regime on the SIDD slice, and demonstrate the
PnP/RED loop solving an inverse problem (inpainting) better than a single denoise.
**Loud negative.** Restormer/NAFNet and diffusion restorers beat holostuff badly on real-noise PSNR —
it is not in the deep-net league; and fixed-rank manifold denoising over-smooths at low noise (the kept
negative — it needs the noise level). The claim is the RED *framework* embodied, not state-of-the-art
denoising.

### George Drettakis — Gaussian splatting · BUCKET C
**The 2026 fight.** 3DGS is everywhere, and the 2026 frontier is squarely a **compression** problem:
rate-distortion-optimal coding of large and *dynamic* (4D) splat scenes for streaming and storage —
CAT-3DGS, HAC, LightGaussian, MPEG-GS standardization, the glTF `KHR_gaussian_splatting` interchange
standard expected Q2 2026.
**Frontier data.** A committed real splat-scene slice (a PLY from a public capture) plus a 2-D density
field for the proof-of-concept.
**Does the seat still want it? Yes — for the structural insight, which is genuinely current.** A splat
scene *is* a bundle (superposition of primitives), and splat compression *is* the rate-distortion
problem holostuff's B5 code addresses; `splat_field` / `splat_archive` make a scene a content-
addressable bundle of role-bound Gaussians with region query and progressive refinement.
**Bar.** On a 2-D field, match or beat the WHT-plate archive at fixed bytes while adding region-query
and progressive refinement — a proof-of-concept that bundle = splat scene and the RD code applies.
**Loud negative — a Bucket-C confession.** holostuff is **isotropic and 2-D**; real 3DGS is anisotropic,
3-D, view-dependent, and millions of Gaussians, and CAT-3DGS/HAC win the actual compression. Anisotropic
splats and a 3-D extension (A13) are deliberately deferred. The structural insight is real and the
engine is two big steps from the frontier — that is the honest claim.

### Brian Eno — generative art · BUCKET C
**The 2026 fight.** Generative music and art with genuine structure — modern generative models (music
transformers, diffusion) produce rich, long-range-coherent output.
**Frontier data.** A committed larger MIDI/chorale corpus (Lakh slice or Nottingham + JSB Chorales).
**Does the seat still want it? Yes — for generation-over-a-composed-manifold.** `generate_vector` is the
B10 holographic-diffusion: run the denoiser backwards from noise and structure emerges — generation and
denoising as the same operation in different regimes. The real, scoped next step is **generation over a
*composed* subspace** (A11): produce novel-but-valid *structures*, not verbatim stored atoms.
**Bar.** Generate chorale/MIDI structures that are valid (high calibrated recall) *and* novel (not a
stored atom), sampling over a composed manifold rather than the bare codebook.
**Loud negative.** Bare-codebook generation is a degenerate sampler (a kept negative); and a trained
music model wins richness and long-range coherence outright. The contribution is the unification —
recall, denoise, generate as one operation — not state-of-the-art generation. *And the meta-lesson worth
keeping, in Eno's own Oblique-Strategies spirit: the coherence-gate win came directly from the
calibrated-novelty negative — honour thy error as a hidden intention.*

---

## How to sequence this

The build order from the first program still holds, re-tiered by where the honest value is:

1. **Bucket A, honesty-arc seats first** (Tarter, Siemion, Cranmer, Pharr, Plate, Duda) — these are
   pure data-harness over shipped faculties plus the `scan` faculty (A1) and the merged
   capacity-and-coverage diagnostic (A6). They land once the fetcher and the first committed slice are
   in. Highest value, lowest new code.
2. **Bucket A, modality seats** (Stam local-gen + invertible round-trip; Ozcan PnP demo A5; Puckette
   audio modality A7; Togelius already shipped via `decide_confidence`).
3. **Bucket C proofs-of-concept** (Drettakis 2-D RD splat; Quílez fractal-from-seed A12; Eno
   composed-subspace A11; Adamatzky multi-terminal A9; Olshausen calibrated resonator confidence A2).
4. **Bucket B, scoped/research** (Baker fragment-assembly abstraction + A3; Macklin unified projection
   A4 + determinism audit; Stoudenmire MPS-bind A14) — last, honestly labelled, negatives expected.

Each landed test keeps the close-out ritual: a `tour.py` line, a `NOTES_concepts.md` entry recording
what the data said — the win *and* the loud negative, both — README counts, an **end-to-end integration
test through `UnifiedMind`** (not an import check — the lesson that naive cross-module chaining once
regressed still stands), and a clean zip rebuild verified from extraction. Every claim carries "on an
N-sample slice of \<dataset\>" and the subset script ships in the repo. Backward-compatible defaults;
negatives on the record.

---

## The honest bottom line

Moose was right that the first pass read like a hobby project, and the fix was not to find datasets
where holostuff wins — it was to point each seat at the problem its field is *actually* fighting in
June 2026 and measure honestly. Done that way, the result is humbling in the engine's usual style:
against BindCraft and RFdiffusion3, against foundation PDE operators, against Restormer and the neural
codecs, against CAT-3DGS and FAISS and the KGE leaderboard, **holostuff does not win the headline
metric, and the program says so, loudly, on every seat.**

What it wins is narrower, real, and current: a *calibrated* false-alarm rate where the SOTA tagger only
gives a score (Cranmer, Tarter, Siemion); *trustworthy* abstention where the fast index only gives a
neighbour (Pharr); an *exact invertible* trajectory where the neural operator only gives a prediction
(Stam); *graceful degradation under erasure* where the reconstruction net only gives an image (Ozcan);
*content-addressable* recall where the codec only gives bytes (Puckette, Plate); *geometry-preserving*
compression where the codec only preserves pixels (Duda); a *deterministic, explainable, uncertainty-
aware* agent where the policy only gives an action (Togelius). And three seats honestly conceded — the
protein frontier, the anisotropic-3-D splat frontier, the tensor-network bind — kept as scoped
abstractions or proofs-of-concept with the negative on record rather than dressed up as wins.

That is the project's method scaled to a frontier program: ground every test in a field's real 2026
state of the art, on a real (if subset) dataset that field is benchmarked on, name the property
holostuff uniquely contributes, and write down — plainly, in the bar and the negative — exactly where
the data, and the deep nets, say it loses. The bigger goal is not beating AlphaFold. It is knowing,
precisely and honestly, where a deterministic, calibrated, content-addressable VSA engine earns its
seat at a table full of tools that outscore it.

---

### Datasets, methods, and the 2026 frontier this program leans on
- **Tarter / Siemion:** Breakthrough Listen ML pipelines; Wald SPRT; Lyon et al. (2016) pulsar
  candidate selection (HTRU2); BH-FDR look-elsewhere control.
- **Baker:** BindCraft (Pacesa et al., *Nature* 2025, 10–100% binder success); RFdiffusion3 (atom-level
  design at scale, 2026); ProteinMPNN displacing Rosetta; the open affinity-ranking problem; Rosetta
  fragment-assembly as the tested abstraction.
- **Olshausen:** Kymn et al. (2024) compositional factorization; resonator networks; convolutional
  sparse coding for cluttered scenes.
- **Cranmer:** JetNet / JetClass (Kansal et al.); evidential deep learning and OOD detection for jet
  tagging (2025); look-elsewhere / trials-factor methodology.
- **Pharr / Duda:** ANN-Benchmarks (Aumüller et al.); FAISS (Douze et al. 2025), DiskANN, CAGRA;
  filtered ANN for RAG; GloVe (Pennington et al. 2014); Duda ANS; KLT water-filling, rate-distortion.
- **Adamatzky:** Tero et al. (2010) *Rules for Biologically Inspired Adaptive Network Design* (Tokyo
  rail); MST/Steiner baselines; OSM/DIMACS road networks.
- **Quílez:** neural fields / implicit neural representations; SDF raymarching; box-counting fractal
  dimension; DTD/Brodatz textures.
- **Togelius:** VGLC level corpus (Summerville et al.); online RL / open-endedness; standard PCG
  playability metrics.
- **Macklin:** AMASS / CMU mocap; position-based / XPBD; graph-network simulators; H.264 motion
  compensation.
- **Stam:** Stam (2001) FFT fluid solver; PDEBench (Takamoto et al. 2022) and the FNO-as-baseline /
  foundation-operator frontier (DPOT, OmniArch); Koopman / DMD.
- **Ozcan:** BBBC / malaria-cell microscopy; Venkatakrishnan et al. (2013) PnP; Romano, Elad, Milanfar
  (2017) RED; deep-learning computational microscopy.
- **Puckette:** MUSDB18-HQ (Rafii et al.); EnCodec / DAC neural codecs; Demucs / BandSplit-RoFormer
  separation; the phase vocoder.
- **Plate:** ogbl-biokg (OGB, Hu et al. 2020); Nickel, Rosasco, Poggio (2016) HolE; ComplEx/RotatE/
  AutoSF+ leaderboard; drug-repurposing link prediction.
- **Stoudenmire:** Stoudenmire & Schwab (2016) supervised learning with tensor networks; MPS/DMRG
  truncation; MNIST/Frey.
- **Milanfar:** SIDD (Abdelhamed et al. 2018) / DND real-noise benchmarks; Restormer, NAFNet,
  self-supervised/diffusion denoising; Buades et al. (2005) NLM; Dabov et al. (2007) BM3D.
- **Drettakis:** Kerbl et al. (2023) 3D Gaussian Splatting; CAT-3DGS, HAC, LightGaussian, MPEG-GS and
  the glTF splat standard (2026); a real splat scene + 2-D field.
- **Eno:** JSB Chorales (Boulanger-Lewandowski 2012) / Lakh / Nottingham MIDI; Ramsauer et al. (2020)
  modern Hopfield (the generative-denoising engine); *Oblique Strategies*.
