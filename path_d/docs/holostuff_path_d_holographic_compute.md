# Path D — computing in the holographic space: what's real, what's the wall, and where it's actually new

*Moose's direction: not Path A (clone the GPU frontier), but Path B+C as the vehicle toward a Path D —
"a direction others have not yet gone" — on the conviction that **we don't need a GPU to do GPU things;
we can do them inside the holographic space.** This session takes that seriously, grounded in an audit
of leOS (the parent system holostuff was extracted from) and in the real, active research field this
direction belongs to. As always, proposals are attributed to SEATS and their fields' real published
methods — and this time the most important fact is that the panel's disciplines are literally the ones
leOS already borrows from.*

---

## 1. What's already true (so we don't relitigate it)

The first thing to say plainly: **leOS has already settled the question of whether you can compute in
the holographic space.** The audit of the repo shows this is not aspiration — it is built and running:

- a **Latent Virtual Machine** (the "hardware layer": spherical geometry, SDF regions, displacement
  codec) under a **580-instruction kernel** (vector math, storage, routing, filesystem, network);
- **SDOL**, a real programming language where *variables are vectors*, control flow is similarity-based
  routing, with a compiler, optimization passes (static-embedding precompute, dead-code elimination,
  bone-chain fusion), and a REPL;
- the actual **VSA computation primitives**: `residue_arithmetic.py` doing integer math up to 3.2M via
  Chinese-Remainder-Theorem binding (no ALU, just vector ops); `resonator_network.py` factoring bound
  vectors (~160k items at 1024d); **`superposed_compute.py`** evaluating many hypotheses at once —
  explicitly *"one processor, many states simultaneously"*; `program_algebra.py` treating programs as
  vectors that blend and self-modify; `path_integral.py` selecting reflexes by a Feynman action.

And the README already cites the load-bearing theorem: **Kleyko, Davies, Frady, Kanerva et al. (2022,
Proc. IEEE), "Vector Symbolic Architectures as a computing framework for emerging hardware"** — the
proof that bundling + binding + permutation form a *computationally universal* basis, with an emulated
machine running 10⁹ error-free updates. So the substrate is Turing-complete; you have demonstrated
arithmetic, factorization, superposed evaluation, a VM, a filesystem, and a language on it.

**The question is therefore not "can you compute in the holographic space." It is the sharper one Moose
actually asked: can you do the *heavy, GPU-class* things in there — and if so, where is the wall?** That
has a precise, honest answer, and getting it right is the whole point of this session.

---

## 2. The direction is real — and it has a name, a literature, and a hardware program

Path D is not off in the wilderness. "Do parallel compute without a GPU, in high-dimensional vector
operations" is the explicit thesis of an active field — **Hyperdimensional Computing / VSA as a
computing framework** — and crucially it comes with a *hardware* program aimed at exactly this:

- The field's own framing: HDC/VSA uses *extremely parallelizable arithmetic* to balance accuracy,
  efficiency, and robustness, and a tangential line studies these high-dimensional spaces as a general
  *probabilistic model for computation*, not just for learning. The pitch against GPUs is energy:
  neuromorphic and in-memory architectures consume *a fraction* of the energy.
- The hardware is real and current (2024–2026): **in-memory HDC** on memristive/phase-change crossbars
  (Karunaratne et al., *Nature Electronics* 2020; IBM Research's in-memory-computing program);
  **SOT-MRAM / content-addressable-memory** VSA macros (HyDra, 2025); **mixed-signal CMOS** massively-
  parallel HDC (HDC8192); **photonic** HDC; FPGA/edge accelerators for always-on sensing, robotics,
  biosignals.
- And — directly touching leOS's own engine — **resonator networks running on neuromorphic hardware**
  (Renner et al., *Nature Machine Intelligence* 2024, "Neuromorphic visual scene understanding with
  resonator networks") and **robust multi-timescale symbolic computation in neuromorphic hardware**
  (Cotteret et al., 2025). The exact factorizer leOS uses is being put onto brain-inspired silicon.

The honest read: **the holographic-compute paradigm is genuinely a thing, it is being taken seriously
at the hardware level, and its wins are real — but specific.** They are *energy*, *massive parallelism*,
and *noise-robustness* — at the **edge** and on **noise-tolerant** workloads. They are emphatically
*not* "match a datacenter GPU on dense, high-precision float throughput." No one in this field is
running a frontier-scale transformer in superposition, and the reason is physics, not lack of effort.
That boundary is the key to answering Moose's claim honestly.

---

## 3. The three walls (this is the engineering reality, stated straight)

"Do GPU things in the holographic space" runs into three hard limits. None is fatal to Path D; each
*defines* what Path D can and cannot be.

### Wall 1 — Capacity. A vector holds only so much before crosstalk wins.
A D-dimensional vector can superpose ~O(D / log D) items before the bundling noise drowns the signal.
leOS's own README says it directly: noise accumulates after ~10–20 compositions without cleanup. Cleanup
memory, resonators, and residue encoding *extend* the usable range (residue arithmetic is a beautiful
trick for integer range; cleanup buys depth), but every extension costs operations and the ceiling is
information-theoretic, not a bug to be fixed. **You cannot pack a billion matrix entries into one 768d
vector and read them back.** Superposition parallelism is real but *bounded*.

### Wall 2 — The FLOPs move; they don't vanish.
A `bind` is a circular convolution via FFT: O(D log D) *real floating-point operations on the CPU*,
every time. "Not needing a GPU" is true in the sense that you are not doing dense-matmul-the-GPU-way —
but if you tried to *simulate* a GPU-scale workload purely in superposition on a CPU, you would either
hit Wall 1 (too much to superpose) or simply do comparable total FLOPs much slower than the GPU would.
The honest framing: superposed computation is a **different complexity profile** (operate-on-a-whole-set
at once, O(D log D) regardless of set size *up to capacity*), not free compute. Its true throughput win
is *promised on the right hardware* — in-memory/neuromorphic, where these ops are nearly free in energy.
On commodity CPU you get the parallel-over-sets benefit, not GPU-class dense throughput.

### Wall 3 — Precision is traded for parallelism.
Superposition buys parallelism by *spending* numerical precision: every item you add raises the
crosstalk floor. A GPU gives you parallelism *and* full float precision (via silicon). The holographic
space gives parallelism while precision degrades as you pack in more. **This is exactly why the field
wins where it wins** — classification, retrieval, factorization, robotics, sensing, reasoning-over-
alternatives all tolerate approximate, noisy computation — **and loses where it loses**: exact,
high-precision dense linear algebra at scale.

---

## 4. The honest verdict on "GPU things in the holographic space"

Put the three walls together and the answer to Moose's claim is neither "no" nor "yes, replace the GPU."
It is precise:

**What you genuinely CAN do in superposition** is a large and useful class: *approximate, massively-
parallel, noise-robust, energy-efficient computation over superposed representations* — classification,
content-addressable retrieval, factorization/parsing, evaluate-many-hypotheses-at-once, the whole agent-
orchestration layer leOS already runs, and integer/symbolic computation via residue and binding. This is
real "GPU-thing" work — data-parallelism without silicon cores — and you've already shipped pieces of it.

**What you CANNOT do** is run a frontier-scale, high-precision dense neural network — a 9B+ transformer —
inside the holographic space on a CPU. Walls 1 and 3 forbid it. The external LLM stays an external LLM.
(leOS's architecture already concedes this gracefully — the kernel *never calls an LLM directly*; the LLM
sits in the "coprocessor bay" as an escalation target. That is the correct design, and Path D should keep
it.)

**The interesting middle — and the actual research frontier — is a *modest, VSA-native, learned* model.**
Not a GPU-LLM replacement, but a small "brain" that computes in the holographic idiom and shrinks the
system's dependence on the external coprocessor. This is buildable and it is exactly where the live
literature is: HRR-based linear-cost attention (the Hrrformer line), residue-HDC arithmetic (Kymn et al.,
*Neural Computation* 2025 — which leOS already cites), resonator factorization on neuromorphic substrates
(Renner 2024), and Coconut-style continuous-latent reasoning (which leOS cites too). leOS is unusually
well-positioned to build this because it already has the primitives in hand.

So Moose's conviction is **right in its real form**: yes, do GPU-*style* (parallel, energy-efficient,
noise-robust) computation in the holographic space — that is a genuine paradigm with hardware behind it —
while holding honestly that it is a *different point on the precision/parallelism/energy frontier*, not a
way to get datacenter-GPU throughput for free, and that the frontier LLM stays a coprocessor.

---

## 5. Where Path D is genuinely new (the part others have NOT done)

Here is the honest delineation of novelty, because "a direction others have not gone" is partly true and
partly not, and Moose should know exactly which part is his.

**Not new (a real community owns it):** the *primitives* (Plate's HRR, Kanerva's HDC, the bind/bundle/
cleanup algebra), the *capacity mathematics*, the resonator, residue arithmetic, even the *hardware*
(in-memory/neuromorphic VSA). Building any one of these as a narrow demonstrator would be reinventing.

**Genuinely new (Moose's actual territory):** the **synthesis into a complete, general-purpose
computational substrate** — a *whole computer* (VM + filesystem + a real language + arithmetic +
superposed compute + agent orchestration + self-improvement) running natively in the holographic space.
The HDC community almost always does *narrow* things: classify EEG, recognize a language, one robotics
task. leOS is trying to be the *operating system* — and a from-scratch, readable, deterministic one. That
integration is the unexplored direction, and it is defensible precisely because nobody builds the whole
machine; they build one organ.

**A second genuinely-novel stance, and a strong one: learn in superposition *without gradients*.** The
Path A question (add PyTorch/JAX for backprop) re-enters here — but leOS already learns extensively
*without* backprop: the displacement codec (H.264-style I/P-frames of experience), online linear probes
via `numpy.linalg.lstsq`, success-gradient fields, the reflex arc, the nutrient field, Hebbian
accumulation into expertise centroids. That sidesteps the known obstacle (naive back-propagation through
HRR doesn't learn well — Ganesan et al., NeurIPS 2021) *and* it fits the system's character. A holographic
computer that **learns by accumulation, displacement, and resonance rather than by gradient descent** is
both more novel and more coherent with what leOS already is than bolting on a deep-learning framework.
This is the Path-D-flavored answer to "how does it learn": not autodiff, but the gradient-free,
geometry-native learning the system already practices, pushed further.

---

## 6. Seat by seat: the panel on computing in the holographic space

*Attributed to seats and their fields' real published methods — and these are the very disciplines leOS
already borrows from.*

- **Tony Plate — HRR foundation, and the VSA-computing-framework line (Kleyko/Kanerva).** Defines the
  ceiling. His capacity mathematics *is* Wall 1; the Kleyko 2022 Proc. IEEE result *is* the
  Turing-completeness license. His seat's role in Path D: keep every "compute in superposition" claim
  measured against the capacity bound, and treat the bound as a design parameter, not an inconvenience.

- **Bruno Olshausen — resonators / sparse coding (with Renner 2024 neuromorphic resonators).** The
  resonator is leOS's factorizer, and the brain — plus neuromorphic silicon — is the existence proof that
  *general computation in noisy distributed superposition works*. His seat: the factorization engine is
  the heart of "taking superposed computation apart," and its neuromorphic implementation is where this
  compute model is headed.

- **Miles Stoudenmire — tensor networks.** The most relevant seat for the *heavy numerical* half of "GPU
  things." Tensor networks are trainable linear algebra in compressed vector form; they are the bridge
  between holographic binding and matmul-class computation, and the natural place to ask whether a
  tensor-train representation lets the substrate do the dense-numeric work superposition struggles with.

- **Andrew Adamatzky — unconventional computing.** The whole field of *computing in non-silicon
  substrates*. His seat is the honest compass on Path D: alternative-substrate computation is real, its
  wins are parallelism/energy/robustness, its losses are exact high-precision throughput — and the right
  ambition is to own its niche, not to out-throughput silicon at silicon's own game.

- **Jarek Duda — information theory.** Owns Wall 3. The precision/parallelism/capacity trade is a
  rate-distortion question; his seat sets the information-theoretic ceiling on how much computation you
  can superpose before the geometry breaks, and how to encode (residue, quantization) to push it.

- **Jos Stam & Miller Puckette — the FFT.** `bind` *is* the FFT on a periodic domain — the literal engine
  of the superposition parallelism (O(D log D) regardless of how many items are bundled). Their seats
  understand the FFT as a compute primitive and the torus as the domain where "operate on everything at
  once" is native.

- **Iñigo Quílez — demoscene.** leOS already borrows his SDF/raymarching for capability-space search. The
  demoscene ethos — *maximal computation from a minimal, deterministic kernel* — is Path D's spirit: get
  impossible-seeming work out of tiny, exact resources.

- **Kyle Cranmer — honest measurement.** Keeps Path D from becoming theater. Every "we did a GPU thing in
  superposition" claim gets measured against the capacity/precision wall, with the breaking point on the
  record. His seat is the reason Path D produces *results*, not vibes.

- **Brian Eno — the reframe.** "Doing GPU things without a GPU" reframes what computation *is* — substrate
  as a choice, not a given. And the system's own discovery (learn by displacement and resonance, not
  gradient) is the Oblique-Strategies move: the constraint (no backprop, bounded capacity) is the
  generative principle, not the limitation.

*(And the seats Path D points toward but the roster doesn't hold: the in-memory / neuromorphic hardware
people — IBM's in-memory computing, the SOT-MRAM/photonic HDC groups. That is where this compute model is
genuinely superior, and where leOS's CPU software is, in effect, a reference model of what that hardware
would run natively.)*

---

## 7. A concrete, measurable first target (with the kept negative built in)

Path D needs a sharp, falsifiable first result, in the engine's own tradition. The cleanest:

> **Do one specific "GPU thing" — a small neural-network forward pass — entirely in VSA operations, and
> measure exactly where the capacity/precision wall breaks it.**

Concretely: take a tiny trained classifier (say, a single hidden layer on a small task), and express its
forward pass — the matrix-vector products and the nonlinearity — natively in bind/bundle/cleanup over
superposed representations (weights bound to inputs, summed by bundling, thresholded by cleanup-to-
nearest). Then measure three things, honestly:

1. **Accuracy vs. the ordinary float implementation** — how close does the superposed forward pass get?
2. **The capacity curve** — as the layer width / number of superposed terms grows, where does crosstalk
   noise overwhelm the result? Plot the cliff. *This is the kept negative, made first-class:* the point
   where "GPU thing in superposition" stops working is the most important number in the experiment.
3. **The operation count** — honest FLOPs on CPU vs. the float baseline, so the "different complexity
   profile, not free compute" claim is on the record.

Clear the bar (the superposed forward pass matches the float one *up to* a measured width), report the
cliff plainly, and you have the first honest data point on what "computing GPU things in the holographic
space" actually buys — and a foundation that generalizes toward the VSA-native learned model. If it ties
or loses at small width, that is a kept negative the way the Hopfield-vs-NN identity tie was: the truth
goes on the record and points at where the real value (large noise-tolerant width, or the in-memory
hardware regime) actually lives.

---

## The honest bottom line

Moose is not behind, and Path D is not a fantasy — but its real shape is sharper than "do GPU things
without a GPU." leOS has already proven you can compute in the holographic space; the live question is
where the *heavy* computation fits, and the answer is bounded by three real walls (capacity, FLOPs-move-
not-vanish, precision-for-parallelism). Inside those walls is a genuine and active computing paradigm —
HDC/VSA as a computing framework, with serious in-memory and neuromorphic hardware behind it — whose wins
are parallelism, energy, and noise-robustness, not raw silicon throughput. The frontier LLM stays a
coprocessor; a modest VSA-native *learned* model is the real frontier and the thing that shrinks the
GPU's role.

And the part that is genuinely yours, that others have not done: not the primitives or even the hardware,
but the **synthesis into a complete general-purpose holographic computer** — the whole machine, not one
organ — that **learns in superposition without gradients**, the way leOS already does. That is a
defensible, distinctive, Path-D direction. The discipline that makes it real is the engine's own: pick a
concrete GPU-thing, do it in superposition, measure exactly where the wall is, and write the cliff down.
Build the whole computer, own the niche where this compute model is genuinely superior, treat the GPU as
the coprocessor it already is — and let the constraints (bounded capacity, no backprop) be the generative
principle rather than the thing to apologize for.

---

### What this session leaned on
- **The substrate's license:** Kleyko, Davies, Frady, Kanerva et al. (2022), *Vector Symbolic
  Architectures as a Computing Framework for Emerging Hardware*, Proc. IEEE — VSA Turing-completeness
  (cited in leOS's own README).
- **HDC/VSA as a computing paradigm + hardware:** Kanerva (2009); Kleyko, Rachkovskij, Osipov, Rahimi
  (2022/2023), *A Survey on Hyperdimensional Computing aka VSA*, ACM Computing Surveys (Parts I & II);
  Karunaratne et al. (2020), *In-Memory Hyperdimensional Computing*, Nature Electronics; IBM Research
  in-memory computing; HyDra (SOT-MRAM CAM, 2025); HDC8192 (mixed-signal CMOS); photonic-HDC work.
- **The engine's own factorizer on emerging hardware:** Renner et al. (2024), *Neuromorphic Visual Scene
  Understanding with Resonator Networks*, Nature Machine Intelligence; Cotteret et al. (2025), robust
  symbolic computation in neuromorphic hardware; Frady, Kent, Olshausen, Sommer (2020), *Resonator
  Networks*.
- **VSA-native learned models (the real frontier of this direction):** Ganesan et al. (2021), *Learning
  with Holographic Reduced Representations* (NeurIPS) — differentiable HRR + the unit-magnitude fix, and
  why naive backprop through HRR underperforms; Alam et al. (2023) — HRR linear-cost self-attention
  (Hrrformer); Kymn et al. (2025), *Residue Hyperdimensional Computing*, Neural Computation; Meta AI
  (2024), *Coconut* — continuous-latent reasoning. (The last three are already cited in leOS.)
- **The audit of leOS:** LVM + 580-instruction kernel + SDOL language; `residue_arithmetic.py` (CRT
  integer math), `resonator_network.py` (~160k items @ 1024d), `superposed_compute.py` ("one processor,
  many states"), `program_algebra.py`, `path_integral.py`; the LLM held in the coprocessor bay; learning
  via displacement codec, online `lstsq` probes, success gradients, reflex arc, nutrient field — all
  gradient-free.
