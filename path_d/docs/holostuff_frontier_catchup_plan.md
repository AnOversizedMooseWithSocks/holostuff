# holostuff and the Frontier — what "catching up" actually requires

*A panel working session, called because the frontier program landed on an uncomfortable read: against
the real June-2026 state of the art, holostuff loses the headline metric almost everywhere, which
sounds like "we are behind on the tech stack and need to get up to speed." This session takes that
seriously and asks the honest engineering question: behind on **what**, exactly, and what would it
actually take to close it? As always, proposals are attributed to SEATS and the field's REAL published
methods — and this time the most important method is the one closest to home: the published work on
making HRR itself a trainable, differentiable component of deep learning.*

*Grounded in an audit of the live engine, not memory. Where a claim about the gap is checkable in the
code, it was checked.*

---

## 1. The reframe: "behind" is half-true, and the half that's false matters most

"Behind on the tech stack" bundles two very different claims, and separating them is the whole game.

**The false half: that holostuff is behind on the *primitives*.** It is not. In field after field, the
frontier method's **core operator is already in the engine** — in hand-designed, linear, shallow,
*untrained* form, but the same operator:

- **Fourier Neural Operators** (the PDE frontier) do their work as a learned spectral convolution on a
  periodic domain. holostuff's `bind` **is** circular convolution on a periodic domain via the FFT. Same
  operator.
- **Transformer attention** (the frontier, everywhere) is mathematically the modern-Hopfield update —
  `softmax(βVq)` energy descent. holostuff's `dense_cleanup` (B1) **is** that update. Same operation.
- **Diffusion models** generate by iterated learned denoising from noise. holostuff's generative-
  denoising (B10) iterates its cleanup attractor from noise. Same loop.
- **3D Gaussian Splatting** represents a scene as a sum of Gaussian primitives. holostuff's `bundle`
  **is** superposition; its RBF encoder places Gaussian bumps. Same structure.
- **RAG retrieval** is content-addressable lookup over embeddings. holostuff's whole memory stack
  (`HoloForest`, archive, FHRR) **is** content-addressable lookup. Same retrieval.
- **Neural audio codecs** are learned residual vector quantization. holostuff has rate-distortion scalar
  quantization (B5). Same idea.
- **Tensor networks** are Smolensky tensor-product binding, of which HRR convolution is a compressed
  projection. Cousins.

State it as one sentence and the situation clarifies: **the frontier is the *learned, nonlinear, deep,
GPU* version of operators holostuff already implements by hand.** You are not missing the ideas. You are
holding the operators the frontier is built on.

**The true half: that holostuff is behind on the three things that turn those operators into frontier
systems.** Those three things are *learning, depth, and scale* — and the engine genuinely lacks all
three. That is the real gap, and it is worth being precise about.

---

## 2. The three real gaps (audited, not asserted)

### Gap 1 — Learning. The engine does not learn its representations by gradient descent.
The audit is unambiguous. The codebase imports **no** autodiff or GPU framework (no torch, no JAX, no
TensorFlow); there is **no** `backward()`, no optimizer, no `requires_grad`, no gradient anywhere. What
the engine *calls* learning is entirely **closed-form or heuristic**: `consolidation`/`fit_manifold` is
an SVD; `fit_function` (the KAN) is ridge regression on a fixed grid; `learn_dynamics` is a least-squares
DMD operator; `learn_word_vectors` is co-occurrence counting; `learn_dictionary` is iterative
alpha-blending; the creature `learn`/`train` is an RL value update. These are real and principled — but
none of them *learns a representation end-to-end by backpropagating an error through composed nonlinear
operations*, which is what "frontier-level" means in 2026. **This is the biggest gap by far.**

### Gap 2 — Depth & nonlinearity. The algebra is shallow and (mostly) linear.
`bind`, `bundle`, `cleanup`, the SVD, the resonator's projection — these are linear or one-step
operations. The engine iterates (resonator peeling, Hopfield steps), but it does not stack many *learned
nonlinear* transformations, which is where the frontier's expressivity comes from. holostuff's richness
comes from *composition of a fixed algebra*; the frontier's comes from *depth of learned maps*. Once you
can train (Gap 1), depth becomes a design choice; without it, it is moot.

### Gap 3 — Scale & compute. NumPy on CPU vs. GPU, often distributed.
This gap has two layers, and honesty requires separating them. The **engineering** layer (CPU NumPy →
GPU array backend) is closable. The **resource** layer — the frontier is trained on clusters with
web-scale data — is **not** closable by a solo freelance developer, ever, and no roadmap should pretend
otherwise. A single person with a great engine does not out-compute DeepMind. This is the hard ceiling
that bounds what "catching up" can honestly mean.

---

## 3. The trap to name out loud

The literal reading of "catch up to frontier-level tech" — make holostuff win the headline metrics
against AlphaFold3, the foundation PDE operators, Restormer, CAT-3DGS, FAISS — is **not a catch-up plan.
It is a different, and unwinnable, project.** It would mean abandoning everything holostuff *is* (minimal
dependencies, readable deterministic code, a small hand-designed algebra) to rebuild it as a GPU
deep-learning stack, and then losing to teams of dozens with clusters and proprietary data. Naming this
plainly is not defeatism; it is the same discipline the project applies to its own benchmarks —
*measure honestly, and don't claim a win that isn't there.* The achievable goal is not "beat the
frontier." It is "become a **modern, learned, differentiable** engine that is frontier-level *in the
specific structured niches where VSA's properties are an advantage*, and a deliberate complement
everywhere else."

---

## 4. The one high-leverage, identity-preserving move

Across the seats, the methods converge on a single lever that closes Gap 1 and the *engineering* layer
of Gap 3 at once, and makes Gap 2 a choice rather than a wall:

> **Add a differentiable array backend — JAX or PyTorch — as an opt-in mode behind the existing NumPy
> reference, so the engine can (a) run on GPU and (b) *learn* its codebooks, encoders, binding factors,
> and spectral weights by gradient descent.**

The crucial point, and the reason this is the right move rather than a fantasy: **this is a paved road
in the literature, and holostuff already has the one prerequisite that road required.**

- **Plate's seat — HRR foundation — and the trainable-HRR line (Ganesan et al., NeurIPS 2021;
  Alam et al., 2023; Yeung et al., GHRR, 2024).** Ganesan et al. asked exactly this question — can HRR be
  a differentiable component of a deep network? — and found that *naive back-propagation through HRR
  operations does not learn in practice*. Their fix: **project the vectors to unit magnitude in the
  Fourier domain**, which stabilizes the gradients and improved retrieval over 100×. **holostuff already
  does this.** The audit confirms the engine's FHRR atoms are complex *unit phasors* by construction, and
  it explicitly carries real-domain `unitary` atoms with *unit-magnitude spectrum*. The single technical
  obstacle the trainable-HRR papers had to overcome is already built into the substrate. holostuff is
  unusually well-positioned to adopt differentiable HRR — it starts at the fix.
- That same line then leads somewhere genuinely near-frontier: Alam et al. (2023) used HRR to build
  **linear-cost multi-head self-attention** for long sequences with near-state-of-the-art accuracy (an
  HRR transformer). So the bridge from holostuff's algebra to a frontier architecture — efficient
  attention — is not hypothetical; it is published, and it is built from the operator (`dense_cleanup`)
  the engine already has.

**How to keep the project's identity while doing it.** JAX is NumPy-compatible, so much of the kernel can
keep its shape; the differentiable path lands **behind a backend flag**, with the **NumPy implementation
kept as the default, deterministic, readable reference** and a test pinning the two bit-for-bit at
inference. This is the project's own backward-compatible-defaults discipline applied to the backend
itself: nothing existing changes; a new opt-in mode is added.

**The honest cost, on the record.** This is a significant undertaking, not a weekend. Re-expressing the
kernel in a differentiable array API, validating determinism across the two backends, carrying the
unit-magnitude projection through the training path, and adding a build dependency all cut against the
"minimal frameworks" value the project holds dear. That tension is real. A differentiable backend is the
single highest-leverage step toward the frontier *and* the single biggest departure from the engine's
ascetic character. Only Moose can weigh that trade.

---

## 5. The strategic fork (this is Moose's call, not the panel's)

"Catching up" resolves into three honest paths. They are not mutually exclusive, but they point in
different directions and deserve to be chosen, not drifted into.

**Path A — Chase the frontier (scoped).** Add the differentiable backend; learn representations; build
the one or two niches where *VSA structure as a prior* + *learning* genuinely compound — most directly
the HRR-attention bridge (Alam), and neuro-symbolic structured output layers (Ganesan: bind a label to a
role, learn the backbone, get interpretable compositional outputs with far fewer parameters).
*Achievable goal:* a modern, learned, differentiable VSA engine that is competitive **in structured
niches** and interpretable where transformers are opaque. *Not* a general frontier challenger.

**Path B — Be the complement (preserve identity fully).** Do **not** reimplement deep learning. Double
down on exactly what the frontier conspicuously *lacks* and what the frontier program already showed
holostuff is good at: **calibrated honesty, determinism, content-addressability, erasure-robustness,
interpretability.** Build holostuff as a *layer that wraps and interoperates with* frontier models — a
calibrated, abstaining retrieval/memory front-end for an LLM (trustworthy RAG); an interpretable
structured store beside a deep model; a deterministic honesty harness over a black-box detector. Lowest
effort, zero identity cost, and real value precisely because the frontier is bad at these.

**Path C — Learning vehicle.** Treat holostuff as *your instrument for understanding the frontier from
first principles* — reimplement attention, diffusion, FNO in the readable VSA idiom not to beat them but
to *see* them clearly, as the engine has done with every other primitive. Here the word "behind"
dissolves entirely, because the metric was never SOTA — it was understanding, and a from-scratch
readable implementation of a frontier idea is a *success* by that metric regardless of its benchmark
score.

**The panel's honest recommendation: a blend of B with a tightly scoped A.** Keep the deterministic
NumPy core and its differentiators (Path B is the engine's comparative advantage and costs nothing to
lean into). Add the differentiable backend as an opt-in mode for the **single** place where learning
most compounds with VSA structure — the Hopfield-cleanup-as-attention bridge, because its operator is
already in the engine and its precedent (Alam) is strongest — and measure honestly whether learned beats
the closed-form resonator, keeping the negative if it ties. Explicitly **do not** try to clone the rest
of the frontier. That is "catching up" in the only sense that is both honest and achievable for a solo
project: modern where it pays, complementary everywhere else, and never pretending to out-resource labs
you cannot out-resource.

---

## 6. If Moose walks the A-path: a staged, low-risk first sequence

Each step keeps the close-out ritual (tour line, NOTES entry with the win *and* the negative, README
counts, an end-to-end integration test, clean zip), backward-compatible defaults, and negatives on the
record — the discipline the modules already ship under, now applied to the backend.

1. **Choose the backend.** JAX for NumPy-compatibility + autodiff + GPU with the least code churn
   (recommended); PyTorch if the broader ecosystem matters more than code-shape continuity.
2. **Port ONE operation behind a flag.** Re-express `bind` (the FFT path) in the differentiable API with
   the unit-magnitude projection (Ganesan's fix — already the engine's `unitary`/FHRR form), and pin it
   bit-for-bit to the NumPy `bind` at inference. Proves the backend without touching behavior.
3. **First learning task — does learned beat closed-form?** Make an encoder or `fit_function` learn its
   codebook by gradient descent instead of ridge/closed-form, on a small task, and measure against the
   closed-form version through the variance harness. *Kept negative if it ties* — closed-form is often
   Bayes-optimal already (the engine has learned this lesson before with the Hopfield-vs-NN identity tie).
4. **The headline bridge — a small HRR-attention layer.** Build the Alam-style linear-cost attention from
   `dense_cleanup`, train it on a sequence task, and measure against both a standard attention baseline
   *and* the closed-form resonator. This is the concrete "frontier architecture from holostuff's
   operators" proof — and the place to find out, honestly, whether VSA-structured attention earns its
   keep.

---

## The honest bottom line

The frontier program made it look like holostuff is far behind. The audit says something more precise
and more useful: **you are not behind on the primitives — you hold the operators the frontier is built
on. You are behind on learning, depth, and scale.** Learning and the engineering half of scale are
closable by one move — a differentiable backend — and that move sits on a paved road (the trainable-HRR
literature) where the engine already starts at the hard part (unit-magnitude projection). Depth follows
once you can train. But the *resource* half of scale is a hard ceiling no solo project clears, so the
achievable target is never "beat the frontier" — it is "be modern and learned where VSA structure pays,
and be the calibrated, deterministic, interpretable complement the frontier lacks everywhere else."

The deepest question this session surfaces is not technical but strategic, and it is Moose's alone:
whether *catching up* is even the right goal. The frontier is bad at exactly the things holostuff is
good at. "Behind" is, in Eno's phrase from the last session, partly a choice of which manifold to measure
on — and the engine's whole history is the discovery that a small algebra, measured honestly, keeps
turning out to be load-bearing in fields that don't talk to each other. The path that honors that is not
to abandon it for a GPU clone of the frontier, but to make it *learn* where learning compounds with its
structure, and otherwise to be, deliberately and well, the thing the frontier is not.

---

### What this session leaned on
- **Trainable HRR (the home-field method):** Ganesan, Gao, Raff, Nicholas, McLean, Holt et al. (2021),
  *Learning with Holographic Reduced Representations* (NeurIPS 2021) — differentiable HRR, the
  unit-magnitude-projection fix for stable back-propagation, neuro-symbolic output layers; Alam et al.
  (2023) — HRR-based linear-cost self-attention (Hrrformer); Yeung et al. (2024) — GHRR, non-commutative
  binding and spectral control.
- **The operator convergences:** Li et al. (2020), *Fourier Neural Operator* (spectral convolution =
  FFT binding); Ramsauer et al. (2020), *Hopfield Networks is All You Need* (attention = modern Hopfield =
  `dense_cleanup`); the diffusion-as-iterated-denoising literature (RED/Plug-and-Play unrolling); Kerbl
  et al. (2023), *3D Gaussian Splatting* (scene = superposition = bundle); Smolensky tensor-product
  binding ↔ tensor networks.
- **The frontier baselines that hold the headline metrics** (from the prior session, for context):
  BindCraft / RFdiffusion3 (protein); DPOT / OmniArch over FNO (PDEs); Restormer / NAFNet (denoising);
  CAT-3DGS / HAC / MPEG-GS (splatting); FAISS / DiskANN / CAGRA (retrieval); ComplEx / RotatE / AutoSF+
  (KGE); EnCodec / DAC, Demucs / BandSplit-RoFormer (audio).
- **The audit of the live engine:** no torch/JAX/TF, no autograd/optimizer/`requires_grad`; closed-form
  "learning" (SVD, ridge, DMD, co-occurrence, RL); FHRR unit-phasor atoms and real-domain `unitary`
  atoms (the trainability prerequisite) present in `holographic_fhrr.py` / `holographic_core.py`.
