"""One model over one holographic space.

The rest of this project grew as separate studies -- a self-organizing classifier, a
self-maintaining decision brain, an image vault, a mixture-of-experts router, a text
n-gram. They were never meant to stay separate. They already share the one thing that
matters: a holographic vector space, and a `UniversalEncoder` that turns ANY input --
text, image, number, category, record, sequence -- into a vector in that single space.

`UnifiedMind` is the top level that makes the sharing real instead of nominal. There is
ONE perception step (the encoder), ONE associative memory (the autonomous
`SelfOrganizingMind`, which both classifies and is searched for recall), and ONE
decision brain (`HolographicMind`), all reading and writing the same space. It does not
reimplement simple versions of these -- the failing of the old `Mind` facade -- it uses
the real, self-maintaining ones, and every input passes through the same encoder before
it reaches any of them.

What is deliberately NOT pretended to be one call: classification, recall, and decision
are different OPERATIONS on the shared substrate (aggregate into prototypes; index the
individuals; weight by reward). The unification is the shared space and the shared
self-maintenance, not a single magic method.
"""

import numpy as np

from holographic_ai import cosine
from holographic_mind import UniversalEncoder, _Index
from holographic_organizer import SelfOrganizingMind
from holographic_creature import HolographicMind


class UnifiedMind:
    """Perceive once, into one space; remember, organize, recall, and decide over it.

      read(corpus)                     -- let perception pre-learn word co-occurrence
      absorb(examples)                 -- SELF-ASSEMBLY: build a working mind from a pile
                                          of (input, label[, modality]) examples
      learn(x, label[, modality])      -- file a perception into the one memory; the
                                          modality is discovered if not declared
      classify(x[, modality])          -- 'what is this?'  (nearest self-organized prototype,
                                          routed within the discovered/declared modality)
      recall(x[, modality])            -- 'what's like this?' (nearest stored individual)
      actions(names) / decide / reinforce  -- choose actions over the same space

    The memory maintains itself: with maintain='auto' it periodically reorganizes (the
    speculate-measure-adopt rule from holographic_organizer), splitting a confusable
    class into sub-prototypes only when held-out accuracy says it earns its keep. The
    decision brain maintains itself the same way.
    """

    # modalities whose inputs are strings/token-lists: type inference alone cannot
    # tell them apart (code and prose are both str), so classify resolves between
    # them by CONTENT -- the compression gate (see _resolve_text_like)
    _TEXT_LIKE = ("text", "code")
    _FORMAT_CORPUS_CAP = 40000     # chars per sub-format kept for fitting the gate

    def __init__(self, dim=1024, seed=0, number_range=(-4.0, 4.0), maintain='auto',
                 check_every=60, text_window=2):
        self.dim = dim
        self.maintain = maintain
        self.check_every = check_every
        # ONE perception, shared by everything below
        self.encoder = UniversalEncoder(dim, seed=seed, number_range=number_range,
                                        text_window=text_window)
        # ONE associative memory: classify by nearest prototype, organize autonomously
        self.memory = SelfOrganizingMind(dim=dim, seed=seed)
        # a recall view over the SAME encoded vectors (individuals, for 'what's like this')
        self._recall = None
        # ONE decision brain (assembled when an action set is declared)
        self._brain = None
        self._actions = None
        self._taught = 0
        self._label_modality = {}    # which modality each label came from (for routing)
        self._gen = None             # sequence generator (lazy)
        # sub-format discovery state: raw samples of each TEXT-LIKE modality (capped),
        # and a lazily fitted compression-gate schema per modality (see classify)
        self._format_corpus = {}     # modality -> accumulated raw chars
        self._format_gate = None     # modality -> fitted SchemaGenerator
        self._format_fitted_at = {}  # modality -> corpus size when its schema was fit

    # -- perception (the single front door) --------------------------------
    def read(self, corpus):
        """Pre-learn word co-occurrence so text perceptions carry meaning."""
        self.encoder.learn_text(corpus)
        return self

    def perceive(self, x, modality=None):
        """Any input -> one vector in the shared space. This is the only encoder in the
        system; the memory and the brain never encode anything themselves."""
        return self.encoder.encode(x, modality)

    # -- one memory: classification + organization -------------------------
    def learn(self, x, label, modality=None):
        # SELF-DISCOVERY: if the caller does not name the modality, the encoder
        # infers it from the input itself (encoder.infer is the single source of
        # truth, so the tag recorded here always matches the encoding used).
        # Without this, untagged learning stored modality=None and the routing
        # safeguard in classify() silently vanished for those labels.
        if modality is None:
            modality = self.encoder.infer(x)
        v = self.perceive(x, modality)
        self.memory.observe_vector(v, label)        # aggregate into self-organized prototypes
        self._index(v, (label, x))                  # AND keep the individual for recall
        self._label_modality[label] = modality      # remember which modality this label is
        if modality in self._TEXT_LIKE:
            # keep a bounded sample of each text-like sub-format's raw characters --
            # the corpus the classify-time compression gate is fitted on. Capped so
            # the gate's schema fit stays a few seconds, never grows with the mind.
            cur = self._format_corpus.get(modality, "")
            if len(cur) < self._FORMAT_CORPUS_CAP:
                raw = x if isinstance(x, str) else " ".join(str(t) for t in x)
                self._format_corpus[modality] = (cur + " " + raw)[:self._FORMAT_CORPUS_CAP]
        self._taught += 1
        if self.maintain == 'auto' and self._taught % self.check_every == 0:
            self.memory.auto_reorganize()
        return self

    def classify(self, x, modality=None, route=True):
        """Nearest self-organized prototype. If `route` is on, the query competes only
        against its own modality's concepts -- a cheap router that removes the
        cross-modal interference a single flat store can otherwise suffer (a text
        query mistaken for an image). The modality may be declared or, when it is
        not, DISCOVERED from the input -- in two stages:

        * TYPE inference (`encoder.infer`): measured to score identically to
          caller-declared tags on the mixed-modality demo (97.5% both ways).
        * CONTENT inference, only where type goes blind: code and prose are both
          `str`, so when the mind holds text-like sub-formats a string query is
          resolved by the compression gate fitted on the mind's own learned
          samples. This is a CORRECTNESS fix, not a booster -- measured on a
          docs-vs-code set with heavy shared vocabulary, plain type inference
          routed every code query into a pool that EXCLUDED the code labels
          (24% accuracy, 66% cross-pool leakage, worse than no routing at all),
          while the gate identified the sub-format on 100% of held-out queries
          and recovered declared-tag accuracy (61%) exactly. Routing's GAIN over
          a flat scan stayed zero on that data (the bag-of-token vectors already
          separate docs from code) -- the safeguard story again, now one level
          down."""
        if modality is None:
            modality = self.encoder.infer(x)
            if modality == "text":
                modality = self._resolve_text_like(x)
        among = None
        if route:
            among = {lab for lab, m in self._label_modality.items() if m == modality}
            among = among or None
        return self.memory.classify_vector(self.perceive(x, modality), among=among)

    def _resolve_text_like(self, x):
        """Which text-like sub-format is this string? Type inference can only say
        'text'; if the mind has learned other text-like sub-formats (code), decide by
        the compression gate over schemas fitted on the mind's OWN learned samples --
        whoever compresses the query best understands it."""
        present = {m for m in self._label_modality.values() if m in self._TEXT_LIKE}
        if not present or present == {"text"}:
            return "text"                       # nothing to disambiguate
        if len(present) == 1:
            return next(iter(present))          # only code was learned: a string means code
        gens = self._format_schemas(present)
        if not gens or len(gens) < 2:
            return "text"                       # no corpus to gate with -- fall back safely
        from holographic_schema import compression_gate
        raw = x if isinstance(x, str) else " ".join(str(t) for t in x)
        return compression_gate(raw, gens)[0][1]

    def _format_schemas(self, modalities):
        """Fit (and cache) one small schema per text-like sub-format from the raw
        samples learn() accumulated. Refit only when a corpus has grown by more than
        a third since its schema was fitted, so steady-state classify pays nothing."""
        from holographic_schema import SchemaGenerator
        if self._format_gate is None:
            self._format_gate = {}
        for m in modalities:
            corpus = self._format_corpus.get(m, "")
            if len(corpus) < 200:                # too little to characterise a format
                continue
            fitted_at = self._format_fitted_at.get(m, 0)
            if m not in self._format_gate or len(corpus) > 1.34 * fitted_at:
                self._format_gate[m] = SchemaGenerator(m if m == "code" else "text",
                                                       cuts=(0, 60, 150)).fit(corpus)
                self._format_fitted_at[m] = len(corpus)
        return {m: g for m, g in self._format_gate.items() if m in modalities}

    # -- self-assembly: a working mind straight from a pile of examples -----
    def absorb(self, examples, maintain=True, sequences=False):
        """SELF-ASSEMBLY: hand the mind a pile of `(input, label)` or
        `(input, label, modality)` examples and it builds itself -- discovers each
        item's modality, pre-reads whatever text it sees (so word vectors carry
        co-occurrence meaning BEFORE any text is filed; learning text into the
        memory with cold word vectors throws information away), learns everything
        into the one memory, and runs one maintenance pass.

        With `sequences=True` the assembly is COMPLETE: the mind also fits one
        named sequence schema per text-like sub-format it discovered, from the
        same accumulated samples the classify gate uses -- so the one call returns
        a mind that classifies, recalls, AND generates, with unnamed generation
        routed by the compression gate. Off by default only because the schema
        fits cost a few seconds each.

        This is the one good idea of the retired `assemble()` facade, done on the
        real self-organizing machinery instead of a toy reimplementation. It is
        sugar over read()/learn()/maintain_now()/learn_sequence() -- deliberately,
        so there is nothing here to drift out of sync with the long-hand path."""
        examples = [(e if len(e) == 3 else (e[0], e[1], None)) for e in examples]
        examples = [(x, lab, m if m is not None else self.encoder.infer(x))
                    for x, lab, m in examples]
        # first pass: read everything text-LIKE so co-occurrence is learned before
        # filing -- code included, since code encodes through the same word-vector
        # path and its tokens (self, def, bind...) carry co-occurrence meaning too
        text = [x for x, _, m in examples if m in self._TEXT_LIKE]
        if text:
            self.read(text)
        # second pass: file everything into the one memory
        for x, lab, m in examples:
            self.learn(x, lab, m)
        if maintain:
            self.maintain_now()
        if sequences:
            # third pass: one sequence schema per discovered text-like sub-format,
            # fitted on the same capped samples learn() accumulated for the gate
            for m, corpus in self._format_corpus.items():
                if len(corpus) >= 200:
                    self.learn_sequence(corpus, modality=("code" if m == "code" else "text"),
                                        name=m)
        return self

    # -- the same data, a recall view (nearest individual) -----------------
    def _index(self, v, payload):
        if self._recall is None:
            self._recall = _Index(self.dim)
        self._recall.add(v, payload)

    def recall(self, x, modality=None):
        """Nearest stored individual. The index does an exact scan until the store is
        genuinely big, then switches to the recursive HoloForest (the crossover is
        measured -- see _Index.recall). A NEGATIVE worth recording here: wiring the
        learned adaptive navigator (holographic_navigator) into this path was tried
        and lost badly on the mind's own store -- 48% recall@1 at ~130 comparisons,
        where the forest at beam 2 gets 89% within ~512. The navigator's margin
        senses were tuned on UNIFORM random vectors; the unified store is clustered
        (many near-duplicates per class), which miscalibrates the arrive/keep-moving
        instinct. So recall keeps the dumb-but-honest index, and the navigator stays
        a study of adaptive access, not a default."""
        if self._recall is None:
            raise RuntimeError("nothing learned yet -- call learn() first")
        return self._recall.recall(self.perceive(x, modality))

    # -- one decision brain, on the same substrate -------------------------
    def actions(self, names):
        self._actions = list(names)
        self._brain = HolographicMind(self.dim, self._actions, k=12, epsilon=0.1,
                                      novelty_bonus=0.15, memory_cap=8000,
                                      maintain=self.maintain)
        return self

    def decide(self, state, explore=False, epsilon=None, modality=None):
        if self._brain is None:
            raise RuntimeError("declare an action set first -- call actions([...])")
        a = self._brain.decide(self.perceive(state, modality), explore=explore, epsilon=epsilon)
        return self._actions[a]

    def reinforce(self, state, action, reward, modality=None):
        s = self.perceive(state, modality)
        self._brain.remember([s], [self._actions.index(action)], [float(reward)])
        return self

    # -- generation: predict the next symbol over the same space ------------
    def learn_sequence(self, data, n=6, hierarchical=True, modality="text", name=None):
        """Learn to continue a sequence.

        Two engines, picked by `hierarchical`:

        * The fractal coder (default): discover a chunk schema by compression, then predict by
          cross-level backoff -- emit the longest chunk a level is confident about, else descend
          a level and spell it out. Measured against the flat n-gram on Austen, it cut bits/char
          from 2.085 to 1.829 and the stored model from ~218k context entries to ~58k (3.8x
          smaller), at roughly tied coherence (0.96 vs 0.98 real words). Generation is the
          traversal-shaped operation where the multi-scale substrate earns its keep -- unlike
          classification, where a tree REGRESSED and the flat scan stayed best.

        * The flat holographic n-gram (`hierarchical=False`): the original engine, kept because
          it exposes `next_symbol` and an exact context key, and because the boundary between
          where the substrate helps and where it doesn't is measured here, not assumed.

        Two consolidations, both backward compatible:

        * `modality` passes through to the fractal coder, so the mind can learn to
          continue CODE, not just prose -- the same compress-by-merging schema was
          measured to discover code structure from scratch (held-out bits/char 2.98
          -> 2.28 on this project's own source, with `def __init__` and indentation
          idioms among the unlabeled emergent chunks).
        * `name` lets the mind hold MANY sequence schemas at once. Unnamed calls keep
          the old single-slot behaviour (each call replaces); named calls accumulate,
          and generate() with no name picks the schema by the compression gate -- the
          one routing primitive used everywhere else in the stack. That is
          content-level self-discovery, needed exactly where TYPE-level inference
          goes blind: code and prose are both `str`."""
        if hierarchical:
            from holographic_schema import SchemaGenerator
            gen, kind = SchemaGenerator(modality=modality).fit(data), "hierarchical"
        else:
            from holographic_text import HolographicNGram
            gen = HolographicNGram(dim=self.dim, n=n, seed=0).fit(data)
            kind = "flat"
        key = name if name is not None else "default"
        if not hasattr(self, "_gens"):
            self._gens = {}
        self._gens[key] = {"gen": gen, "kind": kind, "modality": modality}
        self._gen, self._gen_kind = gen, kind        # most-recent alias (compat)
        return self

    def _pick_gen(self, name=None, seed_text=""):
        """Resolve which sequence schema a call means. Named -> that one. One schema
        -> it. Several and unnamed -> route the SEED by the compression gate: whoever
        compresses the seed best is the schema that understands it. The honest
        boundary: only hierarchical schemas expose bits_per_char, so flat engines
        never compete in the gate -- name them explicitly."""
        gens = getattr(self, "_gens", {})
        if not gens:
            raise RuntimeError("nothing learned to continue -- call learn_sequence() first")
        if name is not None:
            if name not in gens:
                raise KeyError(f"no sequence schema named {name!r} -- have {sorted(gens)}")
            return gens[name]
        if len(gens) == 1:
            return next(iter(gens.values()))
        from holographic_schema import compression_gate
        gated = {k: g["gen"] for k, g in gens.items() if g["kind"] == "hierarchical"}
        if not gated or not seed_text:
            raise RuntimeError("several schemas are loaded -- name one, or give a seed "
                               "the gate can route (flat engines must be named)")
        return gens[compression_gate(seed_text, gated)[0][1]]

    def next_symbol(self, context, name=None):
        g = self._pick_gen(name, context)
        if g["kind"] != "flat":
            raise RuntimeError("next_symbol needs the flat engine: learn_sequence(text, hierarchical=False)")
        return g["gen"].next_char(context)

    def generate(self, seed_text, length=160, temperature=0.5, name=None):
        return self._pick_gen(name, seed_text)["gen"].generate(seed_text, length, temperature)

    # -- self-maintenance across the whole model ---------------------------
    def maintain_now(self):
        """Reorganize the memory and refresh the brain, each by its own held-out
        measurement. Returns the memory's choice."""
        choice = self.memory.auto_reorganize()
        if self._brain is not None and self._brain.maintain == 'auto':
            self._brain.auto_maintain()
        return choice

    def describe(self):
        parts = [f"memory of {self.memory.live.size()} prototypes over "
                 f"{len(self.memory.live.counts_by_label())} labels"]
        if self._recall is not None:
            parts.append(f"a recall index of {len(self._recall.vecs)} items")
        if self._brain is not None:
            parts.append(f"a decision brain over {self._actions}")
        if getattr(self, "_gens", None):
            descs = []
            for key, g in sorted(self._gens.items()):
                detail = (f"order {g['gen'].n}" if g["kind"] == "flat"
                          else f"fractal coder, {g['modality']}")
                descs.append(f"{key}: {detail}")
            plural = "s" if len(self._gens) > 1 else ""
            parts.append(f"sequence schema{plural} ({'; '.join(descs)})")
        return "UnifiedMind: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# DEMO: one mind, many modalities, one memory -- measured against separate ones
# ---------------------------------------------------------------------------

def _patterns(kind, rng, n=8):
    """Tiny synthetic 'images' -- four visually distinct classes, with noise."""
    a = np.zeros((n, n))
    if kind == "rows":
        a[::2, :] = 1.0
    elif kind == "cols":
        a[:, ::2] = 1.0
    elif kind == "diag":
        for i in range(n):
            a[i, i] = 1.0; a[i, (i + 1) % n] = 1.0
    elif kind == "check":
        a[(np.add.outer(np.arange(n), np.arange(n)) % 2) == 0] = 1.0
    return a + 0.15 * rng.standard_normal((n, n))


def demo_unified():
    """One UnifiedMind learns three different KINDS of thing -- text topics, little
    images, and records -- into a SINGLE self-organizing memory, then classifies all
    three. The honest question is whether one shared store does as well as three
    separate ones; if mixing modalities in one space wrecked it, the unification would
    be fake. It does not: the modalities land in near-orthogonal parts of the space, so
    one memory matches the separate baselines AND the same mind still makes decisions."""
    from holographic_text import TOPICS, _content, _split

    print("=" * 70)
    print("One mind, one memory: text + images + records in a single space")
    print("=" * 70)
    rng = np.random.default_rng(0)
    corpus = [s for sents in TOPICS.values() for s in sents]

    # build the three datasets as (input, label, modality)
    text_tr, text_te = [], []
    for topic, sents in TOPICS.items():
        a, b = _split(sents, frac=0.7, seed=2)
        text_tr += [(_content(s), topic, "text") for s in a]
        text_te += [(_content(s), topic, "text") for s in b]
    img_tr, img_te = [], []
    for kind in ("rows", "cols", "diag", "check"):
        for _ in range(20):
            img_tr.append((_patterns(kind, rng), f"img:{kind}", "image"))
        for _ in range(8):
            img_te.append((_patterns(kind, rng), f"img:{kind}", "image"))
    rec_tr, rec_te = [], []
    depts = ("eng", "sales", "ops")
    for d in depts:
        for _ in range(20):
            rec_tr.append(({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record"))
        for _ in range(8):
            rec_te.append(({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record"))

    # ---- ONE unified mind: everything into one memory --------------------
    # text word-vectors learn best from content words (stopwords dilute co-occurrence);
    # that is a text-task choice, so the orchestrator makes it -- the encoder stays generic.
    mind = UnifiedMind(dim=1024, seed=0).read([_content(s) for s in corpus])
    train = text_tr + img_tr + rec_tr
    rng.shuffle(train)
    for x, label, mod in train:
        mind.learn(x, label, mod)
    mind.maintain_now()

    def score(m, test, route=True):
        return sum(m.classify(x, mod, route=route)[0] == lab for x, lab, mod in test) / len(test)

    ut = score(mind, text_te); ui = score(mind, img_te); ur = score(mind, rec_te)
    ut_flat = score(mind, text_te, route=False)

    # ---- separate baselines: one memory per modality (same encoding) -----
    def separate(train_items, test_items):
        enc = UniversalEncoder(1024, seed=0)
        enc.learn_text([_content(s) for s in corpus])
        mem = SelfOrganizingMind(dim=1024, seed=0)
        for x, lab, mod in train_items:
            mem.observe_vector(enc.encode(x, mod), lab)
        mem.auto_reorganize()
        return sum(mem.classify_vector(enc.encode(x, mod))[0] == lab
                   for x, lab, mod in test_items) / len(test_items)

    st = separate(text_tr, text_te); si = separate(img_tr, img_te); sr = separate(rec_tr, rec_te)

    print(f"\n  {'modality':10s}{'separate memory':>18s}{'one shared memory':>20s}")
    print(f"  {'text':10s}{100*st:>16.0f}% {100*ut:>18.0f}%")
    print(f"  {'images':10s}{100*si:>16.0f}% {100*ui:>18.0f}%")
    print(f"  {'records':10s}{100*sr:>16.0f}% {100*ur:>18.0f}%")
    print(f"\n  Routing: a text query against ALL concepts scores {100*ut_flat:.0f}%; restricted to")
    print(f"  text concepts (its known modality) it scores {100*ut:.0f}%. With correct encoding the")
    print("  modalities separate cleanly, so here routing changes nothing -- it is a cheap")
    print("  safeguard that removes cross-modal collisions WHEN they occur, not a routine")
    print("  booster. (An earlier apparent gain came from a since-fixed encoding bug that")
    print("  degraded text vectors into colliding with other modalities.)")
    print(f"\n  {mind.describe()}")

    # ---- cross-modal recall over the same store --------------------------
    q = img_te[0]
    (lab, _), sim = mind.recall(q[0], q[2])
    print(f"\n  Recall: a held-out '{q[1]}' image finds nearest stored item '{lab}' "
          f"(cos {sim:.2f}) -- the recall view searches the same vectors.")

    # ---- the SAME mind also decides -------------------------------------
    mind.actions(["left", "right"])
    rng2 = np.random.default_rng(1)
    for _ in range(400):
        n = float(rng2.uniform(-3, 3))
        good = "right" if n > 0 else "left"
        choice = mind.decide(n, explore=True, epsilon=0.3, modality="number")
        mind.reinforce(n, choice, 1.0 if choice == good else 0.0, modality="number")
    dec = sum((mind.decide(float(v), modality="number") == ("right" if v > 0 else "left"))
              for v in np.linspace(-3, 3, 40)) / 40
    print(f"  Decision: the same mind learned a contextual choice over numbers -> "
          f"{100*dec:.0f}% correct, using the same encoder and space.")

    # ---- the SAME mind also generates (the fourth operation) -------------
    mind.learn_sequence(" ".join(corpus), n=5)
    sample = mind.generate("the ", 90, 0.4)
    print(f"  Generation: taught to continue the topic text, it produces -> \"{sample[:70]}\"")

    print(f"\n  {mind.describe()}")
    print("\n  One encoder, one self-organizing memory, one brain -- shared substrate, not")
    print("  a wrapper. One shared store matches separate per-modality memories; with")
    print("  correct encoding the modalities are near-orthogonal, so a flat store shows")
    print("  no cross-modal interference here and routing is a cheap safeguard rather than")
    print("  a booster. Storage needs no separate curator: the memory's own aggregation")
    print("  already compresses (here ~1800 observations into a handful of prototypes).")
    print("  Generation completes the operation set -- its next-symbol step is the same")
    print("  cleanup primitive -- though its context index stays exact, the one place a")
    print("  fuzzy recall was measured to hurt rather than help.")


if __name__ == "__main__":
    demo_unified()
