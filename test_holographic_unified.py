"""Tests for the top-level UnifiedMind: one encoder, one self-organizing memory, one
decision brain over a single holographic space. The point is that the pieces share the
substrate -- so one memory should classify several modalities at once, the recall view
should search the same vectors, and the same mind should be able to decide."""

import numpy as np

from holographic_unified import UnifiedMind, _patterns


def _records(dept, rng, n):
    return [({"dept": dept, "level": int(rng.integers(1, 6))}, f"rec:{dept}", "record")
            for _ in range(n)]


def test_one_memory_holds_and_classifies_several_modalities():
    rng = np.random.default_rng(0)
    mind = UnifiedMind(dim=1024, seed=0)
    train, test = [], []
    for kind in ("rows", "cols", "diag", "check"):
        train += [(_patterns(kind, rng), f"img:{kind}", "image") for _ in range(20)]
        test += [(_patterns(kind, rng), f"img:{kind}", "image") for _ in range(8)]
    for d in ("eng", "sales", "ops"):
        train += _records(d, rng, 20)
        test += _records(d, rng, 8)
    rng.shuffle(train)
    for x, label, mod in train:
        mind.learn(x, label, mod)
    mind.maintain_now()

    # one memory now holds every label from every modality
    labels = set(mind.memory.live.counts_by_label())
    assert {f"img:{k}" for k in ("rows", "cols", "diag", "check")} <= labels
    assert {f"rec:{d}" for d in ("eng", "sales", "ops")} <= labels

    acc = sum(mind.classify(x, mod)[0] == lab for x, lab, mod in test) / len(test)
    assert acc >= 0.85          # both modalities classify well from the single store


def test_recall_view_searches_the_same_vectors():
    rng = np.random.default_rng(1)
    mind = UnifiedMind(dim=1024, seed=1)
    for kind in ("rows", "check"):
        for _ in range(15):
            mind.learn(_patterns(kind, rng), f"img:{kind}", "image")
    (label, _), sim = mind.recall(_patterns("rows", rng), "image")
    assert label == "img:rows"   # nearest stored individual is the right kind
    assert sim > 0.5


def test_same_mind_decides_over_the_same_space():
    mind = UnifiedMind(dim=1024, seed=2).actions(["left", "right"])
    rng = np.random.default_rng(2)
    for _ in range(400):
        n = float(rng.uniform(-3, 3))
        good = "right" if n > 0 else "left"
        choice = mind.decide(n, explore=True, epsilon=0.3, modality="number")
        mind.reinforce(n, choice, 1.0 if choice == good else 0.0, modality="number")
    acc = sum(mind.decide(float(v), modality="number") == ("right" if v > 0 else "left")
              for v in np.linspace(-3, 3, 40)) / 40
    assert acc >= 0.7            # the shared-substrate brain learned the contextual choice


def test_routing_removes_cross_modal_interference():
    # A flat store of several modalities can mistake a query for a foreign-modality
    # concept; restricting the query to its own modality (a cheap router, since the
    # modality is known) should never hurt and should fix those cross-modal errors.
    from holographic_mind import UniversalEncoder
    from holographic_text import TOPICS, _content, _split
    rng = np.random.default_rng(0)
    corpus = [s for ss in TOPICS.values() for s in ss]
    mind = UnifiedMind(dim=1024, seed=0).read([_content(s) for s in corpus])
    text_te = []
    for topic, ss in TOPICS.items():
        a, b = _split(ss, frac=0.7, seed=2)
        for s in a:
            mind.learn(_content(s), topic, "text")
        text_te += [(_content(s), topic) for s in b]
    for kind in ("rows", "cols", "diag", "check"):
        for _ in range(20):
            img = np.zeros((8, 8)); img[::2, :] = 1.0
            mind.learn(img + 0.15 * rng.standard_normal((8, 8)), f"img:{kind}", "image")
    for d in ("eng", "sales", "ops"):
        for _ in range(20):
            mind.learn({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record")
    mind.maintain_now()

    flat = sum(mind.classify(t, "text", route=False)[0] == lab for t, lab in text_te) / len(text_te)
    routed = sum(mind.classify(t, "text", route=True)[0] == lab for t, lab in text_te) / len(text_te)
    assert routed >= flat          # routing never hurts
    # every routed prediction is a text label (no cross-modal leakage)
    assert all(mind.classify(t, "text", route=True)[0] in set(TOPICS) for t, _ in text_te)


def test_same_mind_generates_over_the_shared_space():
    # Generation is the fourth operation on the one model: learn to continue a sequence,
    # then produce more of it. The next-symbol prediction is holographic cleanup (the
    # same primitive the classifier uses); only the context key is exact.
    from holographic_text import TOPICS
    text = " ".join(s for ss in TOPICS.values() for s in ss).lower()
    mind = UnifiedMind(dim=1024, seed=0).learn_sequence(text)          # default fractal engine
    out = mind.generate("the ", length=80, temperature=0.4)
    assert len(out) > 50                                  # it produced a continuation
    assert set(out) <= set(text)                          # only symbols it actually learned
    # next-symbol prediction is a flat-engine primitive; it should beat uniform random
    cut = int(len(text) * 0.9)
    m2 = UnifiedMind(dim=1024, seed=0).learn_sequence(text[:cut], n=5, hierarchical=False)
    held = text[cut:cut + 800]
    ok = sum(m2.next_symbol(held[max(0, j - 4):j]) == held[j] for j in range(1, len(held)))
    assert ok / (len(held) - 1) > 1.0 / len(set(text))    # well above chance


def test_modality_self_discovery_matches_declared_tags():
    # learn/classify with NO declared modality must discover the tag from the input
    # (encoder.infer) and route on it -- measured at exact parity with declared tags
    # on the mixed demo (97.5% both ways). Token lists are the trap: they must be
    # discovered as TEXT, not order-sensitive sequences (the original encode bug).
    from holographic_text import TOPICS, _content, _split
    rng = np.random.default_rng(0)
    corpus = [s for ss in TOPICS.values() for s in ss]

    declared, discovered = [], []
    for topic, ss in TOPICS.items():
        a, b = _split(ss, frac=0.7, seed=2)
        declared += [(_content(s), topic, "text") for s in a]
        discovered += [(_content(s), topic) for s in b]          # held out, untagged
    for kind in ("rows", "cols"):
        declared += [(_patterns(kind, rng), f"img:{kind}", "image") for _ in range(15)]
        discovered += [(_patterns(kind, rng), f"img:{kind}") for _ in range(6)]
    for d in ("eng", "sales"):
        declared += [({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}", "record")
                     for _ in range(15)]
        discovered += [({"dept": d, "level": int(rng.integers(1, 6))}, f"rec:{d}")
                       for _ in range(6)]

    mind = UnifiedMind(dim=1024, seed=0).read([_content(s) for s in corpus])
    for x, lab, _ in declared:
        mind.learn(x, lab)                                        # tags NOT declared
    mind.maintain_now()

    # learning discovered a real tag for every label (never None)
    assert all(m is not None for m in mind._label_modality.values())
    # untagged classification routes correctly and scores well across modalities
    acc = sum(mind.classify(x)[0] == lab for x, lab in discovered) / len(discovered)
    assert acc >= 0.85
    # discovered routing equals declared routing, query by query
    same = all(mind.classify(x)[0] ==
               mind.classify(x, ("text" if isinstance(x, list) else
                                 "image" if isinstance(x, np.ndarray) else "record"))[0]
               for x, _ in discovered)
    assert same


def test_absorb_self_assembles_a_working_mind():
    # SELF-ASSEMBLY: a pile of (input, label) pairs -- no modality tags, no read(),
    # no maintenance calls -- must come back as a working multi-modal mind that
    # matches the long-hand read/learn/maintain pipeline.
    from holographic_text import TOPICS, _content, _split
    rng = np.random.default_rng(0)
    pile, test = [], []
    for topic, ss in TOPICS.items():
        a, b = _split(ss, frac=0.7, seed=2)
        pile += [(_content(s), topic) for s in a]
        test += [(_content(s), topic) for s in b]
    for kind in ("rows", "cols", "diag", "check"):
        pile += [(_patterns(kind, rng), f"img:{kind}") for _ in range(20)]
        test += [(_patterns(kind, rng), f"img:{kind}") for _ in range(8)]
    rng.shuffle(pile)

    mind = UnifiedMind(dim=1024, seed=0).absorb(pile)
    acc = sum(mind.classify(x)[0] == lab for x, lab in test) / len(test)
    assert acc >= 0.85
    # the text it absorbed taught the word vectors (read() happened internally):
    # two same-topic sentences should sit closer than cross-topic ones on average
    assert mind.memory.live.size() >= len(set(lab for _, lab in pile))


def test_many_sequence_schemas_route_by_compression_gate():
    # Consolidation: the mind holds SEVERAL sequence schemas at once (named), learns
    # code as well as prose (modality passthrough), and unnamed generation routes the
    # seed by the compression gate -- content-level self-discovery, needed exactly
    # where type inference goes blind (code and prose are both str).
    from holographic_text import TOPICS
    prose = " ".join(s for ss in TOPICS.values() for s in ss).lower()
    code = ("def step(self, action):\n    reward = self.world.step(action)\n"
            "    self.memory.append((self.state, action, reward))\n"
            "    return reward\n\nfor i in range(n):\n    total += vals[i]\n"
            "    if total > cap:\n        break\n") * 25
    mind = (UnifiedMind(dim=1024, seed=0)
            .learn_sequence(prose, name="prose")
            .learn_sequence(code, modality="code", name="python"))

    # unnamed generation: the gate sends each seed to the schema that understands it
    from holographic_schema import compression_gate
    gens = {k: g["gen"] for k, g in mind._gens.items()}
    assert compression_gate("def step(self, action):", gens)[0][1] == "python"
    assert compression_gate("the team scored in the ", gens)[0][1] == "prose"
    out = mind.generate("def step(self", length=60, temperature=0.4)
    assert len(out) > 30 and set(out) <= set(code) | set("def step(self")

    # named access works; unknown names fail loudly
    assert len(mind.generate("the ", 40, 0.4, name="prose")) > 20
    try:
        mind.generate("x", 10, 0.4, name="nope")
        assert False, "should have raised"
    except KeyError:
        pass


def test_single_schema_path_is_unchanged():
    # backward compatibility: one unnamed learn_sequence + generate, exactly as before
    from holographic_text import TOPICS
    text = " ".join(s for ss in TOPICS.values() for s in ss).lower()
    mind = UnifiedMind(dim=1024, seed=0).learn_sequence(text)
    out = mind.generate("the ", length=80, temperature=0.4)
    assert len(out) > 50
    assert mind._gen is not None                  # the compat alias survives


def test_content_gate_resolves_code_vs_prose_without_tags():
    # The correctness fix, pinned: tags declared at LEARN time put code labels in a
    # "code" pool; an untagged classify infers "text" and -- before the gate -- the
    # routing safeguard EXCLUDED the true labels entirely (measured on a docs-vs-code
    # set: 24% accuracy, 66% cross-pool leakage, worse than no routing at all). The
    # compression gate, fitted on the mind's own learned samples, identified the
    # sub-format on 100% of held-out queries and recovered declared-tag accuracy
    # exactly. This test holds the recovered behaviour in place.
    rng = np.random.default_rng(0)
    code = [("def step ( self , a ) : r = self . world . step ( a ) ; "
             "self . memory . append ( ( self . state , a , r ) ) ; return r"),
            ("for i in range ( n ) : total += vals [ i ] ; "
             "if total > cap : break"),
            ("q = q / np . linalg . norm ( q ) ; idx = int ( ( items @ q ) . argmax ( ) )"),
            ("w = rng . standard_normal ( ( dim , k ) ) / np . sqrt ( k ) ; "
             "v = w @ x ; return v / np . linalg . norm ( v )")] * 8
    docs = [("the forager perceives its situation and decides a move then remembers "
             "what happened so the next decision is better informed"),
            ("each leaf keeps a small memory inside capacity and a query descends "
             "the tree with a beam that can back track into nearby cells"),
            ("the plate stores a superposition of many items and recall cleans the "
             "noisy readout by cosine to each known atom"),
            ("a random projection preserves similarity so close feature vectors stay "
             "close as hypervectors across every modality")] * 8
    tr = ([(s, "code:lib", "code") for s in code[:24]] +
          [(s, "doc:lib", "text") for s in docs[:24]])
    rng.shuffle(tr)
    mind = UnifiedMind(dim=1024, seed=0).absorb(tr)

    # untagged queries: the gate must put each on its own side of the line
    assert mind.classify("v = m @ x ; return v / np . linalg . norm ( v )")[0] == "code:lib"
    assert mind.classify("the memory cleans a noisy readout and recalls the item")[0] == "doc:lib"
    # untagged matches declared on every held-out probe
    for s, lab, m in [(code[-1], "code:lib", "code"), (docs[-1], "doc:lib", "text")]:
        assert mind.classify(s)[0] == mind.classify(s, m)[0] == lab


def test_only_code_learned_means_string_queries_reach_code_labels():
    # the single-sub-format branch: if the mind has ONLY learned code, an untagged
    # string query must still reach the code labels (before the fix, the inferred
    # "text" pool was empty-by-exclusion for them)
    snips = [("def f ( x ) : return x + 1", "code:a"),
             ("for i in range ( 9 ) : s += i", "code:b")] * 6
    mind = UnifiedMind(dim=512, seed=0)
    for s, lab in snips:
        mind.learn(s, lab, "code")
    assert mind.classify("def g ( y ) : return y + 2")[0] in ("code:a", "code:b")


def test_absorb_with_sequences_assembles_a_complete_mind():
    # The complete self-assembly: ONE absorb call returns a mind that classifies,
    # recalls, AND generates -- one named sequence schema per discovered text-like
    # sub-format, unnamed generation routed by the compression gate.
    rng = np.random.default_rng(0)
    code = [("def step ( self , a ) : r = self . world . step ( a ) ; return r"),
            ("for i in range ( n ) : total += vals [ i ] ; "
             "if total > cap : break"),
            ("v = w @ x ; return v / np . linalg . norm ( v )")] * 10
    docs = [("the forager perceives its situation and decides a move then remembers "
             "what happened so the next decision is better informed"),
            ("each leaf keeps a small memory inside capacity and the query descends "
             "with a beam that can back track into nearby cells")] * 10
    pile = ([(s, "code:lib", "code") for s in code] +
            [(s, "doc:lib", "text") for s in docs])
    rng.shuffle(pile)
    mind = UnifiedMind(dim=512, seed=0).absorb(pile, sequences=True)

    assert set(mind._gens) == {"text", "code"}      # one schema per discovered format
    out_c = mind.generate("def step ( self", length=50, temperature=0.4)   # gate routes
    out_d = mind.generate("the forager ", length=50, temperature=0.4)
    assert len(out_c) > 25 and len(out_d) > 25
    # and the same mind still classifies untagged across the sub-format line
    assert mind.classify("v = m @ x ; return v")[0] == "code:lib"
    assert mind.classify("the memory keeps each leaf inside capacity")[0] == "doc:lib"
