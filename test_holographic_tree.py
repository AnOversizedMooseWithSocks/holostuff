"""Tests for holographic_tree: deterministic recursive partition, capacity
restoration vs a flat memory, and the approximate-NN speed/recall tradeoff."""
import numpy as np
from holographic_ai import random_vector, HolographicMemory
from holographic_tree import HoloTree, HoloForest, StructuredIndex, capacity_curve, nn_benchmark, forest_benchmark


def _items(N, dim, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([random_vector(dim, rng) for _ in range(N)])


def test_tree_is_balanced_and_log_depth():
    tree = HoloTree(256, leaf_size=32, seed=0).build(_items(1000, 256))
    st = tree.stats()
    assert st["max_leaf"] <= 32 and st["leaves"] >= 1000 // 32
    assert st["depth"] <= 2 * np.log2(1000 / 32) + 3        # roughly log N / leaf


def test_build_is_deterministic():
    items = _items(500, 256)
    a = HoloTree(256, leaf_size=32, seed=7).build(items)
    b = HoloTree(256, leaf_size=32, seed=7).build(items)
    assert a.stats() == b.stats()
    q = items[3] + 0.4 * random_vector(256, np.random.default_rng(99))
    assert a.recall(q, beam=4) == b.recall(q, beam=4)       # same seed -> same answer


def test_exact_key_value_recall_perfect_in_tree():
    rng = np.random.default_rng(1); dim = 2048; N = 1024
    keys = np.stack([random_vector(dim, rng) for _ in range(N)])
    vals = np.stack([random_vector(dim, rng) for _ in range(N)])
    tree = HoloTree(dim, leaf_size=64, seed=0).build(keys, vals)
    ok = sum(int(tree.recall(keys[i]) == i) for i in range(N))
    assert ok / N >= 0.98                                   # leaves stay within capacity


def test_tree_beats_flat_when_dataset_is_big():
    # the headline: a single flat memory collapses past capacity; the tree holds
    rows = {r["N"]: r for r in capacity_curve([64, 1024], dim=2048, leaf_size=64, probes=120)}
    assert rows[64]["flat"] >= 0.95 and rows[64]["tree"] >= 0.95   # both fine when small
    assert rows[1024]["flat"] < 0.4                                # flat has collapsed
    assert rows[1024]["tree"] >= 0.95                              # tree still works


def test_nn_recall_improves_with_beam_and_is_cheaper():
    lo = nn_benchmark(N=1200, dim=512, leaf_size=64, beam=1, noise=0.5)
    hi = nn_benchmark(N=1200, dim=512, leaf_size=64, beam=16, noise=0.5)
    assert hi["tree_recall"] > lo["tree_recall"]            # more beam -> better recall
    assert hi["exact_recall"] >= 0.98                        # exact scan is the ceiling
    assert hi["tree_cmp"] < hi["exact_cmp"]                  # and the tree is cheaper


def test_flux_concentrates_like_veins():
    # after many varied queries, flux is uneven -- a few thick veins, many thin
    tree = HoloTree(256, leaf_size=32, seed=0).build(_items(800, 256))
    rng = np.random.default_rng(0)
    for _ in range(400):
        tree.recall(random_vector(256, rng), beam=3)
    flux = np.array(tree.flux())
    assert flux.sum() > 0 and flux.std() > 0                # not uniform


def test_forest_beats_single_tree_at_matched_cost():
    single = nn_benchmark(N=1500, dim=512, leaf_size=64, beam=4, noise=0.5)
    forest = forest_benchmark(N=1500, dim=512, leaf_size=64, n_trees=4, beam=4, noise=0.5)
    assert forest["forest_recall"] > single["tree_recall"]          # more trees, more recall
    assert forest["forest_recall"] >= 0.98                          # reaches ~exact
    assert forest["forest_cmp"] < forest["exact_cmp"]               # still cheaper than a scan


def test_forest_recall_is_correct_on_clean_keys():
    items = _items(600, 256)
    forest = HoloForest(256, n_trees=3, leaf_size=48, seed=0).build(items)
    ok = sum(int(forest.recall(items[i], beam=4) == i) for i in range(0, 600, 5))
    assert ok == len(range(0, 600, 5))                              # exact cues -> exact hits


def test_forest_recall_agreement_is_an_abstention_signal():
    # The trees are independently seeded, so their agreement is a free reliability signal.
    # A stored item recalls with full agreement; random queries split the trees. The
    # default recall (with_agreement=False) is unchanged byte-for-byte.
    import numpy as _np
    from holographic_tree import HoloForest
    rng = _np.random.default_rng(0)
    V = _np.stack([rng.standard_normal(256) for _ in range(300)])
    V /= _np.linalg.norm(V, axis=1, keepdims=True)
    f = HoloForest(256, n_trees=8, seed=0).build(V)
    # default path unchanged
    for i in range(60):
        assert f.recall(V[i]) == f.recall(V[i], with_agreement=True)[0]
    hit, a_hit = f.recall(V[7], with_agreement=True)
    assert hit == 7 and a_hit > 0.7
    rand_agree = _np.mean([f.recall(rng.standard_normal(256), with_agreement=True)[1]
                           for _ in range(40)])
    assert a_hit > rand_agree + 0.1                 # agreement separates known from unknown


# ---- StructuredIndex: the shared content-address index ---------------------------------------------

def test_structured_index_locates_by_content_sublinearly():
    # filed under the items THEMSELVES (rule 1): a clean cue routes home, and at scale it costs far less
    # than a flat scan -- the whole point of giving the index structure.
    dim, N = 512, 3000
    items = _items(N, dim, seed=0)
    idx = StructuredIndex(dim, n_trees=6, leaf_size=64, seed=0).build(items)
    probe = list(range(0, N, 9))
    hits = sum(idx.locate(items[t], beam=6)[0] == t for t in probe)
    assert hits == len(probe)                          # query == key -> reliable routing
    assert idx.locate(items[0], beam=6)[1] < N         # sub-linear: fewer comparisons than a flat scan


def test_structured_index_returns_payload_labels_not_row_numbers():
    dim = 512
    items = _items(40, dim, seed=1)
    labels = [f"city:{i}" for i in range(40)]
    idx = StructuredIndex(dim, seed=0).build(items, payloads=labels)
    assert idx.locate_exact(items[17])[0] == "city:17"
    # structured payloads ride along too (the route use case: (chunk, step))
    ridx = StructuredIndex(dim, seed=0).build(items, payloads=[(i // 14, i) for i in range(40)])
    assert ridx.locate_exact(items[30])[0] == (2, 30)


def test_structured_index_exact_scan_is_exact_and_flat():
    dim, N = 512, 600
    items = _items(N, dim, seed=2)
    idx = StructuredIndex(dim, seed=0).build(items)
    for t in range(0, N, 5):
        payload, comps = idx.locate_exact(items[t])
        assert payload == t and comps == N             # guaranteed nearest, full scan


def test_structured_index_k_nearest_ranked_by_cosine():
    dim = 512
    items = _items(500, dim, seed=3)
    idx = StructuredIndex(dim, n_trees=6, leaf_size=64, seed=0).build(items)
    res = idx.locate_k(items[12], k=4, beam=6)
    assert res[0][0] == 12 and abs(res[0][1] - 1.0) < 1e-6     # itself first, cosine ~1
    cosines = [s for _, s in res]
    assert cosines == sorted(cosines, reverse=True)            # descending


def test_structured_index_agreement_is_a_free_abstention_signal():
    dim = 512
    items = _items(800, dim, seed=4)
    idx = StructuredIndex(dim, n_trees=6, leaf_size=64, seed=0).build(items)
    payload, _, agree = idx.locate(items[5], beam=6, with_agreement=True)
    assert payload == 5 and 0.0 <= agree <= 1.0 and agree > 0.6   # clean cue -> trees agree


def test_structured_index_rejects_mismatched_payloads():
    dim = 512
    items = _items(10, dim, seed=5)
    try:
        StructuredIndex(dim, seed=0).build(items, payloads=["only", "three", "labels"])
        assert False, "expected ValueError for payload/key length mismatch"
    except ValueError:
        pass
