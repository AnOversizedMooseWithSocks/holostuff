"""Relations: meaning as the recovered relationship -- pinned.

Four operations, each measured before integration (see holographic_relations'
module docstring): EXPLAIN why two records are similar (per-role verdict),
NAME how a filler relates to a record, MAP an attribute across records ("the
dollar of mexico"), and CHAIN hops into complex queries. Plus the law that
shaped the API: meaning survives composition only when it touches SYMBOLS
between steps -- the symbol-routed map is exact (360/360) where the direct
algebraic map is ~94% and does not improve with dimension.
"""

import numpy as np

from holographic_relations import KnowledgeStore, relation_map, _cleanup
from holographic_ai import bind, cosine

WORLD = {
    "france":  dict(capital="paris", currency="franc", language="french", continent="europe"),
    "belgium": dict(capital="brussels", currency="franc", language="french", continent="europe"),
    "sweden":  dict(capital="stockholm", currency="krona", language="swedish", continent="europe"),
    "japan":   dict(capital="tokyo", currency="yen", language="japanese", continent="asia"),
    "mexico":  dict(capital="mexico_city", currency="peso", language="spanish", continent="america"),
    "usa":     dict(capital="washington", currency="dollar", language="english", continent="america"),
    "peru":    dict(capital="lima", currency="sol", language="spanish", continent="america"),
    "egypt":   dict(capital="cairo", currency="pound", language="arabic", continent="africa"),
    "kenya":   dict(capital="nairobi", currency="shilling", language="swahili", continent="africa"),
    "vietnam": dict(capital="hanoi", currency="dong", language="vietnamese", continent="asia"),
}


def store():
    ks = KnowledgeStore(dim=2048, seed=0)
    for n, a in WORLD.items():
        ks.add(n, **a)
    return ks


def test_explain_decodes_why_two_records_are_similar():
    ks = store()
    verdicts = {r: (fa, fb, shared)
                for r, fa, fb, shared, _ in ks.explain("france", "belgium")}
    assert verdicts["capital"] == ("paris", "brussels", False)
    assert verdicts["currency"] == ("franc", "franc", True)
    assert verdicts["language"] == ("french", "french", True)
    assert verdicts["continent"] == ("europe", "europe", True)


def test_name_relation_says_how_a_filler_relates():
    # measured 40/40; pinned at a >= 90% floor over the whole world
    ks = store()
    ok = tot = 0
    for n, attrs in WORLD.items():
        for r, v in attrs.items():
            ok += (ks.name(n, v)[0] == r)
            tot += 1
    assert ok / tot >= 0.9


def test_symbol_routed_mapping_is_exact_where_direct_map_is_noisy():
    # THE LAW: the two-step route (name the role -> read it out of the other
    # record) cleans up to a symbol mid-path and measured 360/360; the direct
    # algebraic map skips that cleanup and measured ~94%, with 20 of its 22
    # failures pure HRR noise (not ambiguity) and no improvement with dimension.
    ks = store()
    ok = tot = 0
    for a in WORLD:
        for b in WORLD:
            if a == b:
                continue
            for r, v in WORLD[a].items():
                ans, _ = ks.the_x_of(v, b, a)
                ok += (ans == WORLD[b][r])
                tot += 1
    assert ok / tot >= 0.98                       # measured exact; tiny slack

    # the direct map stays useful but measurably noisier
    direct_ok = direct_tot = 0
    fillers = ks._filler_names()
    for a in ("usa", "japan", "kenya"):
        for b in WORLD:
            if a == b:
                continue
            M = relation_map(ks.recs[a], ks.recs[b])
            for r, v in WORLD[a].items():
                ans, _ = _cleanup(bind(M, ks.fillers.get(v)), fillers, ks.fillers)
                direct_ok += (ans == WORLD[b][r])
                direct_tot += 1
    assert direct_ok / direct_tot >= 0.80         # documented noise floor


def test_chained_queries_stay_exact_through_three_hops():
    ks = store()
    # 2-hop: the currency of the country whose capital is X
    for n, attrs in WORLD.items():
        assert ks.ask(attrs["capital"], ("capital", "currency")) == attrs["currency"]
    # 3-hop (franc is shared by france+belgium -- both are french-speaking, so
    # the answer is right whichever the hop lands on: honest ambiguity, not noise)
    for n, attrs in WORLD.items():
        lang = ks.ask(attrs["capital"], ("capital", "currency"),
                      ("currency", "language"))
        assert lang == attrs["language"]


def test_unified_mind_explains_its_own_records():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=2048, seed=0)
    out = {r: (f1, f2, shared) for r, f1, f2, shared, _ in m.explain(
        {"capital": "paris", "currency": "franc", "language": "french"},
        {"capital": "brussels", "currency": "franc", "language": "french"})}
    assert out["capital"] == ("paris", "brussels", False)
    assert out["currency"] == ("franc", "franc", True)
    assert out["language"] == ("french", "french", True)
