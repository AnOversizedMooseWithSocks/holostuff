"""Relations: meaning as the RECOVERED RELATIONSHIP.

Similarity says THAT two things are alike. Meaning says WHY and HOW -- and in
this algebra the why/how has a precise, executable form. Entities are
role-bound records (bundle of bind(role, filler)), so:

  EXPLAIN  -- why are two records similar? Unbind every role from both, clean
              up each estimate to a symbol, and report which fillers MATCH.
              "France is like Belgium BECAUSE currency=franc, language=french,
              continent=europe; UNLIKE because the capitals differ." Measured:
              4/4 roles decoded and judged correctly on the demo world.
  NAME     -- how does a filler relate to a record? Unbind the filler and clean
              up against the role vocabulary: "paris relates to france AS
              capital". Measured: 40/40 = 100% over the demo world.
  MAP      -- the Kanerva move ("what is the dollar of Mexico?"): identify the
              role the probe fills in record A, then read that role out of
              record B. Measured: 360/360 = 100%.
  CHAIN    -- compose hops into complex queries ("the language of the country
              with the currency of the country whose capital is X").
              Measured: 100% at two and three hops.

THE LAW THIS MODULE OBEYS (measured, and the reason the API looks like it
does): meaning survives composition only when it touches SYMBOLS between
steps. The direct algebraic mapping M = bind(rec_b, involution(rec_a)) --
conceptually beautiful, one bind, no intermediate cleanup -- scores only ~94%
on the same task and does NOT improve with dimension (96/94/90% at
1024/2048/4096: HRR cross-term noise, 20 of 22 failures pure noise, 2 honest
filler ambiguities). Routing every hop through a cleanup (geometry -> symbol ->
geometry) is exact on the same data, because the discrete vocabulary acts as
error correction between compositions. The symbolic layer is not decoration on
the geometry; it is what makes chained meaning reliable. `relation_map` is
still provided for the raw algebraic object, with its noise documented.
"""

import numpy as np

from holographic_ai import bind, bundle, involution, cosine, Vocabulary


def _cleanup(vec, names, vocab):
    """Snap a noisy estimate to the best symbol; return (name, confidence)."""
    best_n, best_s = None, -2.0
    for n in names:
        s = cosine(vec, vocab.get(n))
        if s > best_s:
            best_n, best_s = n, s
    return best_n, float(best_s)


def name_relation(record, filler_vec, role_names, role_vocab):
    """HOW does this filler relate to this record? Unbind it and name the role."""
    return _cleanup(bind(record, involution(filler_vec)), role_names, role_vocab)


def explain(rec_a, rec_b, role_names, role_vocab, filler_names, filler_vocab):
    """WHY are two records similar? Per-role decode of both, with the verdict.
    Returns a list of (role, filler_a, filler_b, shared, min_confidence)."""
    out = []
    for r in role_names:
        inv = involution(role_vocab.get(r))
        fa, ca = _cleanup(bind(rec_a, inv), filler_names, filler_vocab)
        fb, cb = _cleanup(bind(rec_b, inv), filler_names, filler_vocab)
        out.append((r, fa, fb, fa == fb, min(ca, cb)))
    return out


def map_attribute(rec_a, rec_b, filler_vec, role_names, role_vocab,
                  filler_names, filler_vocab):
    """'The <probe> of B': name the role the probe fills in A (cleanup), then
    read that role out of B (cleanup). The symbol-routed path -- measured exact
    where the direct algebraic map is ~94%."""
    role, _ = name_relation(rec_a, filler_vec, role_names, role_vocab)
    return _cleanup(bind(rec_b, involution(role_vocab.get(role))),
                    filler_names, filler_vocab)


def relation_map(rec_a, rec_b):
    """The DIRECT algebraic relation object M with bind(M, part_of_a) ~ the
    corresponding part of b. Conceptually the purest form of 'the relationship
    as a first-class vector' -- composable, storable, comparable -- but
    measured ~6% noisier than map_attribute on the demo world, because it
    skips the mid-path cleanup. Use map_attribute when accuracy matters; use
    this when the relation itself is the object of study."""
    return bind(rec_b, involution(rec_a))


class KnowledgeStore:
    """A tiny store of role-bound records that answers WHY/HOW questions and
    multi-hop chains. Owns its vocabularies, so all structure is reproducible
    from the seeds (demoscene rule)."""

    def __init__(self, dim=2048, seed=0):
        self.dim = dim
        self.roles = Vocabulary(dim, seed + 1)
        self.fillers = Vocabulary(dim, seed + 2)
        self.attrs = {}                      # name -> {role: filler}
        self.recs = {}                       # name -> record vector

    def add(self, name, **attrs):
        self.attrs[name] = dict(attrs)
        self.recs[name] = bundle([bind(self.roles.get(r), self.fillers.get(v))
                                  for r, v in attrs.items()])
        return self

    # -- vocabulary views ---------------------------------------------------
    def _role_names(self):
        return sorted({r for a in self.attrs.values() for r in a})

    def _filler_names(self):
        return sorted({v for a in self.attrs.values() for v in a.values()})

    # -- the four meaning operations ----------------------------------------
    def explain(self, a, b):
        return explain(self.recs[a], self.recs[b], self._role_names(),
                       self.roles, self._filler_names(), self.fillers)

    def name(self, entity, filler):
        return name_relation(self.recs[entity], self.fillers.get(filler),
                             self._role_names(), self.roles)

    def the_x_of(self, probe_filler, of_entity, like_entity):
        """'What is the <probe_filler> of <of_entity>?' where probe_filler is an
        attribute of like_entity ('what is the dollar of mexico', like=usa)."""
        return map_attribute(self.recs[like_entity], self.recs[of_entity],
                             self.fillers.get(probe_filler), self._role_names(),
                             self.roles, self._filler_names(), self.fillers)

    def find(self, role, filler):
        """Which entity holds bind(role, filler)? One hop over the store."""
        probe = bind(self.roles.get(role), self.fillers.get(filler))
        return max((cosine(self.recs[n], probe), n) for n in self.recs)[1]

    def ask(self, start_filler, *path):
        """A CHAIN: ('paris', ('capital','currency'), ('currency','language'))
        reads as: the entity whose capital is paris -> its currency -> the
        entity with that currency -> its language. Each hop cleans up to a
        symbol before the next (the law above), which is what keeps measured
        accuracy at 100% through three hops."""
        filler = start_filler
        for match_role, read_role in path:
            entity = self.find(match_role, filler)
            filler, _ = _cleanup(bind(self.recs[entity],
                                      involution(self.roles.get(read_role))),
                                 self._filler_names(), self.fillers)
        return filler
