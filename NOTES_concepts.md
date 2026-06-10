# Concept notes -- physics/nature analogies vs. the substrate

This project runs on one rule: an idea earns its place only by beating a baseline
under honest measurement, and negatives are kept. These are six natural-process
analogies considered as possible improvements, filtered by that rule. The point
of writing them down is the *reasoning* and the one *measured boundary* -- not a
pile of new features. Most of these are deliberately NOT implemented; the value
is knowing why.

## 1. Double diffusion / salt fingering  -- TESTED, clean negative (a boundary)

Physics: heat diffuses ~100x faster than salt, so a layer that is stable in
temperature but unstable in salinity throws "fingers" of mixing that neither
variable makes alone. The fast and slow signals, and the gap between them, are
the whole mechanism.

Mapping: a memory prototype has a FAST signal (its centroid -- every observation
nudges it) and a SLOW signal (internal stratification -- members quietly
splitting into sub-modes, which takes many observations to show). The analogy
predicts a cheap pre-screen for `auto_reorganize`: a class can be STABLE in
held-out accuracy while UNSTABLE in internal variance -- the finger -- a split
forming before accuracy degrades. If the variance signature ranked classes by
how much a split helps, the expensive resolution sweep could run only on
"fingering" classes.

What the measurement said: it does not transfer to the hyperdimensional
substrate, and the reason is quantified. A two-means stratification signal
(how much better two centroids fit a class's members than one) separates truly
bimodal classes from unimodal blobs by **3.9 sigma at dim 8**, but the
separation decays to ~0.3 sigma -- pure noise -- by dim 128 and stays there at
512 (the working dimension). Concentration of measure is the culprit: in high
dimensions a unimodal Gaussian blob looks just as "splittable" to k=2 as a
genuinely bimodal class, so the local variance gradient the fingers need does
not exist. (`_exp_double_diffusion.py` reasoning, dimension sweep recorded here.)

The useful conclusion: this EXPLAINS why the existing accuracy-only trigger is
the right design. The cheap variance pre-screen the analogy suggests is not a
missed optimization -- it is mathematically unavailable at 512-d. The slow
signal has to be held-out accuracy itself, which is exactly what
`auto_reorganize` already measures. A real salt-finger memory would need a
low-dimensional variable to stratify on; the substrate deliberately has none.

## 2. Surface tension  -- NOT implemented (likely a refinement, and we have a curation negative)

Water minimizes surface area; a membrane resists deformation in proportion to
curvature. Mapping: the reorganization gate could resist splitting a prototype
cluster that is already smooth/coherent (high tension) and yield for a lumpy one
(low tension), replacing the flat 1-SE leanness margin with a coherence-scaled
one. Plausible and measurable, but (a) it is a refinement of a gate that already
works, and (b) the codebase has a clean negative on layering a curation
controller over the organizer's own aggregation. Worth a small experiment only
if the 1-SE rule is later shown to mis-fire; not a priority.

## 3. Gravity lensing  -- NOT implemented (mostly re-describes existing machinery)

Mass bends geodesics; light bends around it and can multiply into rings. Mapping:
high-frequency prototypes should bend nearby queries toward themselves. But that
is already what the ReflexCache's hot set and any frequency weighting do -- the
analogy re-describes a prior we have. Its one non-redundant prediction is
*multiple images*: a query near a massive prototype routed two legitimate ways,
which maps to keeping both a coarse and a fine prototype for a heavily trafficked
class -- and that is what the multi-resolution organizer already does on demand.
No new mechanism falls out.

## 4. Flocking (boids)  -- TESTED, clean negative

Three local rules (separation, alignment, cohesion) produce global coordination
with no leader. Mapping is to a RECORDED negative: averaging independently
trained minds was worse than picking the best single one (their policies differ
too much to average), so the UI trains several and keeps the best. Flocking
proposes a third option that negative did not rule out: candidate policies
nudged toward LOCAL agreement only with their nearest neighbours in policy
space -- not global averaging (which destroyed them) and not single best-pick
(the current winner), but local alignment that keeps global diversity.

Tested as a reversible decision-time value blend: each candidate's action scores
pulled 30% toward its single most-similar peer's scores, then the committee
votes. Two regimes, 16x16 mazes:

  * Well-trained candidates: flock ties best-pick at 100% everywhere AND beats
    plain averaging where averaging collapses (maze 5: flock 100%, average 0%) --
    so local alignment really does dodge the averaging failure. Encouraging, but
    best-pick is already saturated, so no separation.
  * Under-trained candidates (the regime that matters -- where a committee should
    earn its keep): best-pick **100%**, flock **67% mean, 0% worst** (collapsed
    on 2 of 6 mazes).

The why is decisive: when candidates disagree, local alignment can pull a GOOD
mind toward a confidently-wrong neighbour -- flocking has no notion of which
neighbour is correct, only who is nearby. Best-pick's probe, even noisy, directly
MEASURES escape and selects a winner. Measurement beats consensus; the recorded
negative stands and best-pick remains the champion. (`_exp_flock.py` reasoning.)

This sharpens a principle the whole project already runs on: where a cheap
measurement of the real objective is available (probe escape rate, held-out
accuracy, compression bits), trust it over any structural prior about how the
candidates "should" relate. Flocking is a structural prior; it loses to the
measurement.

## 5. Prism / spectral decomposition  -- partially realised; premise CHECKED and refuted for the open problem

A prism separates superposed frequencies by refractive index. The holographic
substrate IS superposition, and unbind/cleanup IS separation, so this is less an
analogy than a description of what the engine already does (the ResonatorNetwork
factoring a scene into per-object attribute atoms is a literal prism).

The target it seemed to point at: the survival forager bundles "food east" +
"danger north" + walls into ONE superposed state prototype, and the guess was
that this fusion fragments learning and causes the recorded wall-pocket
dithering. Before building a state-decomposition mechanism, the premise was
checked cheaply -- instrument a long cluttered-world life and ask whether
dithering correlates with state ALIASING (distinct physical cells sharing a
near-identical bundled state, which is what fusion would cause). It does NOT:
only 3.5% of distinct-cell state pairs alias (cos > 0.85), and 89% of visited
cells get a STABLE chosen action. The forager is not confused about where it is;
it makes a locally-consistent but globally-trapped choice at pocket entrances.
Fusion is not the disease, so a prism is not the cure -- the premise check saved
building the wrong mechanism.

Where the prism analogy IS already real and correct: the ResonatorNetwork in the
compositional-scene work, which genuinely splits a superposed scene back into its
attribute bands. That is the legitimate home of the idea; the forager is not.

## 7. Projection (one 3-D object, many 2-D shadows)  -- TESTED, the first WIN, with a measured hazard

The concept: a single complex object casts many different shadows depending on
the projection plane; conversely, where many flat patterns OVERLAP, the overlap
is a registration mark that they are projections of one higher object (the
Contact-blueprint reading: the pages only made sense stacked).

Where it is literally true in this system: the creature brain's state vectors
are bundles of a SMALL atom vocabulary, so its thousands of 512-D prototypes
must all lie in the span of those atoms -- they are high-dimensional shadows of
one intrinsically low-rank object. Measured on trained brains: 99.9% of
prototype energy sits in **22-24 of 512 dimensions** (forage and 16x16 maze
both). `HolographicMind.consolidate()` discovers the subspace by SVD over the
prototypes themselves -- the overlap IS the registration mark -- and re-stores
the entire memory as coefficients in it. Results at full behavioural parity:
**21x smaller memory, ~5x faster decide()** (forage 122 -> 120 stars at
1.36 -> 0.25 ms/decision; 16x16 maze 90% -> 95% escapes).

The hazard, found by measuring before integrating: **a shadow hides new
structure**. A brain consolidated in a poison-free world compressed to rank 9,
and the danger sense then carried only **4% of its energy inside the basis** --
poison was nearly invisible to its values. So consolidation ships with a
residual guard (the flux-guard pattern, fourth appearance): every incoming
state's out-of-basis energy is tracked as a slow EMA, and when it grows past a
threshold the basis EXPANDS from a small ring of recent raw states (new
orthogonal directions appended; old prototypes get zero coefficients there,
which is exact -- they truly had no such component). Measured under a world
shift: basis 9 -> 13, danger in-basis energy 4% -> 100%, learning continues in
the grown space. Compress when the world is stable, grow when it is not.

Both behaviours are pinned in test_holographic_brain.py. This is the only one
of the seven concepts to produce shipped machinery -- and its hazard would have
shipped too, silently, without the measure-first step.

## 6. Demoscene  -- the operating constraint, not a feature

Maximum effect from minimal, fully deterministic code; seeded RNG everywhere so
every result reproduces. This is already the house style and the discipline the
other five are held to: any addition must stay tiny and seed-reproducible, or it
does not go in.

## 8. Chained concepts -> meaning (the user's follow-on: projection + composition)

The thesis: if things are decomposed and structured properly, basic concepts
chain into complex ones, and the structure should surface not just THAT two
things are similar but WHY and HOW. Tested, and the thesis holds -- with one
deep, measured qualifier about what makes it hold.

In this algebra the why/how has an executable form. Entities are role-bound
records (bundle of bind(role, filler)) -- exactly what the UniversalEncoder
already builds from dicts -- and four operations recover the relationship
(holographic_relations.py, all numbers from the 10-entity demo world):

  EXPLAIN  per-role decode of two records with a match verdict ("france is like
           belgium BECAUSE currency=franc, language=french, continent=europe;
           UNLIKE because the capitals differ") -- 4/4 roles correct.
  NAME     unbind a filler from a record and clean up against the roles:
           "paris relates to france AS capital" -- 40/40 = 100%.
  MAP      "what is the dollar of mexico?" -- name the role the probe fills in
           one record, read that role out of the other -- 360/360 = 100%.
  CHAIN    "the language of the country with the currency of the country whose
           capital is X" -- 100% at two AND three hops.

THE QUALIFIER, which is the real finding: meaning survives composition only
when it touches SYMBOLS between steps. The direct algebraic relation object
M = bind(rec_b, involution(rec_a)) -- one bind, conceptually the purest "the
relationship as a first-class vector" -- scores ~94% on the same mapping task,
its failures are 20/22 pure HRR cross-term noise (only 2 honest filler
ambiguities), and it does NOT improve with dimension (96/94/90% at
1024/2048/4096). Routing every hop through a cleanup (geometry -> symbol ->
geometry) is exact on the same data: the discrete vocabulary acts as error
correction between compositions. The symbolic layer is not decoration on the
geometry -- it is what makes chained meaning reliable. (This is also why the
3-hop chain survives: each hop snaps to a symbol before the noise can compound,
the same per-hop-cleanup law the maze corridor reflex and the generation work
each found in their own domains.)

Integration: holographic_relations.py (KnowledgeStore with explain/name/
the_x_of/find/ask) and UnifiedMind.explain(x1, x2), which articulates the
per-role verdict for any two record dicts using the mind's own encoder, with
cleanup candidates drawn from the inputs themselves. One honest detail pinned
in passing: numeric fillers (wheels: 4 vs 2) decode correctly but at near-zero
confidence, because scalar encodings make nearby numbers near-neighbours BY
DESIGN -- the verdict is right and the low confidence honestly reports that the
call was close. Pinned in test_holographic_relations.py.

---

Scorecard, all testable ideas now measured (two wins, three negatives, two parked):
  (8) chained concepts -> meaning -- WIN, shipped: explain/name/map/chain over
      role-bound records (4/4, 100%, 100%, 100% through 3 hops), with the law
      that the direct symbol-free algebraic route is ~94% and dimension does
      not save it -- meaning survives composition only by touching symbols
      between steps.
  (7) projection -- WIN, shipped: consolidate() compresses the brain's memory
      21x and speeds decide() ~5x at parity, by discovering the low-rank object
      all the prototypes are shadows of; its measured hazard (a shadow hides new
      structure -- danger at 4% in-basis energy after a poison-free
      consolidation) shipped WITH its cure (the expanding residual guard,
      4% -> 100% under a world shift).
  (1) double diffusion -- closed, useful boundary result: the cheap variance
      pre-screen is mathematically unavailable at 512-d (separation decays from
      3.9 sigma at dim 8 to 0.3 sigma noise by dim 128), which is exactly why the
      accuracy-only reorganization trigger is the correct design.
  (4) flocking -- closed, clean negative: local consensus ties best-pick when all
      candidates are good and LOSES when they disagree (67% vs 100%), because
      measurement of the real objective beats a structural prior about how
      candidates relate.
  (5) prism -- premise checked and refuted for the forager (dithering is not
      caused by state fusion: 3.5% aliasing, 89% stable per-cell action); the
      idea is already correctly realised in the ResonatorNetwork.
  (2) surface tension, (3) gravity lensing -- parked with reasons (a refinement
      of a working gate; a re-description of existing machinery).

Three honest negatives and a parked pair -- and the wall-pocket dithering's real
cause is now narrowed by elimination: NOT aliasing, NOT poison risk, NOT memory
depth. It is a stable, locally-consistent bad choice at pocket entrances -- a
value-estimate trap, not a perception or representation problem. That points the
next real attempt at the VALUE side (e.g. an exploration bonus that decays with
revisit count -- repulsion from recently-exhausted ground), not at any of these
six analogies. The analogies did their job: they generated hypotheses, the
measurements killed the wrong ones cheaply, and the elimination sharpened the
real target.
