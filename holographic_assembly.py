"""B6 (part 2) -- fragment assembly as a flow search: the Tero solver generalised beyond mazes.

The Baker/Rosetta seat builds a global structure from a library of local fragments under an energy.
Its combinatorial core is: choose which fragment sits at each position so that (a) consecutive
fragments overlap-agree and (b) the total placement energy is minimised. That is a MIN-COST PATH through
a layered (position x fragment) trellis -- exactly what the Tero flow solver finds. So the maze solver
(a min-cost path on a grid) and fragment assembly (a min-cost path on a trellis) are the SAME search,
and the chosen fragments come back out as a B7 typed structure (each fragment bound to its position).

Energy is encoded as path LENGTH: a placement of energy c becomes c+1 unit hops via relay nodes, so the
unit-length Tero solver's shortest path IS the minimum-energy assembly (min hops == min energy). The +1
keeps every layer transition costing at least one hop, so the trellis stays strictly layered.

MEASURED vs the exact DP (Viterbi) optimum:
  * Complete library -> assembles the target EXACTLY (energy 0).
  * Library missing a true fragment -> forced mismatches; the flow assembly's energy MATCHES the DP
    optimum (e.g. 9 == 9), i.e. it finds the globally best assembly, not a locally greedy one.

KEPT NEGATIVES / scope: this is the combinatorial CORE of fragment assembly (choose fragments to minimise
an energy), not a protein force field -- the "energy" here is a placement mismatch, a stand-in for the
Rosetta score. The relay-node energy encoding bloats the graph by the total energy (fine for small
libraries/targets; for large energies, weight Tero's edges by length directly instead). Like the maze
solver it is centralized (a Laplacian solve per step). It is the principled flow-search generalisation
the research program pointed to; a full assembler with a real energy is a larger effort.

Pure NumPy + holostuff spirit; deterministic; reuses tero_solve (B6) and StructureRecipe (B7).
"""

from holographic_flow import tero_solve
from holographic_recipe import StructureRecipe


def _energy(frag, pos, target):
    """Placement energy: mismatches between `frag` placed at `pos` and the target (the Rosetta-score
    stand-in)."""
    return sum(1 for j in range(len(frag)) if frag[j] != target[pos + j])


def _build_trellis(target, library, K):
    """Layered (position, fragment) trellis. Edges weighted by placement energy, encoded as unit hops
    via relay nodes so the unit-length Tero solver finds the min-ENERGY path. Returns (nbr, last)."""
    last = len(target) - K
    nbr = {}

    def add_cost(u, v, c):
        prev = u
        for k in range(c):
            relay = (u, v, k)
            nbr.setdefault(prev, []).append(relay)
            nbr.setdefault(relay, []).append(prev)
            prev = relay
        nbr.setdefault(prev, []).append(v)
        nbr.setdefault(v, []).append(prev)

    for f in library:
        add_cost("S", (0, f), _energy(f, 0, target) + 1)        # +1: every transition is >=1 hop
        add_cost((last, f), "T", 1)
    for pos in range(last):
        for f in library:
            for g in library:
                if f[1:] == g[:-1]:                              # fragments must overlap-agree
                    add_cost((pos, f), (pos + 1, g), _energy(g, pos + 1, target) + 1)
    return nbr, last


def assemble(target, library, frag_len=2, steps=300, mu=1.5, dt=0.2, dim=1024, seed=0):
    """Assemble `target` from `library` (overlapping fragments) by MIN-ENERGY flow search (Tero).
    Returns a dict: assembled string, its energy, the chosen (pos, fragment) list, and a B7
    StructureRecipe binding each fragment to its position -- the assembly as a typed holographic
    structure. Deterministic."""
    K = frag_len
    nbr, last = _build_trellis(target, library, K)
    path = tero_solve(nbr, "S", "T", steps=steps, mu=mu, dt=dt)
    if path is None:
        return {"assembled": None, "energy": None, "fragments": None, "recipe": None}
    chosen = [n for n in path if isinstance(n, tuple) and len(n) == 2 and isinstance(n[0], int)]
    assembled = chosen[0][1] + "".join(g[1][-1] for g in chosen[1:])
    energy = sum(_energy(f, p, target) for (p, f) in chosen)
    r = StructureRecipe(dim, seed)                              # the assembly as a B7 typed structure
    parts = [r.bind(r.atom(f"pos:{p}", unitary=True), r.atom(f"frag:{f}")) for (p, f) in chosen]
    r.mark_output(r.superpose(parts))
    return {"assembled": assembled, "energy": energy, "fragments": chosen, "recipe": r}


def assemble_optimal_energy(target, library, frag_len=2):
    """Exact minimum-energy assembly via DP (Viterbi over the trellis) -- the reference the flow search
    must match."""
    K = frag_len
    last = len(target) - K
    INF = 10 ** 9
    dp = {(0, f): _energy(f, 0, target) for f in library}
    for pos in range(last):
        nd = {}
        for f in library:
            if (pos, f) not in dp:
                continue
            for g in library:
                if f[1:] == g[:-1]:
                    e = dp[(pos, f)] + _energy(g, pos + 1, target)
                    if e < nd.get((pos + 1, g), INF):
                        nd[(pos + 1, g)] = e
        dp.update(nd)
    return min(dp[(last, f)] for f in library)
