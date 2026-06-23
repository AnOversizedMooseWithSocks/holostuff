"""Tests for B6: the Tero flow-conductance maze solver -- optimal, deterministic, and matching the
true shortest path on braided mazes; the principled-physics counterpart to the stochastic ant."""

from holographic_creature import GridWorld
from holographic_flow import tero_solve, solve_maze_flow


def test_tero_finds_optimal_on_braided_mazes():
    for fs in (3, 7, 11):
        w = GridWorld(16, 16, maze=True, fixed_seed=fs, braid=1.0)
        path, info = solve_maze_flow(w)
        assert info["reached"]
        assert info["extracted_len"] == info["optimal"]    # flow collapses onto the shortest tube


def test_tero_is_deterministic():
    w = GridWorld(16, 16, maze=True, fixed_seed=7, braid=1.0)
    p1, _ = solve_maze_flow(w)
    p2, _ = solve_maze_flow(w)
    assert p1 == p2                                          # no randomness: identical every run


def test_tero_picks_the_short_route_through_a_loop():
    # a braided diamond: 0-1-2 (short, 2 edges) vs the detour 0-3-4-2 (3 edges)
    nbr = {0: [1, 3], 1: [0, 2], 2: [1, 4], 3: [0, 4], 4: [3, 2]}
    path = tero_solve(nbr, 0, 2, steps=120)
    assert path == [0, 1, 2]


def test_disconnected_graph_returns_none():
    nbr = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
    assert tero_solve(nbr, "A", "D", steps=30) is None


def test_missing_endpoint_returns_none():
    nbr = {0: [1], 1: [0]}
    assert tero_solve(nbr, 0, 9, steps=10) is None          # goal not in the graph


def test_solve_maze_flow_reports_optimum_and_determinism_flag():
    w = GridWorld(16, 16, maze=True, fixed_seed=15, braid=1.0)
    path, info = solve_maze_flow(w)
    assert info["deterministic"] is True
    assert info["extracted_len"] == info["optimal"] and info["cells"] > 0
