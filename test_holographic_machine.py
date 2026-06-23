"""Holographic stored-program machine: programs encoded as hypervectors, executed by VSA ops."""
import numpy as np
from holographic_ai import bind, bundle, cosine, permute
from holographic_machine import HoloMachine


def test_program_executes_exactly():
    m = HoloMachine(dim=4096, seed=7)
    prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    acc, trace = m.run(m.assemble(prog))
    expected = bundle([bind(m.data_atoms["a"], m.data_atoms["b"]), m.data_atoms["c"]])
    assert cosine(acc, expected) > 0.99
    assert trace == [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")]


def test_halt_stops_execution():
    m = HoloMachine(dim=2048, seed=3)
    prog = [("LOAD", "a"), ("HALT", ""), ("BIND", "b")]   # the BIND must never run
    acc, trace = m.run(m.assemble(prog))
    assert trace == [("LOAD", "a")]
    assert cosine(acc, m.data_atoms["a"]) > 0.99


def test_permute_op():
    m = HoloMachine(dim=2048, seed=1)
    acc, _ = m.run(m.assemble([("LOAD", "a"), ("PERMUTE", ""), ("HALT", "")]))
    assert cosine(acc, permute(m.data_atoms["a"], 1)) > 0.99


def test_modest_program_decodes_fully():
    # a 32-instruction program at dim 4096 reads back perfectly (well under the capacity cliff).
    import random
    m = HoloMachine(dim=4096, seed=5)
    rng = random.Random(0)
    prog = [(rng.choice(["LOAD", "BIND", "BUNDLE", "PERMUTE"]), rng.choice(m.data_names)) for _ in range(32)]
    pv = m.assemble(prog)
    assert all(m.decode_instruction(pv, i) == prog[i] for i in range(32))


def test_inception_clean_nesting_is_deep():
    # a program nested as the ONLY file at each level survives many levels (near-lossless bind chain).
    m = HoloMachine(dim=4096, seed=7)
    base = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    want = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")]
    v = m.assemble(base)
    for _ in range(6):
        v = m.disk(v)                       # wrap with no other files
    for _ in range(6):
        v = m.open_slot(v)
    _, trace = m.run(v)
    assert trace == want


def test_inception_busy_disk_has_a_depth_floor():
    # KEPT NEGATIVE / the law: with other files on each disk, a buried program corrupts with depth.
    m = HoloMachine(dim=4096, seed=7)
    base = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
    want = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c")]

    def nest(depth):
        v = m.assemble(base)
        for d in range(depth):
            v = m.disk(v, m.junk_files(3, d))
        for d in range(depth):
            v = m.open_slot(v)
        return m.run(v)[1]

    assert nest(2) == want          # shallow nesting on a busy disk still works
    assert nest(8) != want          # deep nesting on a busy disk does not -- the floor is real


def test_call_runs_a_library_function():
    # a function embedded in the holographic library, invoked by name, transforms the accumulator.
    m = HoloMachine(dim=4096, seed=7)
    m.define("tag_b", [("BIND", "b"), ("HALT", "")])     # an ACC->ACC function: ACC = bind(ACC, b)
    acc, trace = m.run(m.assemble([("LOAD", "a"), ("CALL", "tag_b"), ("HALT", "")]))
    assert ("CALL", "tag_b") in trace
    assert cosine(acc, bind(m.data_atoms["a"], m.data_atoms["b"])) > 0.99


def test_call_composes_library_functions():
    # two functions from ONE library vector compose like ordinary code.
    m = HoloMachine(dim=4096, seed=7)
    m.define("tag_b", [("BIND", "b"), ("HALT", "")])
    m.define("shift", [("PERMUTE", ""), ("HALT", "")])
    acc, _ = m.run(m.assemble([("LOAD", "a"), ("CALL", "tag_b"), ("CALL", "shift"), ("HALT", "")]))
    from holographic_ai import permute as _p
    assert cosine(acc, _p(bind(m.data_atoms["a"], m.data_atoms["b"]), 1)) > 0.99


def test_library_is_one_vector():
    # the whole function library is a single hypervector addressable by name.
    m = HoloMachine(dim=2048, seed=2)
    m.define("f", [("BIND", "b"), ("HALT", "")])
    m.define("g", [("PERMUTE", ""), ("HALT", "")])
    assert m.library.shape == (2048,)


def test_run_backward_compatible_without_call():
    # programs that don't use CALL behave exactly as before (init_acc defaults to None).
    m = HoloMachine(dim=4096, seed=7)
    acc, trace = m.run(m.assemble([("LOAD", "a"), ("BIND", "b"), ("HALT", "")]))
    assert trace == [("LOAD", "a"), ("BIND", "b")]
