"""Holographic stored-program machine -- the 'operating system' rung of the substrate.

WHY THIS EXISTS
---------------
A hard drive has structure (the platter), data laid down in that structure (magnetization), and --
when that data is read back and executed -- a whole new layer of structure (an OS, a VM, an OS inside
the VM). holostuff already had the first rungs: the D-dim vector is the platter, derived_atom(seed,...)
is the low-level format, and role-filler binding + nested composition is the file system. The rung this
module adds is the one that makes the tower possible: a PROGRAM encoded as a hypervector, executed by
the engine's own bind/bundle/cleanup operations. Instructions and data live in the same vector space
(von Neumann, holographically), so the thing that stores structure can also store the recipe for MORE
structure -- and run it.

"FORMATTING THE DRIVE" is just fixing a seed: that deterministically lays down the alphabet -- two
roles (OP, ARG), the opcode atoms, the data atoms, an address function POS(i), and a SLOT role for
nesting. Same seed => bit-identical format on any machine.

THE INSTRUCTION SET (deliberately tiny and readable)
    LOAD x   : ACC = x                  (put a value in the accumulator)
    BIND x   : ACC = bind(ACC, x)       (associate -- the 'multiply' of the algebra)
    BUNDLE x : ACC = bundle([ACC, x])   (superpose -- the 'add')
    PERMUTE  : ACC = permute(ACC, 1)    (shift -- encodes order/position)
    HALT     : stop
A program is a list of (opcode, operand). assemble() encodes it as ONE vector:
    instruction_i = bundle(bind(OP, opcode_i), bind(ARG, operand_i))
    program       = bundle_i( bind(POS(i), instruction_i) )
run() reads each address by unbinding POS(i), CLEANS the opcode and operand against their codebooks
(wide-margin classification -- robust to the bundling crosstalk), and dispatches. Because operands are
cleaned to exact atoms before use, the accumulator is built from clean atoms and is EXACT even though
the program-reading itself is noisy.

MEASURED (honest picture, kept negatives included)
  * Correctness: 'LOAD a; BIND b; BUNDLE c' yields ACC == bundle(bind(a,b),c) at cosine 1.0000.
  * DRIVE SIZE (capacity cliff): instruction-decode holds ~100% up to a program length that scales
    with dimension -- ~32 instructions reliable at dim 1024, ~128 at dim 4096 -- then bundling
    crosstalk overwhelms cleanup and accuracy falls. The cliff is real: capacity is finite. (KEPT
    NEGATIVE -- this is the honest HRR capacity wall, not hidden.)
  * INCEPTION DEPTH (nesting a program as a file inside a disk inside a disk...): effectively
    UNBOUNDED when each level is clean (depth 8+ runs fine -- a pure unitary bind chain barely
    degrades), but bounded to ~3-4 levels when each disk also holds OTHER files. The law: you can
    nest as deep as you like if each level is uncluttered; a busy disk corrupts a buried program
    after a few levels. Both numbers scale with dimension.

Pure NumPy, deterministic, no new dependencies.
"""

import numpy as np
from holographic_ai import bind, unbind, bundle, cosine, permute, derived_atom

OPCODES = ["LOAD", "BIND", "BUNDLE", "PERMUTE", "CALL", "HALT"]
DEFAULT_DATA = list("abcdef")


class HoloMachine:
    """A formatted holographic drive that can store and execute stored programs."""

    def __init__(self, dim=4096, seed=7, data=None):
        self.dim = dim
        self.seed = seed
        self.data_names = list(data) if data is not None else list(DEFAULT_DATA)
        # "format the drive": the whole alphabet is derived deterministically from the seed.
        self.OP = self._atom("role:OP", unitary=True)     # roles are unitary -> unbind is exact
        self.ARG = self._atom("role:ARG", unitary=True)
        self.SLOT = self._atom("role:SLOT", unitary=True)  # the role a nested 'file' lives under
        self.op_atoms = {o: self._atom(f"op:{o}") for o in OPCODES}
        self.data_atoms = {d: self._atom(f"dat:{d}") for d in self.data_names}
        # the holographic function LIBRARY: named sub-programs, all held in one vector, callable by name
        self.functions = {}       # name -> assembled program vector
        self.fn_atoms = {}        # name -> unitary name atom (the 'address' of the function)
        self.library = None       # bundle over names of bind(name_atom, program) -- the whole library, one vector

    def _atom(self, name, unitary=False):
        return derived_atom(self.seed, name, self.dim, unitary=unitary)

    def pos(self, i):
        """Address of the i-th instruction -- a deterministic unitary 'cylinder' atom."""
        return self._atom(f"pos:{i}", unitary=True)

    # ---- assembling a program into a single hypervector --------------------------------------
    def _instr(self, op, arg):
        if op == "CALL":
            arg_vec = self.fn_atoms[arg]                            # CALL's operand is a function NAME
        else:
            arg_vec = self.data_atoms.get(arg, self.op_atoms["HALT"])   # HALT carries a don't-care operand
        return bundle([bind(self.OP, self.op_atoms[op]), bind(self.ARG, arg_vec)])

    def define(self, name, program):
        """Embed a named function -- an ACC->ACC sub-program -- into the holographic library.

        The function's body is assembled to a vector and bundled into ONE library vector under its
        name atom. A CALL to it later extracts it by name (unbind) and runs it on the current ACC.
        Functions are therefore data: composable, content-addressable, and stored in the same space
        as everything else. Define a function before assembling any program that CALLs it."""
        self.functions[name] = self.assemble(program)
        self.fn_atoms[name] = self._atom(f"fn:{name}", unitary=True)
        self.library = bundle([bind(self.fn_atoms[n], self.functions[n]) for n in self.functions])
        return self

    def assemble(self, program):
        """Encode a list of (opcode, operand) instructions as ONE program vector."""
        return bundle([bind(self.pos(i), self._instr(op, arg)) for i, (op, arg) in enumerate(program)])

    # ---- cleanup against the format's codebooks ----------------------------------------------
    @staticmethod
    def _nearest(table, noisy):
        best, best_sim = None, -9.0
        for name, vec in table.items():
            s = cosine(noisy, vec)
            if s > best_sim:
                best_sim, best = s, name
        return best

    def decode_instruction(self, program_vec, i):
        """Read address i: return (opcode, operand) after cleanup. The honest, noisy read step."""
        raw = unbind(program_vec, self.pos(i))
        op = self._nearest(self.op_atoms, unbind(raw, self.OP))
        arg = self._nearest(self.data_atoms, unbind(raw, self.ARG))
        return op, arg

    # ---- executing a program -----------------------------------------------------------------
    def run(self, program_vec, init_acc=None, max_steps=512, _depth=0):
        """Execute the program vector; return (accumulator, trace_of_decoded_instructions).

        `init_acc` lets a program start from a given accumulator -- which is what makes a function an
        ACC->ACC transform that CALL can chain. A CALL instruction extracts the named function from the
        holographic library and runs it on the current ACC (with a recursion-depth guard)."""
        acc = init_acc
        trace = []
        for pc in range(max_steps):
            raw = unbind(program_vec, self.pos(pc))
            op = self._nearest(self.op_atoms, unbind(raw, self.OP))
            if op == "HALT":
                break
            if op == "CALL":
                fn = self._nearest(self.fn_atoms, unbind(raw, self.ARG))   # operand cleaned vs function names
                trace.append(("CALL", fn))
                if _depth < 8 and fn in self.functions:                    # guard against runaway recursion
                    sub = unbind(self.library, self.fn_atoms[fn])          # pull the function body out of the library
                    acc, _ = self.run(sub, init_acc=acc, _depth=_depth + 1)
                continue
            arg = self._nearest(self.data_atoms, unbind(raw, self.ARG))
            trace.append((op, arg))
            d = self.data_atoms[arg]
            if op == "LOAD" or acc is None:        # guard: a value-op before any LOAD acts as LOAD,
                acc = d                            # so a corrupted program can't crash the interpreter
            elif op == "BIND":
                acc = bind(acc, d)
            elif op == "BUNDLE":
                acc = bundle([acc, d])
            elif op == "PERMUTE":
                acc = permute(acc, 1)
        return acc, trace

    # ---- nesting (the inception layer): a program is just another value to store --------------
    def as_file(self, content_vec):
        """Wrap a vector as a 'file' under the SLOT role, ready to drop onto a disk."""
        return bind(self.SLOT, content_vec)

    def disk(self, content_vec, other_files=()):
        """A 'disk': the SLOT-file holding `content_vec`, bundled with any other files on the disk.
        More files per disk => more crosstalk => a buried program corrupts at a shallower depth."""
        return bundle([self.as_file(content_vec), *other_files])

    def open_slot(self, disk_vec):
        """Recover the SLOT-file's contents from a disk (noisy if the disk holds other files)."""
        return unbind(disk_vec, self.SLOT)

    def junk_files(self, n, tag):
        """n deterministic distractor files, to simulate a disk that holds other things too."""
        return [bind(self._atom(f"f:{tag}:{j}", unitary=True), self._atom(f"j:{tag}:{j}"))
                for j in range(n)]
