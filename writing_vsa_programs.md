# Writing VSA Programs in holostuff

*A guide to `HoloMachine` — the small stored-program machine that lets you express custom logic as a
holographic program: a sequence of instructions encoded into ONE hypervector, executed on the same
bind/bundle/cleanup algebra as the rest of the engine. Every example below was run and its result is shown.*

## What this is for

Sometimes you don't want a permanent faculty of the mind — you want to run *your own* logic over the
vector algebra: a scientist with a dataset who wants to try several experiments, none of which needs to
become a fixture of holostuff; or a game with NPCs whose behaviour should live at the VSA level so it
reacts to game data automatically when a trigger fires. `HoloMachine` is that: you write a short program,
`assemble()` it into a single vector, and `run()` it. A program is *data* — content-addressable, storable,
and composable in the same space as everything else — so it's cheap to make, throw away, and remake.

The engine's job is to provide the primitives; your program composes them. `HoloMachine` is the
composition layer for one-off / domain-specific logic that shouldn't be baked into the core.

## The model in one breath

A program is a Python list of `(opcode, operand)` tuples. `assemble()` encodes the whole list as ONE
hypervector — each instruction is `bind(POS_i, bundle(bind(OP, opcode), bind(ARG, operand)))`. `run()`
reads each address back by unbinding its position, **cleans** the opcode and operand against their
codebooks (so the accumulator is built from *exact* atoms even though the read itself is noisy), and
executes. It returns `(accumulator, trace)` — the final ACC vector and the list of decoded instructions
that actually ran.

There is one register: the **accumulator** (ACC). Instructions transform it.

## Quick start

```python
from holographic_machine import HoloMachine

vm = HoloMachine(dim=4096, seed=7)            # default data alphabet is 'a'..'f'
prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("HALT", "")]
acc, trace = vm.run(vm.assemble(prog))
# acc == bundle(bind(a, b), c)   -> cosine 1.0000
# trace == [('LOAD','a'), ('BIND','b'), ('BUNDLE','c')]
```

`HALT` and `PERMUTE` take no real operand — pass `""`. The data atoms `a`..`f` exist by default; to use
your own names, pass `data=[...]` to the constructor (below).

## The instruction set

| Instruction | Operand | Effect on ACC |
|---|---|---|
| `LOAD x` | data name | `ACC = x` (put a value in the accumulator) |
| `BIND x` | data name | `ACC = bind(ACC, x)` — associate (the algebra's "multiply") |
| `BUNDLE x` | data name | `ACC = bundle([ACC, x])` — superpose (the algebra's "add") |
| `PERMUTE` | `""` | `ACC = permute(ACC, 1)` — shift (encodes order/position) |
| `CALL f` | function name | run library function `f` (an ACC→ACC sub-program) on ACC |
| `APPLY g` | faculty name | `ACC = g(ACC)` via a host-supplied handler (the bridge to the mind) |
| `IFMATCH x` | data name | run the NEXT instruction only if `cosine(ACC, x) >= branch_tol` (default 0.5), else skip it |
| `REPEAT n` | count 1..8 | run the FOLLOWING `CALL` n times |
| `ITERATE f` | function name | re-apply function `f` to ACC until it converges (fixed point) or a host `stop(acc)` is met |
| `HALT` | `""` | stop |

Operand codebooks are kept separate, which is why cleanup is reliable: `LOAD/BIND/BUNDLE/IFMATCH` operands
clean against the **data** atoms, `CALL/ITERATE` against **function names**, `APPLY` against **faculty
names**, `REPEAT` against the small-integer **counts**.

## Functions: name a sub-program, call it by name

A function is an ACC→ACC sub-program embedded into a holographic library and invoked by name. Define it
before assembling any program that calls it.

```python
vm.define("tag_b", [("BIND", "b"), ("HALT", "")])     # ACC := bind(ACC, b)
acc, trace = vm.run(vm.assemble([("LOAD", "a"), ("CALL", "tag_b"), ("HALT", "")]))
# acc == bind(a, b)   -> cosine 1.0000 ;  trace == [('LOAD','a'), ('CALL','tag_b')]
```

Functions are data too — they're stored in the same vector space, so they compose and nest (a function
can CALL another, with a recursion-depth guard).

## Triggers: reacting to data with `IFMATCH`

This is the "react to game data when a trigger is hit" pattern, and it's the VSA-native answer to
"do we have callbacks?" — `IFMATCH` is a one-instruction conditional that gates whatever comes next on a
*similarity* test against the accumulator. Pair it with `CALL` for an if-then: *if the current state looks
like this trigger, run this response.*

```python
g = HoloMachine(dim=4096, seed=7, data=["enemy_near", "calm", "flee_signal"])
g.define("raise_alarm", [("LOAD", "flee_signal"), ("HALT", "")])      # the response

# the reactive program: IF state matches 'enemy_near', run raise_alarm
trigger = g.assemble([("IFMATCH", "enemy_near"), ("CALL", "raise_alarm"), ("HALT", "")])

# fire it on a state that MATCHES the trigger (start ACC at the current state):
acc_hit,  _ = g.run(trigger, init_acc=g.data_atoms["enemy_near"])
# -> ACC == flee_signal (cosine 1.0000): the response ran. trace: [('IFMATCH','enemy_near'), ('CALL','raise_alarm')]

# fire the SAME program on a state that does NOT match:
acc_miss, _ = g.run(trigger, init_acc=g.data_atoms["calm"])
# -> ACC == calm (cosine 1.0000): the response was skipped. trace: [('IFMATCH','enemy_near')]
```

The match is a cosine threshold, so the trigger fires not just on the exact atom but on anything *close
enough* to it — which is what you want for a noisy game state that resembles the trigger condition. Tune
the threshold per `run(..., branch_tol=...)`.

A practical NPC loop, then, is: encode the NPC's current observation into ACC (with the mind's encoder),
run a small trigger program that checks ACC against the conditions you care about and CALLs the matching
response — all at the VSA level, reusing your atoms. The host code just feeds the observation in and reads
the response out.

## Loops

```python
vm.define("shift", [("PERMUTE", ""), ("HALT", "")])
acc, _ = vm.run(vm.assemble([("LOAD", "a"), ("REPEAT", 3), ("CALL", "shift"), ("HALT", "")]))
# acc == permute(a, 3)   -> cosine 1.0000  (REPEAT runs the following CALL n times)
```

`ITERATE f` is the fixed-point loop — re-apply `f` to ACC until it stops changing (`cosine(acc, prev) >=
converge_tol`) or a host `stop(acc)` predicate says the desired output is reached. That's the
input→process→feed-back loop behind cleanup / resonator / denoise, now expressible as a program.

## Calling the mind's faculties: `APPLY` + handlers

`APPLY g` runs a real holostuff faculty on ACC — but the *bare* VM has no faculties, so the host (your
code, or the mind) supplies them as a `handlers` dict mapping a faculty name to a unary `acc -> acc`
function. An `APPLY` whose faculty has no handler is a **safe no-op**, so a program always runs.

```python
fm = HoloMachine(dim=512, seed=7, data=["a", "b", "c"], faculties=["cleanup"])
codebook = {n: fm.data_atoms[n] for n in ("a", "b", "c")}
def cleanup_handler(acc):                       # a real cleanup: snap ACC to the nearest known atom
    return max(codebook.values(), key=lambda v: cosine(acc, v))

acc, _ = fm.run(fm.assemble([("APPLY", "cleanup"), ("HALT", "")]),
                init_acc=noisy_a, handlers={"cleanup": cleanup_handler})
# noisy_a had cosine 0.90 to atom 'a'; after APPLY cleanup -> cosine 1.00.
# With NO handler supplied, APPLY is a no-op and ACC is unchanged (still 0.90).
```

This is the seam between a VSA program and the engine: wire `handlers={"denoise": mind.denoise, "cleanup":
mind.cleanup, ...}` and your program can invoke the mind's measured faculties on its accumulator.

## Ephemeral by design (the scientist's use case)

Because a program is just a vector, you make one, run it, and discard it — none of it has to become part of
holostuff. A scientist can hold a dataset's items as `data=[...]` atoms, then assemble and run a different
program per experiment, reusing the same machine. Nothing is registered globally; nothing persists unless
you choose to store the program vector (which you can — it's content-addressable like any other vector).

## Honest limits (the capacity wall, kept on the record)

`HoloMachine` is real HRR, so it has HRR's finite capacity — this is measured and not hidden:

- **Program length (the drive's capacity cliff).** Instruction-decode holds ~100% up to a length that
  scales with dimension — solid through roughly **18 instructions at dim 1024** (and more at higher dim),
  then near the ~20 edge the decode turns *operand-dependent* (some 20-instruction programs decode, others
  don't) and beyond it bundling crosstalk overwhelms cleanup. The way past it is **`run_chunked`** (below) —
  *not* factoring into `define`d functions, which does **not** help (see the note in that section).
- **Nesting depth.** A program can be stored as a "file" inside a "disk" inside a disk… effectively
  unbounded when each level is clean, but bounded to ~3–4 levels when each disk also holds other files (a
  busy level corrupts a buried program after a few hops). Both numbers scale with dimension.
- **The read is noisy, the result is exact.** Reading an instruction back is a noisy unbind, but operands
  are *cleaned to exact atoms* before use — so the accumulator is built from clean atoms and the computed
  value is exact even though the program-reading step is approximate.

## Running a program past the cap: `run_chunked`

A long program — a scientist's experiment protocol, a long data-processing pipeline — outgrows one
structure. `run_chunked` runs it anyway by splitting it into ≤`chunk`-instruction pieces, each its **own
clean program vector**, and **threading the accumulator** across them. The accumulator is the only thing
that crosses a seam, exactly like a re-anchored route carries only its last clean tile.

```python
vm = HoloMachine(dim=1024, seed=7)
names = [chr(ord("a") + i % 6) for i in range(60)]
long_program = [("LOAD", names[0])] + [("BIND", names[i]) for i in range(1, 60)] + [("HALT", "")]

vm.run(vm.assemble(long_program))     # 60 instructions in ONE structure -> cosine 0.08 to the right answer (the cliff)
vm.run_chunked(long_program)          # host-threaded <=14-instr chunks    -> cosine 1.00, exact, past the cap
```

`run_chunked(program, chunk=14, init_acc=None, handlers=None, ...)` returns `(acc, trace)` just like `run()`,
with the trace the chunks' traces concatenated. Notes:

- **Keep `chunk` well under the edge.** The default 14 leaves deliberate margin below the dim-1024 ~20 edge
  (sitting *on* the edge fails for some programs — measured). Raise it at higher dim (20 is solid at dim 2048+).
- **Why not `define`/`CALL` instead?** Because it does **not** work: `CALL` pulls each sub-program out of a
  *bundled library*, and bundling several function-vectors into one library re-introduces the very cliff
  (measured: the same 60-instruction program via `CALL` collapses to cosine 0.06). Functions are still great
  for *reuse*; they are not the tool for *length*. `run_chunked`'s independent host-threaded chunks are.
- **Control flow is kept intact at seams.** A chunk never ends on `IFMATCH`/`REPEAT` (the gate/repeat and the
  instruction it targets stay together), but don't rely on a single construct spanning a boundary beyond that.
  Put `HALT` at the end (a trailing one is stripped; a mid-program `HALT` stops the whole run).

## API summary

```python
vm = HoloMachine(dim=4096, seed=7, data=[...], faculties=[...])  # data/faculty names define the codebooks
vm.define(name, program)                 # embed an ACC->ACC function, callable by CALL/ITERATE/REPEAT
program_vec = vm.assemble(program)       # list of (opcode, operand) -> ONE vector
acc, trace = vm.run(program_vec,         # execute; returns (accumulator, decoded-instruction trace)
                    init_acc=None,        #   start ACC at a given vector (e.g. the current NPC state)
                    handlers={...},       #   faculty name -> unary acc->acc function (for APPLY)
                    stop=None,            #   predicate acc->bool to end an ITERATE early (goal reached)
                    branch_tol=0.5,       #   IFMATCH similarity threshold
                    max_steps=512)        #   safety cap on total instructions executed
acc, trace = vm.run_chunked(program,     # run a program TOO LONG for one structure: thread the accumulator
                    chunk=14)             #   across clean <=chunk-instruction pieces (default 14; raise at higher dim)
```

`HoloMachine` lives in `holographic_machine.py`. It is intentionally *adjacent* to the mind, not a faculty
of it — the mind exposes primitives; `HoloMachine` is how you compose your own program out of them.
