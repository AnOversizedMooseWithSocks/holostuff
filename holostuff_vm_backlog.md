# holostuff VM / procedure backlog

*The procedure arc (M1-M7) turned the stored-program VM into a real little language: values, binding,
faculty calls (APPLY), subroutines + recursion (CALL), goal-addressable recall, recipe completion,
synthesis, conditionals (IFMATCH), and a fixed-point loop (ITERATE). This backlog finishes the VM into
something you can actually compute non-trivial things with, then picks up the strongest procedure-arc
follow-ons. Ordered by value; each item carries an honest scope and a measurement bar. Same discipline as
the modules: delegate to existing machinery, keep negatives, close-out ritual per item, clean zip.*

---

## VM completion

### VM-1 — matmul in the loop (`exact_matmul` as an APPLY faculty)   [HIGH]  ✓ DONE
*Shipped: `ITERATE [APPLY matmul]` converges a stochastic matrix to its stationary distribution (cosine 0.9993, 3 iters). 820->822.*
**What.** Wire the RNS exact matmul in as an APPLY handler so a loop body can do real linear algebra on
the accumulator: `APPLY matmul` means `ACC := W @ ACC` for a matrix the mind is configured with. With
ITERATE this is the literal recurrent pattern — input, process by a matrix, feed the result back, repeat.
**Why.** This is the bridge between "the magic of AI is looping" and the exact matmul the engine already
has. `ITERATE [APPLY matmul]` is a recurrent linear map iterated to a fixed point — a real algorithm, not a
toy.
**How.** `set_matmul(W)` configures the mind's current matrix; `_procedure_handlers` gains a `matmul`
entry doing `exact_matmul(W, acc)` (fixed-point exact). Register `matmul` in DEFAULT_FACULTIES.
**Scope/risk.** ACC is treated as a raw vector for this step (leaving the VSA algebra deliberately); W must
be dim x dim to keep ACC iterable; floats are fixed-point quantized (the only error is that rounding, not
crosstalk). One configured matrix at a time.
**Bar.** A worked iterative algorithm: configure a column-stochastic transition matrix, `ITERATE [APPLY
matmul]` on an initial distribution, and converge to the stationary distribution (matches the dominant
eigenvector). Backward-compatible; bare VM `APPLY matmul` is a no-op.

### VM-2 — counted loop (`REPEAT <n>`)   [MED-HIGH]  ✓ DONE
*Shipped: `REPEAT n; CALL f` runs f n times (exact via permute). 822->824.*
**What.** `REPEAT n` repeats the NEXT instruction n times (n a small-integer count atom), completing the
loop set: a counted FOR alongside ITERATE's convergence/goal WHILE.
**Why.** Some loops are "do this exactly k times" (unroll a fixed-depth recurrence, apply a transform k
times). ITERATE can't express a fixed count cleanly.
**How.** A small operand codebook of count atoms (`cnt:1..cnt:8`); `REPEAT` reads the count and re-executes
the following single instruction that many times (the next instruction is typically a CALL for a multi-op
body). Mirrors IFMATCH's "gate the next instruction" structure.
**Scope/risk.** Counts bounded to the count-atom set (say 1..8); repeats ONE following instruction (use CALL
for a block). Honest: not arbitrary loop bounds.
**Bar.** `REPEAT cnt:3; CALL f` runs f exactly three times (verified by the accumulator and the trace);
decodes back as data; bare-VM safe.

### VM-3 — a real worked program + nested-control validation   [MED]  ✓ DONE
*Shipped: nested loops + a denoise->classify->tag worked program, deterministic. 824->826.*
**What.** Assemble ONE non-trivial procedure that combines the new primitives (e.g. condition -> iterate a
matmul/cleanup body -> branch on the result), proven end to end; plus tests that nested control flow
(ITERATE inside a CALLed body, REPEAT around a CALL, IFMATCH guarding an ITERATE) composes and the depth
guard holds.
**Why.** Control flow is only real if it composes. A worked program is the proof and the demo.
**How.** A `tour` program + integration tests exercising 2-3 levels of nested control flow.
**Bar.** The worked program computes the right result; nested-control tests are green; determinism holds
run-to-run.

---

## Procedure-arc follow-ons

### PIPE-1 — automatic data-analysis pipeline as a VSA PROGRAM   [DONE +5, 826->831]
**Result.** `run_analysis_pipeline(signal)` runs ONE HoloMachine program (APPLY analyze; ITERATE denoise;
APPLY decompose; IFMATCH structured; CALL train+validate; APPLY save) -- each APPLY delegates to a real
faculty. Structured signal (poly+noise): finds the 2-term law (explained var 0.998), held-out error 0.10,
256 pts -> 157 B law, CALL fires. Pure noise: explained var 0.0, the IFMATCH SKIPS the CALL, saved raw_only.
The loop body denoises the signal against its own low-rank trajectory (SSA/Cadzow), the prior a lone vector
lacks (~3.4x noise cut, ~idempotent so ITERATE settles). **Recursive "every-level" peel SHIPPED (+4, 831->835):**
recursive=True swaps the single decompose for `ITERATE _peel_step`, peeling structure layer by layer until the
MDL gate admits no term. Noisy trend+sine -> 3 levels (line trend -> mobius periodic -> cleanup), cumulative
0.997 where one decompose gets ~0.29; poly+exp stops at 1 level; noise and trend+2sines find 0 (honest limits
of decompose's topology detection). Gating on the MDL verdict (n_terms>=1), NOT an explained-fraction floor,
was the fix -- a real level can be small. Bare sine on a line domain still missed (inherited).

### SYN-1 — deeper synthesis (meet-in-the-middle)   [RESOLVED: measured negative + canonicalize +4, 835->839]
**Outcome.** Meet-in-the-middle is NOT built -- the precondition was measured and it would be dead complexity.
The invertible algebra COLLAPSES: any interleaving of k binds and m permutes == permute(x, m) bound by the
operand product (cosine 1.0000), so a depth-(k+m) program is depth-<=2 -- nothing deep to find, and M5's
fingerprint already solves that class from one example. The ops where depth IS real (BUNDLE, nonlinear APPLY)
do not invert cleanly, so the backward search has no clean target. Kept negative, documented in NOTES.
**The flip (shipped):** `canonicalize_procedure(program)` -- the collapse as a program OPTIMIZER. Reduces a
bind/permute program to its minimal verified-equivalent form (k binds -> 1 product bind; m permutes stay unit
shifts): 5 ops -> 3, five binds -> 1, all verified by execution (cosine 1.0). Also an equivalence oracle (two
reorderings -> same canonical). BUNDLE/nonlinear ops are honest barriers (refused, not partial).

### REC-1 — sublinear procedure recall (HoloForest-indexed fingerprints)   [RESOLVED: forest rejected, scan vectorized +1, 839->840]
**Outcome.** Forest-indexing is NOT used -- measured premature twice over. A HoloForest over fingerprints is
3-8x SLOWER than a linear scan for realistic libraries (N=50-1000), crossing over only ~N=4000 -- and the
single library vector cannot hold ~4000 procedures (bundle crosstalk caps it at a few hundred). Accuracy is
fine; the issue is speed. Kept negative, documented in NOTES.
**The flip (shipped):** the real bottleneck was that the O(N) fingerprint scan was a PYTHON LOOP. Vectorized
it -- cache fingerprints as one unit-normalized matrix, recall = one matvec (mat @ qhat) + argmax: 6-26x over
the loop, 3-7x over the forest, same identity/score as the loop (verified). Named-subset path keeps the loop.


### GEN-1 — operand prediction in recipe completion   [DONE +3, 840->843]
**Result.** learn_recipe_grammar now also trains a JOINT (opcode, operand) predictor; `complete_instruction`
predicts the full next instruction (opcode AND operand). Patterned operands (templates a->b->c, d->e->f):
held-out operand accuracy 1.00 (learns context->operand). Random operands: chance (0.23, 1/6=0.17) -- a random
operand is unknowable -- but the opcode SHAPE is still predicted (operand-independent), so complete_procedure
stays the robust call. KEPT NEGATIVE: the n-gram CONFIDENCE is unreliable (a context seen once returns 1.00),
so the honest discriminator is generalization, not the score. Backward-safe (no grammar -> (None,None,0.0)).

---

## All backlog items resolved (VM-1/2/3, PIPE-1 + recursive peel, SYN-1, REC-1, GEN-1). 795 -> 843 tests.
Three shipped capabilities (data-analysis pipeline as a VSA program + recursive peel; program canonicalizer;
operand prediction), two measured negatives that redirected to the real fix (meet-in-the-middle -> the algebra
collapses so canonicalize instead; forest recall -> premature, vectorize the scan instead), and the VM
completion (matmul-in-loop, counted REPEAT, worked programs).
