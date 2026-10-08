# Both panels re-run with a geometric-mean divisor — registered predictions

Written before `rerun.py` exists, per `PROTOCOL_TEMPLATE.md` rule 1.

## What is being tested, and on what data

`experiments/stable_holdout/RESULT.md` records a post-hoc observation: the frozen rule's divisor
`scale_full` is the **arithmetic mean of a checkpoint's eight stable maxima**, the statistic it
summarizes is a maximum of a GLR path and is heavy-tailed, and replacing that mean by the
**geometric mean of the same eight frozen numbers** reverses the sign of the correction's effect
on held-out between-checkpoint dispersion on both panels.

That observation says nothing about detection. This re-runs the **entire panel analysis** under
the geometric divisor and asks what the paper's results would have been:

- thresholds for all twenty checkpoints on both panels, changed and leave-one-replicate-out
  stable variants, rebuilt by the frozen freeze arithmetic with one function substituted;
- in-sample stable false alarms (P2's registered count criterion), 160 streams per panel;
- held-out stable false alarms, 160 streams per panel, on the streams of
  `experiments/stable_holdout` that entered no calibration set;
- detections and restricted delays on all 320 changed streams per panel;
- P1's paired per-checkpoint sign test, re-run exactly as `stage2_analyze.py` adjudicates it;
- the comparison at a **matched held-out false-alarm budget**, which is the only comparison that
  separates a better divisor from a more conservative one.

Nothing is refitted, nothing is recollected, and no GPU is needed. The per-window scores are the
cached ones in `panel_{d,e}_baselines/scores`, the Stage 1 maxima are the frozen ones in
`stage1_streams.csv`, and the held-out maxima are the ones already scored against refit bundles
that reproduced all 320 Stage 1 values to `1e-9`.

**This arm is retrospective and its estimator was chosen after seeing held-out data.** It does
not re-adjudicate Panel D or Panel E, it does not replace any frozen number, and it is reported
beside the registered analysis rather than in place of it. Its purpose is to answer one
question — *was the correction's failure a property of the correction or of the summary function
it happened to use?* — not to supply a new headline.

## Why this estimator, and what its status is

Two arguments, and they have different weights.

**The a-priori one, which does not depend on the held-out data.** The reliability criterion this
revision is built on computes one-way variance components of the **log** maximum statistic
(`experiments/statistic_reliability/reliability.py`, `variance_components(..., value="logmax")`)
and reports `m*rho/(m*rho + 1 - rho)`. That formula is the reliability of a **mean of `m` log
values**, which is the log of their geometric mean. The implemented rule divides by the
arithmetic mean. **The criterion the paper proposes and the estimator the paper runs are not the
same estimator**, and that mismatch was available to be noticed from the paper's own two
sections without any held-out stream. It is a defect in the manuscript independent of how this
re-run comes out, and it is reported as one.

**The a-posteriori one, which does.** The sign of the dispersion change flips. This is the part
that is post-hoc, and no amount of the first argument launders it.

**What does not change: the guarantees.** Proposition 6 requires only that `A_c > 0` be
independent of the calibrated maxima; it says nothing about the form of `A_c`, so the disjoint
bound covers a geometric divisor verbatim. Proposition 7's counterexample uses `m = 1`, where the
geometric and arithmetic means coincide, so the plug-in rule inherits no bound under either
divisor. **Switching divisors changes efficiency, not validity**, and neither withdrawal is
reinstated by anything below.

## What I already knew

From `experiments/stable_holdout/evaluate.py`, on held-out data, before writing this:

- change in held-out between-checkpoint SD of the log maximum under the geometric divisor:
  `-6.7%` on Panel D and `-24.7%` on Panel E, against `+22.6%` and `+7.9%` under the arithmetic
  mean, and `+22.6%` / `-10.4%` under the median;
- across the twenty Panel D checkpoints, log SD of the divisor itself `0.883` arithmetic against
  `0.748` geometric, with the held-out target at `0.631`; correlation with that target `0.519`
  against `0.647`;
- the checkpoint scale persists at `r = 0.716` and `0.664`.

So G-D below — dispersion — is **not** a prediction and is not adjudicated. It is restated as
input.

**I have computed nothing about thresholds, false alarms, detections, delays or either endpoint
under this divisor.** G1 through G6 are blind.

## Predictions

**G1 — the geometric divisor raises most thresholds.** A threshold is
`A_c * max_{c' != c} (max_j W_{c',j} / A_{c'})`. The geometric mean is below the arithmetic mean
for every checkpoint (strictly, unless its eight maxima are equal), so the target's divisor
shrinks and every foreign ratio grows. The foreign maximum is attained at whichever checkpoint
has the widest within-checkpoint spread, which is where the two means differ most, so the
numerator should inflate more than a typical checkpoint's own divisor shrinks.

> On both panels, the geometric threshold exceeds the frozen normalized threshold for **at least
> 15 of 20** checkpoints in the changed-stream variant.

*Against:* a majority falling, which would make the rule uniformly more liberal and any
detection gain below attributable to level rather than to fit.

**G2 — at a matched held-out false-alarm budget, it detects at least as well on Panel D.** This
is the primary comparison and the one the paper's reliability claim stands or falls on. Each
rule's threshold is scaled to spend the same false-alarm budget on the 160 held-out streams —
streams that set no threshold under either divisor — and detections are then counted on the
frozen changed streams.

> Panel D primary-condition detections under the geometric divisor are **at or above** the
> arithmetic rule's count at the same held-out budget.

*Against:* strictly fewer. **That outcome would be serious for this revision and not merely for
this arm:** the reliability criterion is defined through held-out between-checkpoint dispersion,
so a divisor that reduces dispersion on both panels and still detects worse would mean the
criterion orders panels by a quantity that does not govern the endpoint. It would be reported in
those terms.

**G3 — it does not rescue Panel E.** `experiments/walker_pathway/` locates that panel's deficit
upstream of the threshold: the matched score carries `16x` less signal than on Hopper and the
GLR's quadratic accumulation turns that into a `46x` deficit in evidence. A divisor cannot
recover signal that the residualization removed.

> Panel E primary detections under the geometric divisor, at a matched held-out budget, remain
> **at or below 12 of 80** — far below the ADWIN-style comparator's `24`, so the panel's
> registered failure and its comparative failure both stand.

*Against:* exceeding 12, which would mean Walker2d failed because of the estimator rather than
the environment, and would reopen the paper's second headline result.

**G4 — P1 would still have passed on Panel D.** Re-running the registered paired sign test on
per-checkpoint mean restricted delay, pooled minus geometric-normalized, across twenty
checkpoints.

> At least **15** positive differences and a one-sided exact sign-test `p < 0.05`.

*Against:* a failing sign test, which would mean the registered endpoint's success depended on
the arithmetic mean specifically.

**G5 — P2's registered caps would still have been met on Panel D.** In-sample stable false
alarms under the leave-one-replicate-out geometric divisor.

> At most `8` per policy mode of 80 — the registered literal cap — on Panel D.

*Against:* exceeding 8 in either mode, which would mean the geometric rule buys its fit with
specificity the registered protocol would have rejected.

**G6 — held-out specificity does not deteriorate.** Held-out false alarms under the full
eight-stream geometric divisor, scored on the 320 streams of `experiments/stable_holdout`.

> At or below the arithmetic rule's `3 of 160` (Panel D) and `2 of 160` (Panel E).

*Against:* more, which combined with G1 would be internally contradictory and would indicate an
error rather than a finding.

## Claim boundary

If these hold, the paper may say: the correction's failure to reduce held-out dispersion was a
property of the summary function and not of the correction, the paper's own reliability criterion
describes the geometric mean rather than the arithmetic one it ran, and a retrospective re-run of
both panels under that estimator gives *(whatever is observed)*.

It may **not** say: that the geometric divisor is the right estimator, which two panels chosen
after the fact cannot establish; that the registered results would have been these had the
estimator been registered, since the freeze is what makes a registered result mean anything; that
any bound is restored; or that Panel E replicates, under any outcome short of G3 failing, and
even then only as a hypothesis.

**No frozen artifact is written and no reported prospective number changes.** If this arm looks
better than the registered one, that is a reason for a future panel to register it, not a reason
to renumber this one.
