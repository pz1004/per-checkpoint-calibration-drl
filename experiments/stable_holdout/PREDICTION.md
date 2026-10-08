# Independent stable-stream evaluation — registered predictions

Written before `collect.py` and `score.py` exist, per `PROTOCOL_TEMPLATE.md` rule 1.

## What is being tested, and on what data

The second-round review's priority 5: **evaluate the frozen procedure on independent stable
streams, without retuning.** Neither prospective panel holds out a stable set. The 160 streams
that set each panel's thresholds are the 160 whose false alarms are then counted.

This collects **160 new stable streams per panel** --- replicates 4 through 7, both policy
modes, twenty checkpoints --- on Panel D (Hopper-v5) and Panel E (Walker2d-v5). Nothing is
retrained and nothing is retuned:

- policies are reloaded from the frozen `model.zip`;
- updating streams reload the frozen `replay_buffer.pkl`;
- bundles are refit per checkpoint from the frozen `stable_replay.parquet`, which is
  deterministic and must reproduce Stage 1's cached scores **exactly** before any new stream is
  scored (rule 3);
- thresholds, scales and the order statistic are the frozen ones from `thresholds.json`.

Environment seeds continue Stage 1's formula at the new replicate indices, so they are disjoint
from every seed already used: Panel D `30000 + seed*10 + replicate`, Panel E
`20000 + seed*10 + replicate`, replicate in 4..7.

**Nothing frozen is written.** Output goes to `experiments/stable_holdout/panel_{d,e}/`.

## What I already knew, and how much it weakens this

A great deal, and this must be discounted accordingly. Before writing these predictions I had
already computed, from the **in-sample** stage-1 streams:

- the one-way variance components of the log max statistic: ICC `0.125` on Panel D and `0.095`
  on Panel E, with bootstrap intervals over checkpoints covering zero;
- split-half scale persistence within the existing eight streams: log-scale Pearson
  `r = +0.39 (p=0.09)` on Panel D, `+0.22 (p=0.35)` on Panel E;
- that normalizing held-out-within-sample streams by a disjoint own-scale estimate *increases*
  between-checkpoint spread at a four-stream budget;
- from simulation at each panel's empirical shape, marginal false-alarm rates of `0.047` at a
  six-stream scale budget and `0.037` at eight.

So H1 to H4 are **informed extrapolations, not blind predictions**, and a reader should treat a
confirmation as weak corroboration that in-sample structure transfers, not as an independent
discovery. What is genuinely unknown is whether any of it survives contact with streams that
were collected after the thresholds were frozen and that entered no calibration set at all.

H5 is the exception: I have not computed it in any form.

## Predictions

**H1 --- the eight-stream estimator is not worse than the six-stream one.** The changed-stream
rule divides by `scale_full`, an eight-stream mean that has never been evaluated on a null
stream; the reported stable counts come from a six-stream leave-one-replicate-out mean. Larger
budget means a less noisy divisor, so:

> Held-out stable false alarms under the frozen normalized rule with `scale_full` will be **at
> or below** the in-sample six-stream counts --- `6 of 80` on Panel D, `5 of 80` on Panel E ---
> and the held-out rate will lie in `[0.01, 0.09]` on both panels.

*Against:* a held-out count above 8 of 80 on either panel, or a rate outside that interval.

**H2 --- checkpoint scale persists weakly on Hopper and not measurably on Walker2d.** Correlate
each checkpoint's frozen eight-stream `scale_full` against the mean of its eight held-out
maxima, on the log scale, across twenty checkpoints.

> Panel D: Pearson `r` in `[0.20, 0.70]`, positive.
> Panel E: `r` below `0.45`, with a 95% interval containing zero.

*Against:* `r > 0.70` on either panel, which would mean the scale is far more persistent than
the split-half estimate implies and that the remedy has a firmer target than I claim.

**H3 --- the calibrated statistic's ICC stays small.** One-way variance components of the log
max statistic on the 160 held-out streams.

> ICC in `[0.00, 0.30]` on both panels, with Panel D above Panel E.

*Against:* an ICC above 0.30 on either panel, which would restore the paper's original
mechanism claim at the calibrated statistic and make the reliability criterion moot.

**H4 --- normalization does not reduce out-of-sample checkpoint spread much.** Between-checkpoint
SD of the log held-out maximum, raw against divided by the frozen `scale_full`.

> Panel D: reduction of less than 25%. Panel E: no reduction, or an increase.

*Against:* a reduction above 25% on Panel D or any material reduction on Panel E.

**H5 --- honest specificity does not rescue the Panel E verdict.** Using the held-out streams to
set each rule's false-alarm budget --- so specificity is measured on data that set no threshold
--- re-run the uniform-control comparison against the existing frozen changed streams.

> The frozen verdicts hold: the uniform control still fails on Panel D and still succeeds on
> Panel E.

*Against:* either verdict flipping, which would mean the frozen comparisons were decided by the
in-sample specificity budget rather than by the rules.

## Claim boundary

If these hold, the paper may say: the frozen procedure's false-alarm behaviour was evaluated on
independently collected stable streams that entered no calibration set, and the registered
counts were not an artifact of reusing the calibration data.

It may **not** say: that the false-alarm rate is controlled at any nominal level (the guarantee
is a separate question, settled by Proposition 6 and its replacement, not by this collection);
that the result transfers to a deployed checkpoint selected on performance; or that a
confirmation of H2 to H4 is independent evidence, given the disclosure above.

If H1 fails, the registered stable counts do not describe the procedure's behaviour on future
null data, and the specificity claim narrows to the calibration sample. That is reported as a
failure, not reconciled.
