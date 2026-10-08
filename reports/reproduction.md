# Reproducing the public numerical results

Use Python 3.11 and `python -m pip install -r requirements.txt` from the repository root. NumPy 2.3.2, pandas 2.3.1, SciPy 1.16.1 and PyYAML 6.0.2 match the original lock. Python's standard library supplies hashing and exact fractions.

## Commands

```bash
bash scripts/01_example.sh
bash scripts/02_reproduce.sh
```

The scripts use `python` from the activated environment. `PYTHON=/path/to/python` selects another interpreter. An optional first argument selects a fresh output directory, for example `bash scripts/02_reproduce.sh outputs/run-02`. Existing output directories are rejected and retained.

The minimal example copies only the diagnostic engine and example into an isolated worktree. It checks eight points and 216 decisions; full-panel detections are 5/12 for LOCO and 7/12 for scaling-only, with zero stable alarms. No paths are supplied, so delays are unavailable. This is a deterministic software example, not an additional experiment.

## Full run

The runner verifies the 104 declared code/input files against [reproduction_manifest.json](reproduction_manifest.json), copies them into `<output>/worktree/`, and executes unchanged analysis code there. It never relies on the private manuscript, internal documentation, or a reviewer archive.

| Stage | Exact regenerated outputs | Independent checks |
|---|---:|---|
| Original cached-path analysis | 12 CSVs | Membership, calibration, historical regression |
| Matched membership and threshold diagnostics | 28 CSVs | Arithmetic, stopping times, adverse inputs |
| Quantile diagnostics | 19 CSVs | Independent formulas and adverse cases |
| Reusable diagnostic inputs | 8 files | Source identities and exact serialized tables |
| Correction removal and checkpoint omission | 10 CSVs | Independent decisions, membership coverage, worked cases |
| Exact-rational propagation audit | 9 files | Rational fixtures and interpretation checks |
| Effect sizes and finite-input sensitivity | 14 files | 84 controls, 1,584 scenarios, 25,344 fits and 5,968,512 independently replayed decisions |

The numerical total is 69 CSVs (12 + 28 + 19 + 10), plus eight adapter-input files and nine reference files. Every output set must match the expected filenames, byte counts and SHA256 hashes. A mismatch fails the run; the checker does not update its expectations.

`summary.json` is written with `status: PASS` only after all stages finish. `logs/` records subprocess output; intermediate progress is printed to the terminal. Generated CSVs and exact-reference outputs are inside the staged experiment paths under `worktree/`. The run requires roughly 3 GB of free space and takes about 15 minutes on the tested CPU environment.

Some lower-level scripts resolve sibling result paths in their own experiment tree. Use the public runner to create those dependencies in order; afterwards, commands from the [diagnostic interface](../experiments/revision_20260930_r4/DIAGNOSTIC_INTERFACE.md) can be applied from that generated worktree.

## Boundaries

This reproduces processed-input numerical analyses. It does not authenticate upstream data collection, reconstruct raw scores, train policies, fit detectors, retry Panel B recovery, compile LaTeX, compare typeset tables, or regenerate manuscript figures. The public preview and small summary CSVs are included for inspection. Large generated CSVs stay out of Git and are recreated by the runner.

The [verification receipt](verification.json) records the clean-export check. Neither that check nor this code establishes independent new-checkpoint performance or a deployment alarm guarantee.

## Finite-input sensitivity

The frozen [protocol](../experiments/revision_20260930_r4/sensitivity_protocol.json) specifies magnitudes 0, 1e-6, 1e-4, 1e-3, 1e-2 and 0.05. It uses 32 deterministic single-stream vectors and 100 PCG64 vectors with seed 20261006, reused across magnitudes, input scopes, families, controls and full/omitted fits. Factors multiply complete available statistic paths; additional-stable records have only maxima and retain unavailable delays. Divisors, calibration maxima and stable-only selectors are recomputed. No raw scoring, training, simulation or detector fitting is performed.

After the full runner has constructed its processed tables, the standalone commands from that worktree are:

```bash
python experiments/revision_20260930_r4/review_analysis.py \
  --inputs experiments/revision_20260930_r4/diagnostic_inputs \
  --reference experiments/revision_20260930_r4/numerical_reference \
  --protocol experiments/revision_20260930_r4/sensitivity_protocol.json \
  --output outputs/fresh-review
python experiments/revision_20260930_r4/review_check.py \
  --inputs experiments/revision_20260930_r4/diagnostic_inputs \
  --reference experiments/revision_20260930_r4/numerical_reference \
  --protocol experiments/revision_20260930_r4/sensitivity_protocol.json \
  --results outputs/fresh-review --report outputs/fresh-review-check.json
```

`effects.csv` and `effect_intervals.csv.gz` contain exact beta/tau values, threshold changes and strict interval membership for all 84 controls. `vectors.csv.gz` and `factors.csv.gz` retain every shared factor; `divisors.csv.gz` contains formula outputs. `fits.csv` records complete donor and tuning-boundary ties and changes from zero input. `decisions.csv.gz` contains paired primary/tuning records with available delays and signed, absolute and relative crossing margins. `paired.csv` separates correction removal from omission and reports gains, losses and changed delays on identical IDs. `persistence.csv` gives descriptive scenario counts. The zero gate checks every selected historical threshold, donor, boundary, decision and available delay before nonzero scenarios run.

The checker independently replays all scenarios with scalar order statistics, checks all 84 summaries against exact-rational records, and tests strict endpoint equality, within/across-checkpoint ties, unchanged sets, simultaneous gains/losses, fixed-cap deletion monotonicity and invalid inputs. NumPy/pandas versions must match `requirements.txt`; unpinned arithmetic can fail the zero gate.

These are finite-input sensitivity results. Scenario counts are not probabilities, confidence intervals or worst-case guarantees. All scenarios preserve memberships and caps; the omitted fits retain 152 tuning and 76 primary records.
