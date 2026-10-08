# Calibration diagnostic interface

`diagnostic.py` accepts tables and supplied divisor values; it imports neither RL loaders nor historical analysis code. It validates positive finite domains, declared memberships and complete calibration coverage. It does not authenticate how an arbitrary caller obtained a supplied divisor. `study_manifest.py` constructs the study's named divisor families, and `diagnostic_verify.py` independently recomputes their formulas from the declared members. A valid membership table alone does not prove a supplied value is an arithmetic mean, geometric mean or quantile; pooled divisors retain the explicit value-one check.

```bash
python experiments/revision_20260930_r4/study_manifest.py --output /fresh/inputs
python experiments/revision_20260930_r4/diagnostic.py --manifest /fresh/inputs/manifest.json --output /fresh/results
python experiments/revision_20260930_r4/diagnostic_verify.py --manifest /fresh/inputs/manifest.json --results /fresh/results --output /fresh/audit.json
python experiments/revision_20260930_r4/diagnostic_checks.py --manifest /fresh/inputs/manifest.json --output /fresh/checks.json
```

All output directories must be fresh. The two verification programs additionally encode the case-study acceptance counts; the diagnostic engine accepts other valid table sizes. Python, NumPy and pandas are required. The evidence archive supplies the environment lock, not a complete offline wheelhouse.

## Manifest schema version 1

JSON contains `schema_version: 1`, a positive integer `horizon`, optional `miss_delay` (must equal horizon + 1), `tables` mapping the six names below to relative CSV filenames, and optional `indexed_paths` pointing to a CSV. Paths are resolved relative to the manifest. CSV numeric fields are binary64, read with round-trip conversion. Boolean fields use True/False.

| Table | Required columns and roles |
|---|---|
| streams | stream_key (unique statistic-specific identity), stream_id, panel, statistic, checkpoint_seed, replicate, mode (frozen/updating), condition, role (original_stable/additional_stable/changed), maximum |
| divisors | divisor_id, panel, statistic, calibration, checkpoint_seed, excluded_replicate (-1 for full membership), divisor, membership_count |
| memberships | divisor_id, stream_key; all checkpoint-local original stable members, excluding the specified replicate across modes where applicable; pooled divisor one also declares memberships |
| calibration | panel, statistic, calibration, stream_key, divisor_id; exactly every declared original-stable stream for the point's panel and statistic, with one full-membership divisor identity per checkpoint within each calibration group; foreign membership is reconstructed by excluding the retained target checkpoint |
| targets | source_point_id, stream_key, divisor_id, tuning; optional paired historical_threshold and historical_fired trigger exact preservation checks; historical_delay is supplied by this adapter for independent verification |
| points | source_point_id, panel, statistic, calibration, family, budget (native means fixed multiplier), target_budget_policy, tuning_role, caps_json, multiplier, include_control, include_influence |

Policies: `matched_six` excludes the target replicate in both modes; `additional_eight` uses full original-stable membership; `historical_six_stable_eight_changed` excludes the replicate only for original-stable targets. The engine validates membership against the actual streams and declared membership_count; six/eight names describe the study policies, not an assumption of independent information. Native requires a supplied positive multiplier. For a new retuned point, multiplier may be empty (or the column omitted); a supplied value is checked against stable-only selection. caps_json is either `{"frozen":4,"updating":2}` or `{"total":2}`. include_control adds one scaling-only point. include_influence adds each checkpoint omission for both rules and requires include_control. Native controls/deletions are excluded. Omitted checkpoints disappear from calibration, tuning and evaluation; retained local divisor memberships stay fixed.

Indexed paths contain stream_key,index,value with one row for every one-based index through horizon for each available stream. Empty values explicitly mark abstentions; their indices remain in the delay clock. Infinity and invalid text are rejected. A missing stream path means detection from the maximum only and unavailable delay. A supplied path must reproduce its maximum exactly. In this study all changed paths exist; additional stable paths do not.

Every operating point must have a calibration group. Before creating the output directory, the engine rejects missing streams, missing checkpoints, absent groups and inconsistent checkpoint divisor identities with the affected `(panel, statistic, calibration)` group in the error. Expected coverage comes from `streams`, not a fixed study checkpoint count. Calibration ratios must be positive and finite; each retained checkpoint ratio is also checked immediately before export. Schema version 1 and all CSV columns are unchanged.

## Calculations and numerical semantics

For each family, the `calibration` table identifies each checkpoint's full-membership divisor `d_c` (`excluded_replicate=-1`). Use `M_cj/d_c` to compute checkpoint ratios R, global B, all global donor ties, every foreign maximum B_minus, all foreign donor ties, and b=B_minus/B. Separately, the `targets` table identifies `d_c_r` according to the operating point's target membership policy. Calibration and target divisor IDs have different roles even when their values coincide (for example pooled scaling). In the matched study design these memberships are eight and six, respectively. Historical LOCO base is `(B_minus * d_c_r)`, threshold is `base * multiplier`, and theta is multiplier*B. Scaling-only base is the target divisor d_c_r; its fitted multiplier is theta. The output also evaluates `(theta*d)*b` and flags floating-point decision differences. That expression is an exact-arithmetic identity; it never silently replaces accepted products.

The stable selector takes descending ratio position cap+1 within each constrained mode (or over all stable records for a total cap), takes the largest constrained cutoff, and advances upward in binary64 until actual strict-product crossings meet every cap. Ranks have min/max tie positions. Each mode's candidate at its declared cap (or the total cap index for diagnostic comparison) is exported; under a total cap those per-mode candidates are explicitly not extra constraints. Invalid/nonpositive/nonfinite maxima, divisors, bases or ratios, duplicate identities, unknown members, changed-data fitting/tuning and infeasible caps are rejected.

## Outputs and comparators

Every table carries point/source IDs, control type, omitted checkpoint (blank for full-panel), target-budget policy and divisor identity; stream/checkpoint/divisor IDs appear wherever applicable.

- operating_points: selected multiplier, theta/B, achieved F/U counts and denominators, adjustment steps, numerical-coordinate discrepancies.
- checkpoint_parameters and donor_records: R/B/B_minus/b, foreign counts, all tied global/foreign streams and number of distinct donor checkpoints.
- tuning_streams and tuning_constraints: ratios, rank intervals, tied boundary IDs, per-mode candidate cutoffs, binding constraints, selected values and achieved counts.
- calibration_records: retained original-stable members and ratios; target-specific foreign sets exclude that target checkpoint.
- stream_decisions: effective thresholds, maxima, strict detections, delay availability, indexed delays, reference thresholds/outcomes, and paired gains/losses.
- condition_summary: all stable modes and all four changed conditions at the same fit; counts, denominators, checkpoint coverage, paired IDs and restricted mean delays.
- calibration_comparisons and calibration_gain_loss_streams: arithmetic-referenced family comparisons at identical settings, for each rule type.

Full-panel controls compare with their source LOCO rule. Each deletion compares with its own full-panel rule restricted to exactly the same retained stream IDs. Original rows compare with themselves. The 440 scenarios retain integer caps and have 76 stable streams per mode, 76 streams per changed condition and 144 foreign calibration records per target. These are finite-panel influence calculations, not fixed-rate comparisons, bootstrap samples or confidence intervals. Deterministic test fixtures are software checks, not simulator runs.

## Standalone worked example

`examples/minimal/README.md` gives commands that copy only `diagnostic.py` and that example to a fresh directory. Its schema-1 manifest and six CSVs declare three checkpoints, eight original-stable members per checkpoint, six replicate-excluded target members, arithmetic means, zero per-mode caps, a scaling-only control and all three checkpoint omissions for both rules. `check_example.py` independently recomputes memberships, arithmetic formulas, selection, thresholds and all 216 decisions using only the Python standard library. Expected full-panel counts are 5/12 versus 7/12 with zero stable alarms. All eight points have unavailable delays because no indexed paths are supplied. The separate arbitrary supplied-divisor fixture in `diagnostic_checks.py` remains a test of the engine's provenance boundary.
