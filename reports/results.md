# Recorded comparison results

These are retrospective within-panel comparisons. Their primary changed condition is `gradual_updating_down`: gradual gravity reduction while the policy keeps updating. Panel `D` is Hopper; panel `E` is Walker2d. Full-panel primary denominators are 80 streams; frozen and updating stable modes each also contain 80 streams.

## Correction-removal controls

Select the source identifier below in [diagnostic_condition_summary.csv](diagnostic_condition_summary.csv), keep `condition = gradual_updating_down` and an empty `omitted_checkpoint`, and compare `control_type = loco` with `scaling_only`. Read achieved alarms from [diagnostic_operating_points.csv](diagnostic_operating_points.csv) using the same identifiers.

| Source point identifier | LOCO detections | Scaling-only detections | Gains / losses | Frozen / updating alarms, both rules |
|---|---:|---:|---:|---:|
| `D:cut_search:pooled:zero:matched6` | 17/80 | 55/80 | 38 / 0 | 0/80 / 0/80 |
| `D:cut_search:arithmetic:original_caps:matched6` | 73/80 | 72/80 | 0 / 1 | 0/80 / 2/80 |

The first row uses zero caps. The second uses frozen/updating caps 4/2 and matched six-stream arithmetic target divisors. Pooled calibration uses a divisor of one; a membership label does not mean six records were needed to estimate that constant.

For the adverse arithmetic example, restricted mean delay changes from 44.1125 to 44.65 monitoring windows; a miss receives 81. The lost primary stream is checkpoint 703, replicate 2.

Among the 84 full-panel scaling-only controls, 54 rows have both `gained = 0` and `lost = 0` for the primary condition. This means the primary detected sets are unchanged. It does not establish equality of delays, stable decisions, or deployment performance.

## Estimation memberships

[historical_operating_points.csv](historical_operating_points.csv) and [matched_operating_points.csv](matched_operating_points.csv) distinguish original target memberships from matched six-stream target estimation. [quantile_operating_points.csv](quantile_operating_points.csv) records the fixed 0.90-quantile normalizer with its own membership/family identifiers. Compare rows with compatible memberships and budgets.

## Figure preview

The README image is a raster preview of the preserved checkpoint diagnostic figure. Upper panels compare Walker2d arithmetic, geometric and quantile GLR at total cap two with eight-stream estimation. The lower panel shows paired arithmetic-to-quantile cut-search changes at matched six-stream target estimation and caps 4/1. It displays five gains and four losses; 71 unchanged decisions are omitted. The original vector figure remains in the private local manuscript tree.

## Interpretation limits

A cap is an empirical constraint on the selected stable data, not a probability bound. Shared calibration and retrospective selection limit population interpretation. Additional-stable caches provide maxima without indexed paths, so their delay analyses are unavailable. The full public command checks numerical outputs; typeset-table and figure-pixel reproduction belongs to the separate local manuscript workflow.

## Effect size and finite-input sensitivity

All 84 correction-removal controls are summarized by beta and tau with exact-rational threshold comparisons. In the Hopper pooled zero-cap example beta=tau=0.924175 (rounded): outside thresholds fall by 7.5825%, producing 38 primary gains. In the Hopper arithmetic 4/2-cap example beta=0.943710 and tau=1: inside thresholds rise by 5.9648%, with one primary loss. Walker2d arithmetic at total cap two has beta=0.962199 and tau=1; its inside threshold rises by 3.9286% while its primary detected set is unchanged.

For the five prescribed settings and paired full/omitted fits, all primary detected sets persist through magnitude 1e-4 in the 1,584 tested scenarios, although some tuning boundaries change at 1e-4. Larger perturbations change decisions. At magnitude 0.05, Hopper pooled correction-removal gains range from 0 to 44 with stable-only input changes and 0 to 39 with primary paths also perturbed, without losses. Arithmetic gains range from 0 to 2 and losses from 0 to 1 or 0 to 3, respectively. Walker2d quantile omission preserves its primary detected set in every tested scenario; available delays need not remain equal. These observations are finite-input sensitivity, not probabilities, confidence intervals, worst-case bounds or deployment guarantees. See [the protocol and reproduction instructions](reproduction.md#finite-input-sensitivity).
