# Minimal calibration diagnostic example

This deterministic example needs only `diagnostic.py`, this directory, and an installed Python environment with NumPy and pandas (the study archive includes `pyproject.toml` and `uv.lock`). It does not use study data, the study adapter, or simulation software. Run from the extracted evidence root with a compatible `python` on PATH:

```bash
example_run=$(mktemp -d)
cp experiments/revision_20260930_r4/diagnostic.py "$example_run/diagnostic.py"
cp -R experiments/revision_20260930_r4/examples/minimal "$example_run/minimal"
cd "$example_run"
python diagnostic.py --manifest minimal/manifest.json --output results
python minimal/check_example.py --results results --report checked.json
```

The engine requires a fresh results directory. The checker requires a new report path. Its success output is `status: PASS`. `expected_summary.json` lists all eight operating points; full-panel LOCO detects 5/12 changed streams and scaling-only detects 7/12, with zero stable alarms for both.

| Omitted checkpoint | LOCO detections | Scaling-only detections |
|---|---:|---:|
| None | 5/12 | 7/12 |
| 11 | 3/8 | 4/8 |
| 31 | 4/8 | 5/8 |
| 71 | 6/8 | 6/8 |

The schema-1 manifest names six input CSVs. Each of three checkpoints has four replicates, with frozen/updating stable maxima in consecutive pairs:

- Checkpoint 11: `[1,2,3,4,5,6,7,8]`.
- Checkpoint 31: `[2,3,4,5,6,7,8,10]`.
- Checkpoint 71: `[1,2,3,4,6,8,10,14]`.

At each checkpoint the four changed maxima are `[12,18,24,30]`. Full-membership calibration divisors are arithmetic means of all eight local stable maxima. Target divisors are arithmetic means of the six remaining maxima after excluding the target replicate in both stable modes. `calibration.csv` and `targets.csv` declare these separate divisor identities. The means are supplied as binary64 values in `divisors.csv`; `memberships.csv` records their exact members.

One retuned point uses zero alarms in each stable mode. It requests its scaling-only control and both rules after each checkpoint omission, for eight output operating points. The omitted checkpoint is removed from calibration, tuning and evaluation; each retained local divisor keeps its original members. Each omission compares with its own full-panel rule on identical retained streams. The full-panel control compares with full-panel LOCO.

No indexed paths are supplied. Detection is determined by strict `maximum > threshold`; all delays and first-crossing times are unavailable, including misses. `horizon: 4` and `miss_delay: 5` declare conventions for optional paths and do not authorize inventing stopping times from maxima.

`check_example.py` uses only the Python standard library and imports no diagnostic code. It recomputes arithmetic means from the declared members, all eight stable-only fits with strict binary64 products, 216 thresholds and decisions, retained-stream comparisons, omission memberships and unavailable delays. This formula-verifying example is separate from the internal arbitrary-supplied-divisor fixture in `diagnostic_checks.py`, which tests the engine's provenance boundary. Neither example is scientific experimental evidence.
