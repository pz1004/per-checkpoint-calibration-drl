"""Retrospective calibration comparisons from immutable score caches.

No detector fitting, raw-data recovery, simulation or changed-outcome threshold selection.
Outputs require a fresh directory. See PROTOCOL.md for the estimands and limits.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rl_hanbat.socis.sequential import adwin_evidence, glr_trace, memoryless_trace

PANELS = {"D": ("confirmation_panel", "panel_d_baselines", range(700, 720), (4, 2)),
          "E": ("walker_panel", "panel_e_baselines", range(800, 820), (4, 1))}
MODES = ("frozen", "updating")
PRIMARY = "gradual_updating_down"
CALIBRATIONS = ("pooled", "arithmetic", "geometric")
CAPS = (0, 1, 2, 3, 5, 8)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def load_panel(panel: str) -> tuple[pd.DataFrame, list[Path], dict, list[dict]]:
    """Rebuild paths while checking normalization, membership and abstentions."""
    directory, baseline, seeds, _ = PANELS[panel]
    rows, inputs, windows = [], [], []
    missing = 0
    for seed in seeds:
        file = ROOT / "experiments" / baseline / "scores" / f"seed_{seed}.json"
        inputs.append(file)
        for r in json.loads(file.read_text()):
            scores = r["scores"]
            assert len(scores) == 120 and r["checkpoint_seed"] == seed
            warm = np.array([x for x in scores[:40] if x is not None], dtype=float)
            assert len(warm) >= 32
            center = float(np.median(warm))
            scale = 1.4826 * float(np.median(np.abs(warm-center)))
            assert scale >= 1e-12
            assert center == r["center"] and scale == r["scale"]
            z = [None if x is None else (float(x)-center)/scale for x in scores]
            paths = {"GLR": glr_trace(z[40:], float("inf")).statistic,
                     "cut_search": adwin_evidence(z, delta=.1, minimum_subwindow=5)[40:]}
            assert abs(float(paths["GLR"].max()) - r["max_statistic"]) < 1e-9
            missing += sum(x is None for x in scores)
            for statistic, path in paths.items():
                assert len(path) == 80 and np.isfinite(path).all() and path.max() > 0
                for t, x in enumerate(z[40:]):
                    if x is None and t:
                        assert path[t] == path[t-1]
            rows.append({k: r[k] for k in ("checkpoint_seed", "replicate", "condition", "mode", "stream_id")} |
                        {"panel": panel, "center": center, "scale": scale,
                         "warmup_eligible": len(warm), "monitoring_eligible": sum(x is not None for x in z[40:]),
                         "stable": r["condition"].startswith("stable"), "source": str(file.relative_to(ROOT)),
                         "GLR": paths["GLR"], "cut_search": paths["cut_search"],
                         "memoryless": memoryless_trace(z[40:], float("inf")).statistic,
                         "cut_80": adwin_evidence(z[40:]), "watch": np.asarray(r["watch"])[40:]})
            for t, (raw, standardized) in enumerate(zip(scores, z, strict=True), 1):
                windows.append({"panel": panel, "stream_id": r["stream_id"], "window": t,
                                "eligible": raw is not None, "raw_score": raw, "z": standardized,
                                "phase": "warmup" if t <= 40 else "monitoring",
                                **{stat: None if t <= 40 else float(path[t-41]) for stat, path in paths.items()}})
    f = pd.DataFrame(rows)
    assert len(f) == 480 and f.stream_id.is_unique
    assert f.groupby(["checkpoint_seed", "condition"]).size().eq(4).all()
    assert missing == (0 if panel == "D" else 37)
    historical = ROOT / "experiments" / directory / "stage1_streams.csv"
    inputs.append(historical)
    original = pd.read_csv(historical, float_precision="round_trip").set_index("stream_id")
    stable = f[f.stable].set_index("stream_id")
    gap = max(abs(float(r.GLR.max()) - original.loc[sid, "max_statistic"]) for sid, r in stable.iterrows())
    assert gap < 1e-9
    return f, inputs, {"streams": len(f), "abstentions": missing, "stage1_maximum_gap": gap}, windows


def fit(stable: pd.DataFrame, statistic: str, calibration: str) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Eight-stream foreign scales; full/eight or replicate-excluded/six targets."""
    assert stable.stable.all()  # Interface forbids changed records in fitting.
    full, excluded, divisors, membership, assignments = {}, {}, [], [], []
    reduce = {"pooled": lambda x: 1., "arithmetic": lambda x: float(np.mean(x)),
              "geometric": lambda x: float(np.exp(np.mean(np.log(x))))}[calibration]
    for seed, group in stable.groupby("checkpoint_seed", sort=True):
        for rep in [-1, 0, 1, 2, 3]:
            subset = group if rep == -1 else group[group.replicate != rep]
            values = np.array([p.max() for p in subset[statistic]], dtype=float)
            divisor = reduce(values)
            assert divisor > 0 and len(values) == (8 if rep == -1 else 6)
            if rep == -1:
                full[seed] = divisor
            else:
                excluded[seed, rep] = divisor
            divisor_id = f'{group.panel.iloc[0]}:{statistic}:{calibration}:{seed}:exclude_{rep}'
            divisors.append({"divisor_id": divisor_id, "checkpoint_seed": seed, "excluded_replicate": rep,
                             "divisor": divisor, "estimation_streams": 0 if calibration == "pooled" else len(values)})
            if calibration != "pooled":
                membership.extend({"divisor_id": divisor_id, "stream_id": r.stream_id, "stable_maximum": float(getattr(r, statistic).max())}
                                  for r in subset.itertuples())
    thresholds = {}
    for seed in full:
        foreign = stable[stable.checkpoint_seed != seed]
        normalized = []
        for r in foreign.itertuples():
            maximum = float(getattr(r, statistic).max())
            ratio = maximum / full[r.checkpoint_seed]
            normalized.append(ratio)
            assignments.append({"target_checkpoint": seed, "foreign_checkpoint": r.checkpoint_seed,
                                "foreign_stream_id": r.stream_id, "maximum": maximum,
                                "foreign_divisor": full[r.checkpoint_seed], "normalized_maximum": ratio})
        assert len(normalized) == 152
        foreign_level = max(normalized)
        thresholds[seed] = {"foreign_level": foreign_level, "full_divisor": full[seed],
                            "changed": foreign_level * full[seed],
                            **{f"stable_{r}": foreign_level * excluded[seed, r] for r in range(4)}}
    return thresholds, divisors, membership, assignments


def thresholds_for(f: pd.DataFrame, table: dict, independent_stable: bool = False) -> np.ndarray:
    return np.array([table[r.checkpoint_seed][f"stable_{r.replicate}" if r.stable and not independent_stable else "changed"]
                     for r in f.itertuples()], dtype=float)


def choose(stable_maxima: np.ndarray, base: np.ndarray, modes: np.ndarray,
           caps: tuple[int, int] | int) -> dict:
    """Select only from stable records and enforce actual multiplication semantics.

    The ratio order statistic is a candidate, not an assumed attainable count.
    Advance by representable floats if rounded base*lambda violates the cap.
    """
    ratio = stable_maxima / base
    if isinstance(caps, int):
        candidate = float(np.sort(ratio)[::-1][caps])
    else:
        candidate = max(float(np.sort(ratio[modes == mode])[::-1][cap]) for mode, cap in zip(MODES, caps, strict=True))

    def counts(multiplier: float) -> list[int]:
        fired = stable_maxima > base * multiplier
        return [int(fired[modes == mode].sum()) for mode in MODES]

    def feasible(count: list[int]) -> bool:
        return sum(count) <= caps if isinstance(caps, int) else all(a <= b for a, b in zip(count, caps, strict=True))

    initial = counts(candidate)
    multiplier, steps = candidate, 0
    while not feasible(counts(multiplier)):
        multiplier = float(np.nextafter(multiplier, np.inf))
        steps += 1
        assert steps <= 32
    return {"multiplier": multiplier, "ratio_candidate": candidate, "boundary_ulp_increments": steps,
            "candidate_frozen_alarms": initial[0], "candidate_updating_alarms": initial[1],
            "ratio_ties_at_candidate": int((ratio == candidate).sum()),
            "actual_boundary_equalities": int((stable_maxima == base*multiplier).sum())}


def evaluate(f: pd.DataFrame, statistic: str, base: np.ndarray, point: dict) -> tuple[dict, list[dict], list[dict]]:
    rows = []
    thresholds = base * point["multiplier"]
    for r, threshold, original_base in zip(f.itertuples(), thresholds, base, strict=True):
        path = getattr(r, statistic)
        if path is None:  # Additional stable caches contain only GLR maxima.
            fired, delay = bool(r.max_statistic > threshold), None
        else:
            crossing = np.flatnonzero(path > threshold)
            fired = bool(len(crossing))
            delay = int(crossing[0])+1 if fired else 81
        rows.append({"point_id": point["point_id"], "panel": r.panel, "stream_id": r.stream_id,
                     "checkpoint_seed": r.checkpoint_seed, "condition": r.condition, "mode": r.mode,
                     "replicate": r.replicate, "stable": r.stable, "fired": fired,
                     "first_crossing": delay if fired else None, "restricted_delay": delay,
                     "stopping_time_available": path is not None, "threshold": threshold,
                     "base_threshold": original_base,
                     "maximum": float(path.max()) if path is not None else r.max_statistic})
    s = pd.DataFrame(rows)

    def summarize(g: pd.DataFrame) -> dict:
        primary = g[g.condition == PRIMARY]
        changed = g[~g.stable]
        result = {"primary_detections": int(primary.fired.sum()), "primary_denominator": len(primary),
                  "primary_mean_delay": float(primary.restricted_delay.mean()),
                  "primary_delay_sum": float(primary.restricted_delay.sum()),
                  "changed_detections": int(changed.fired.sum()), "changed_denominator": len(changed)}
        for mode in MODES:
            selected = g[g.stable & g["mode"].eq(mode)]
            result[f"{mode}_alarms"] = int(selected.fired.sum())
            result[f"{mode}_denominator"] = len(selected)
        result["actual_total_alarms"] = result["frozen_alarms"] + result["updating_alarms"]
        return result

    checkpoints = [{"point_id": point["point_id"], "panel": point["panel"], "checkpoint_seed": seed, **summarize(g)}
                   for seed, g in s.groupby("checkpoint_seed")]
    return {**point, **summarize(s)}, rows, checkpoints


def dispersion(stable: pd.DataFrame, held: pd.DataFrame) -> list[dict]:
    rows = []
    for statistic in ("GLR", "cut_search"):
        for chosen in itertools.combinations(range(4), 2):
            est = stable[stable.replicate.isin(chosen)]
            test = stable[~stable.replicate.isin(chosen)]
            assert not set(est.stream_id) & set(test.stream_id)
            rows.extend(dispersion_cell(est, test, statistic, ','.join(map(str, chosen)), 4, 4))
    rows.extend(dispersion_cell(stable, held, "GLR", "all_original_to_additional", 8, 8))
    return rows


def dispersion_cell(est: pd.DataFrame, test: pd.DataFrame, statistic: str, allocation: str,
                    budget: int, test_budget: int) -> list[dict]:
    rows = []
    e = est.assign(maximum=[float(p.max()) for p in est[statistic]])
    maxima = [float(p.max()) if p is not None else float(x) for p, x in zip(test[statistic], test.get("max_statistic", [0]*len(test)), strict=True)]
    t = test.assign(logmax=np.log(maxima))
    for summary, reducer in (("mean_log_maximum", lambda g: g.mean()), ("q90_log_maximum", lambda g: g.quantile(.9)),
                             ("max_log_maximum", lambda g: g.max())):
        target = reducer(t.groupby("checkpoint_seed").logmax)
        raw = float(target.std(ddof=1))
        for calibration in ("arithmetic", "geometric"):
            scale = e.groupby("checkpoint_seed").maximum.apply(lambda x: x.mean() if calibration == "arithmetic" else np.exp(np.log(x).mean()))
            normalized = float((target - np.log(scale)).std(ddof=1))
            rows.append({"panel": est.panel.iloc[0], "statistic": statistic, "calibration": calibration,
                         "allocation": allocation, "estimation_streams_per_checkpoint": budget,
                         "evaluation_streams_per_checkpoint": test_budget, "summary": summary,
                         "checkpoints": len(target), "raw_sd": raw, "normalized_sd": normalized, "change": normalized/raw-1,
                         "uncertainty": "descriptive; no population interval; overlapping allocations"})
    return rows


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    inputs = [Path(__file__), Path(__file__).with_name("PROTOCOL.md"), ROOT/"src/rl_hanbat/socis/sequential.py", ROOT/"experiments/geometric_divisor/PREDICTION.md", ROOT/"experiments/stable_holdout/PREDICTION.md"]
    collections = {k: [] for k in ("streams", "windows", "maxima", "divisors", "divisor_membership", "calibration_assignments", "thresholds",
                                   "operating_points", "stopping_times", "checkpoint_results", "dispersion", "baseline_regression")}
    regression = {}
    for panel, (directory, baseline, _, cap) in PANELS.items():
        print(f"Reconstructing panel {panel}", flush=True)
        f, source, checks, windows = load_panel(panel)
        inputs += source
        collections["windows"] += windows
        collections["streams"] += f.drop(columns=["GLR", "cut_search", "memoryless", "cut_80", "watch"]).to_dict("records")
        stable = f[f.stable]
        held_file = ROOT/f"experiments/stable_holdout/panel_{panel.lower()}/holdout_streams.csv"
        inputs.append(held_file)
        held = pd.read_csv(held_file, float_precision="round_trip").assign(panel=panel, stable=True, GLR=None)
        held["condition"] = "stable_" + held["mode"]
        assert len(held) == 160 and not set(held.stream_id) & set(f.stream_id)
        collections["streams"] += held.assign(source=str(held_file.relative_to(ROOT))).drop(columns=["GLR"]).to_dict("records")
        collections["dispersion"] += dispersion(stable, held)
        historical_file = ROOT/f"experiments/{directory}/stage2_streams.csv"
        frozen_file = ROOT/f"experiments/{directory}/thresholds.json"
        inputs += [historical_file, frozen_file]
        historical = pd.read_csv(historical_file, float_precision="round_trip").set_index(["checkpoint_seed", "replicate", "condition"])
        frozen = json.loads(frozen_file.read_text())["thresholds"]
        for statistic in ("GLR", "cut_search"):
            collections["maxima"] += [{"panel": panel, "statistic": statistic, "stream_id": r.stream_id,
                                        "maximum": float(getattr(r, statistic).max()), "stable": r.stable,
                                        "checkpoint_seed": r.checkpoint_seed} for r in f.itertuples()]
            for calibration in CALIBRATIONS:
                meta = {"panel": panel, "statistic": statistic, "calibration": calibration}
                table, divisors, members, assignments = fit(stable, statistic, calibration)
                for name, records in (("divisors", divisors), ("divisor_membership", members), ("calibration_assignments", assignments)):
                    collections[name] += [{**meta, **r} for r in records]
                base = thresholds_for(f, table)
                collections["thresholds"] += [{**meta, "stream_id": r.stream_id, "checkpoint_seed": r.checkpoint_seed,
                                               "base_threshold": b, "target_estimation_streams": (6 if r.stable else 8) if calibration != "pooled" else 0,
                                               "foreign_calibration_streams": 152, **table[r.checkpoint_seed]}
                                              for r, b in zip(f.itertuples(), base, strict=True)]
                if statistic == "GLR" and calibration in ("arithmetic", "pooled"):
                    field = "scale_normalized_threshold_changed" if calibration == "arithmetic" else "pooled_loco_threshold"
                    assert max(abs(table[s]["changed"]-frozen[str(s)][field]) for s in table) < 1e-9
                for name, caps in [("native", None), ("original_caps", cap), ("floor_halved", tuple(c//2 for c in cap)), ("zero", (0, 0))]:
                    point = {**meta, "family": "original_cache", "budget": name,
                             "point_id": f'{panel}:{statistic}:{calibration}:{name}',
                             "frozen_cap": None if caps is None else caps[0], "updating_cap": None if caps is None else caps[1],
                             "selection": ("historical_frozen_rule_reconstruction" if statistic == "GLR" and calibration in ("pooled", "arithmetic") else "retrospective_unscaled_rule") if caps is None else "stable_original_only",
                             **({"multiplier": 1.} if caps is None else choose(np.array([p.max() for p in stable[statistic]]), base[f.stable], stable["mode"].to_numpy(), caps))}
                    result, stopping, checkpoint = evaluate(f, statistic, base, point)
                    collections["operating_points"].append(result)
                    collections["stopping_times"] += stopping
                    collections["checkpoint_results"] += checkpoint
                    if name == "native" and statistic == "GLR" and calibration in ("pooled", "arithmetic"):
                        prefix = "normalized" if calibration == "arithmetic" else "pooled"
                        for row in stopping:
                            old = historical.loc[(row["checkpoint_seed"], row["replicate"], row["condition"])]
                            assert row["fired"] == bool(old[f"{prefix}_fired"])
                            assert row["restricted_delay"] == old[f"{prefix}_delay"]
                        assert result["primary_detections"] == {("D", "pooled"): 50, ("D", "arithmetic"): 68, ("E", "pooled"): 0, ("E", "arithmetic"): 4}[panel, calibration]
                if statistic == "GLR" and calibration != "pooled":
                    frontier = pd.concat([held, f[~f.stable]], ignore_index=True)
                    frontier_base = thresholds_for(frontier, table, independent_stable=True)
                    for total_cap in (None, *CAPS):
                        point = {**meta, "family": "additional_stable", "budget": "native" if total_cap is None else f"total_{total_cap}",
                                 "point_id": f'{panel}:GLR:{calibration}:additional_{total_cap}', "total_cap": total_cap,
                                 "selection": ("native_held_out_evaluation" if calibration == "arithmetic" else "retrospective_unscaled_reuse") if total_cap is None else "additional_stable_tuning_only",
                                 **({"multiplier": 1.} if total_cap is None else choose(held.max_statistic.to_numpy(), frontier_base[:len(held)], held["mode"].to_numpy(), total_cap))}
                        result, stopping, checkpoint = evaluate(frontier, "GLR", frontier_base, point)
                        collections["operating_points"].append(result)
                        collections["stopping_times"] += stopping
                        collections["checkpoint_results"] += checkpoint
        # The historical global-level cut search is distinct from new pooled LOCO cut search.
        sweep_file = ROOT/f"experiments/{baseline}/sweep.csv"
        inputs.append(sweep_file)
        prior = pd.read_csv(sweep_file).set_index("method")
        for method, statistic in (("memoryless", "memoryless"), ("adwin", "cut_search"), ("adwin_monitoring_only", "cut_80"), ("watch", "watch")):
            selected = choose(np.array([p.max() for p in stable[statistic]]), np.ones(len(stable)), stable["mode"].to_numpy(), cap)
            point = {"panel": panel, "point_id": f'{panel}:historical:{method}', **selected}
            result, _, _ = evaluate(f, statistic, np.ones(len(f)), point)
            old = prior.loc[method]
            for new, original in (("primary_detections", "primary_detections"), ("primary_mean_delay", "primary_mean_delay"), ("frozen_alarms", "false_alarms_frozen"), ("updating_alarms", "false_alarms_updating")):
                assert np.isclose(result[new], old[original], atol=1e-10, rtol=0), (panel, method, new)
            collections["baseline_regression"].append({"panel": panel, "historical_method": method, **result})
        regression[panel] = checks | {"all_historical_GL R_stopping_times".replace(" ", ""): "matched", "historical_baselines": "matched"}
    for name, records in collections.items():
        write(pd.DataFrame(records), out/f"{name}.csv")
    outputs = {p.name: sha(p) for p in sorted(out.glob("*.csv"))}
    manifest = {"revision": "revision_20260929_r3", "analysis": "retrospective descriptive", "status": "complete_for_declared_D_E_scope",
                "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(inputs))}, "outputs": outputs,
                "regression": regression, "exclusions": ["Panel B local parity blocked; no retry", "additional stable cut-search evaluation unavailable",
                "additional stable stopping times unavailable from maxima-only caches", "independent evaluation of selected retuned families absent",
                "no validated inferential significance level; no new simulation"]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({"output": str(out), "rows": {k: len(v) for k,v in collections.items()}, "regression": regression}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
