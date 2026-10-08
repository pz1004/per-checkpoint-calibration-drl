"""Adverse-case tests for the fixed quantile sensitivity; originals stay untouched."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

import quantile_analysis as production
import quantile_verify as independent


def run(results: Path, output: Path) -> None:
    independent.require(not output.exists(), "fresh probe report")
    probes = []
    examples = [
        ([1., 2., 3., 4.], 3.7),
        ([1., 2., 3., 4., 5., 6.], 5.5),
        ([1., 2., 3., 4., 5., 6., 7., 8.], 7.3),
        ([2.] * 8, 2.),
        ([1.] * 7 + [1e12], .7 + .3e12),
    ]
    for values, expected in examples:
        independent.close(production.quantile(np.array(values)), expected, "known quantile fixture")
        independent.close(independent.percentile(values), expected, "independent fixture")
    probes.append(dict(case="four/six/eight interpolation, ties and extreme", result="PASS", fixtures=len(examples)))
    for value, label in ((0., "zero"), (-1., "negative"), (np.nan, "NaN"), (np.inf, "infinity")):
        values = np.arange(1., 9.)
        values[0] = value
        try:
            production.quantile(values)
        except ValueError as error:
            independent.require("positive finite domain" in str(error), "explicit domain failure")
            probes.append(dict(case=label + " scale input", result="REJECTED", error=str(error)))
        else:
            raise ValueError("invalid scale input silently accepted")
    f, _, _ = production.r4.load()
    try:
        production.fit(f)
    except ValueError as error:
        probes.append(dict(case="changed rows passed to fit", result="REJECTED", error=str(error)))
    else:
        raise ValueError("fit admitted changed streams")
    altered = f.copy(deep=True)
    for stat in ("GLR", "cut_search"):
        altered[stat] = [path if stable else np.full_like(path, 1e12)
                         for path, stable in zip(altered[stat], altered.stable, strict=True)]
    first = production.fit(f[f.stable])
    second = production.fit(altered[altered.stable])
    for i in range(3):
        independent.require(first[i].equals(second[i]), "changed outcomes altered fitted records")
    independent.require(first[3] == second[3], "changed outcomes altered foreign extremes")
    points, stops, _ = production.build_points(altered, second[0], second[3])
    before_points = independent.read(results / "operating_points.csv").set_index("point_id")
    before_stops = independent.read(results / "stopping_times.csv").set_index(["point_id", "stream_id"])
    after_stops = stops.set_index(["point_id", "stream_id"])
    for p in points.itertuples():
        for field in ("multiplier", "frozen_alarms", "updating_alarms"):
            independent.require(getattr(p, field) == before_points.loc[p.point_id, field], "selection invariance")
    for field in ("threshold", "base_threshold", "divisor_id"):
        independent.require(before_stops[field].equals(after_stops[field]), "threshold invariance")
    independent.require(points.primary_detections.eq(80).all(), "perturbation changed outcomes")
    probes.append(dict(case="changed outcomes replaced by 1e12", result="PASS", points=24,
                       thresholds=11520, primary_detections="80 at every point", stable_selection="unchanged"))
    for name, field, value in [
        ("stopping_times.csv", "divisor_id", "wrong_membership"),
        ("foreign_extreme_donors.csv", "foreign_stream_id", "wrong_donor"),
        ("threshold_allocation.csv", "decomposed_ratio", -99.),
        ("operating_points.csv", "source_point_id", "missing_anchor"),
    ]:
        with tempfile.TemporaryDirectory(prefix="r4-quantile-corruption-") as temp:
            temp = Path(temp)
            for p in results.glob("*.csv"):
                (temp / p.name).symlink_to(p.resolve())
            changed = independent.read(results / name)
            changed.loc[0, field] = value
            (temp / name).unlink()
            production.r4.write(changed, temp / name)
            try:
                independent.audit(temp, skip_cuts=True)
            except ValueError as error:
                probes.append(dict(case=name + ":" + field, result="REJECTED", error=str(error)))
            else:
                raise ValueError("independent audit admitted corruption")
    output.write_text(json.dumps(dict(status="PASS", probes=probes), indent=2) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.results, args.output)
