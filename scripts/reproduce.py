"""Run the unchanged numerical analyses without private manuscript dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R3 = "experiments/revision_20260929_r3"
R4 = "experiments/revision_20260930_r4"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Expected a relative file path: {relative}")
    return path


def reproduce(mode: str, output: Path) -> dict:
    manifest = json.loads((ROOT / "reports/reproduction_manifest.json").read_text())
    inputs = manifest["inputs"]
    if mode == "minimal":
        inputs = {
            name: spec
            for name, spec in inputs.items()
            if name == f"{R4}/diagnostic.py" or f"{R4}/examples/minimal/" in name
        }
    for name, spec in inputs.items():
        source = within(ROOT, name)
        if source.stat().st_size != spec["bytes"] or sha(source) != spec["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")

    output.mkdir(parents=True, exist_ok=False)
    work = output / "worktree"
    work.mkdir()
    logs = output / "logs"
    logs.mkdir()
    for name in inputs:
        target = within(work, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    steps = []
    comparisons = {}

    def run(script: str, *arguments: str | Path) -> None:
        print(f"[{len(steps) + 1}] {script}", flush=True)
        command = [sys.executable, str(work / script), *map(str, arguments)]
        result = subprocess.run(
            command,
            cwd=work,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        log = logs / f"{len(steps) + 1:02d}_{Path(script).stem}.log"
        log.write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{script} failed; see {log}\n{result.stderr[-2000:]}")
        steps.append({"script": script, "log": str(log.relative_to(output))})

    def compare(group: str) -> None:
        expected = manifest["outputs"][group]
        folder = work / expected["directory"]
        actual = {
            p.name
            for p in folder.iterdir()
            if p.is_file()
            and (
                group in {"diagnostic_inputs", "numerical_reference", "review_analysis"}
                or p.suffix == ".csv"
            )
        }
        if actual != set(expected["files"]):
            raise ValueError(f"Unexpected output membership: {group}")
        for name, spec in expected["files"].items():
            path = folder / name
            if path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]:
                raise ValueError(f"Output differs from frozen evidence: {group}/{name}")
        comparisons[group] = len(actual)
        print(f"Verified {len(actual)} exact outputs: {group}", flush=True)

    if mode == "minimal":
        run(
            f"{R4}/diagnostic.py",
            "--manifest",
            work / R4 / "examples/minimal/manifest.json",
            "--output",
            output / "results",
        )
        run(
            f"{R4}/examples/minimal/check_example.py",
            "--results",
            output / "results",
            "--report",
            output / "checked.json",
        )
        report = {
            "status": "PASS",
            "mode": mode,
            "independent_check": json.loads((output / "checked.json").read_text()),
            "steps": steps,
        }
    else:
        r3 = work / R3 / "results"
        r4 = work / R4 / "results"
        quantile = work / R4 / "quantile_results"
        diagnostic_inputs = work / R4 / "diagnostic_inputs"
        diagnostic = work / R4 / "diagnostic_results"
        reference = work / R4 / "numerical_reference"
        run(f"{R3}/analyze.py", "--output", r3)
        compare("r3")
        run(
            f"{R3}/independent_audit.py",
            "--results",
            r3,
            "--output",
            output / "r3_audit.json",
        )
        run(f"{R4}/analyze.py", "--output", r4)
        compare("r4")
        run(f"{R4}/verify.py", "--results", r4, "--output", output / "r4_audit.json")
        run(
            f"{R4}/negative_checks.py",
            "--results",
            r4,
            "--output",
            output / "r4_adverse_checks.json",
        )
        run(f"{R4}/quantile_analysis.py", "--output", quantile)
        compare("quantile")
        run(
            f"{R4}/quantile_verify.py",
            "--results",
            quantile,
            "--output",
            output / "quantile_audit.json",
        )
        run(
            f"{R4}/quantile_negative_checks.py",
            "--results",
            quantile,
            "--output",
            output / "quantile_adverse_checks.json",
        )
        run(f"{R4}/study_manifest.py", "--output", diagnostic_inputs)
        compare("diagnostic_inputs")
        run(
            f"{R4}/diagnostic.py",
            "--manifest",
            diagnostic_inputs / "manifest.json",
            "--output",
            diagnostic,
        )
        compare("diagnostic")
        run(
            f"{R4}/diagnostic_verify.py",
            "--manifest",
            diagnostic_inputs / "manifest.json",
            "--results",
            diagnostic,
            "--output",
            output / "diagnostic_audit.json",
        )
        run(
            f"{R4}/diagnostic_checks.py",
            "--manifest",
            diagnostic_inputs / "manifest.json",
            "--output",
            output / "diagnostic_adverse_checks.json",
        )
        run(
            f"{R4}/second_review_checks.py",
            "--inputs",
            diagnostic_inputs,
            "--results",
            diagnostic,
            "--output",
            output / "worked_case_checks.json",
        )
        run(
            f"{R4}/numerical_audit.py",
            "--inputs",
            diagnostic_inputs,
            "--historical",
            diagnostic,
            "--output",
            reference,
        )
        compare("numerical_reference")
        run(
            f"{R4}/senior_reasoning_checks.py",
            "--reference",
            reference,
            "--diagnostic",
            diagnostic,
            "--output",
            output / "interpretation_checks.json",
        )
        review = work / R4 / "review_results"
        run(f"{R4}/review_analysis.py", "--inputs", diagnostic_inputs,
            "--reference", reference, "--protocol", work / R4 / "sensitivity_protocol.json",
            "--output", review)
        compare("review_analysis")
        run(f"{R4}/review_check.py", "--inputs", diagnostic_inputs,
            "--reference", reference, "--protocol", work / R4 / "sensitivity_protocol.json",
            "--results", review, "--report", output / "review_analysis_check.json")
        report = {
            "status": "PASS",
            "mode": mode,
            "exact_output_counts": comparisons,
            "numerical_csvs": sum(
                comparisons[g] for g in ("r3", "r4", "quantile", "diagnostic")
            ),
            "reference_files": comparisons["numerical_reference"],
            "steps": steps,
            "scope": "Processed-input numerical reproduction; manuscript, "
            "typeset tables, "
            "figure comparisons and upstream training/scoring are excluded.",
        }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("minimal", "full"))
    parser.add_argument(
        "--output", type=Path, required=True, help="A fresh output directory"
    )
    arguments = parser.parse_args()
    reproduce(arguments.mode, arguments.output.resolve())


if __name__ == "__main__":
    main()
