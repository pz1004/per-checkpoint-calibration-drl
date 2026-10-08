"""Check Git's publishable file set without staging, committing or uploading."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SUPPORT = {
    ".gitignore",
    ".python-version",
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "scripts/01_example.sh",
    "scripts/02_reproduce.sh",
    "scripts/reproduce.py",
    "scripts/check_public_tree.py",
    "assets/checkpoint_diagnostics.png",
    "reports/README.md",
    "reports/reproduction_manifest.json",
    "reports/results.md",
    "reports/reproduction.md",
    "reports/verification.json",
    "reports/historical_operating_points.csv",
    "reports/matched_operating_points.csv",
    "reports/quantile_operating_points.csv",
    "reports/diagnostic_operating_points.csv",
    "reports/diagnostic_condition_summary.csv",
}


def git_files(*arguments: str) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", *arguments], cwd=ROOT, check=True, capture_output=True
    )
    return {name.decode() for name in result.stdout.split(b"\0") if name}


def main() -> None:
    manifest = json.loads((ROOT / "reports/reproduction_manifest.json").read_text())
    expected = set(manifest["inputs"]) | PUBLIC_SUPPORT
    visible = git_files("--cached", "--others", "--exclude-standard")
    tracked_ignored = git_files("--cached", "--ignored", "--exclude-standard")
    if tracked_ignored:
        raise SystemExit(f"Ignored paths are still tracked: {sorted(tracked_ignored)}")
    if visible != expected:
        raise SystemExit(
            f"Unexpected public paths: {sorted(visible - expected)}\n"
            f"Missing public paths: {sorted(expected - visible)}"
        )
    oversized = [
        name for name in visible if (ROOT / name).stat().st_size >= 100_000_000
    ]
    if oversized:
        raise SystemExit(f"Generated files exceed the public size limit: {oversized}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "public_files": len(visible),
                "public_bytes": sum((ROOT / name).stat().st_size for name in visible),
                "tracked_private_files": len(tracked_ignored),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
