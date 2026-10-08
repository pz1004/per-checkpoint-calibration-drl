#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
exec "${PYTHON:-python}" scripts/reproduce.py full --output "${1:-outputs/reproduction}"
