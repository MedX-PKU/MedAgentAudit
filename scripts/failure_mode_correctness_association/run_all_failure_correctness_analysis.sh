#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run python "${SCRIPT_DIR}/build_failure_correctness_case_manifest.py"
uv run python "${SCRIPT_DIR}/build_revised_repetition_labels.py"
uv run python "${SCRIPT_DIR}/run_failure_correctness_analysis.py" \
  --bootstrap-replicates 2000 \
  --quadrature-nodes 9 \
  --sensitivity-quadrature-nodes 15
