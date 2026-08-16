#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="/mnt/c/Users/LeiGu/Dropbox/[Preprint 2026] Auditing medical multi-agent AI reveals risks of false consensus/Preprint/analysis/failure_mode_correctness_association"

cd "${PROJECT_ROOT}"

uv run python "${SCRIPT_DIR}/build_failure_correctness_case_manifest.py" \
  --output-dir "${OUTPUT_DIR}"

uv run python "${SCRIPT_DIR}/run_preliminary_failure_correctness_models.py" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-replicates 500 \
  --seed 20260816

uv run python "${SCRIPT_DIR}/summarize_preliminary_failure_correctness_results.py" \
  --output-dir "${OUTPUT_DIR}"
