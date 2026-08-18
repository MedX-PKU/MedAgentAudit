#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/leigu/Documents/[Preprint 2026] Auditing medical multi-agent AI reveals risks of false consensus/code_repo"
SCRIPT_ROOT="${PROJECT_ROOT}/scripts/failure_mode_correctness_association_stratified"

cd "${PROJECT_ROOT}"

.venv/bin/python \
  "${SCRIPT_ROOT}/run_stratified_analysis.py"

.venv/bin/python \
  "${SCRIPT_ROOT}/build_stratified_analysis.py"
