#!/usr/bin/env python3
"""Run all-mode stratified sensitivity models without touching frozen scripts.

This script only imports the frozen analysis module and reads the frozen case
manifest. It does not modify any existing code, log, or result file. All
outputs go to a new Dropbox folder passed via --output-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


FROZEN_SCRIPT_DIR = Path(
    "/home/leigu/Documents/[Preprint 2026] Auditing medical multi-agent AI "
    "reveals risks of false consensus/code_repo/"
    "scripts/failure_mode_correctness_association"
)
sys.path.insert(0, str(FROZEN_SCRIPT_DIR))

import run_failure_correctness_analysis as rfa
from failure_mode_schema import FAILURE_MODES


STRATA_VARIABLES = ("modality", "mas", "dataset")
MIN_GROUP = 10

# When stratifying by a factor, that factor becomes constant inside every
# stratum, so its fixed effect must be dropped to keep the design full rank.
FORMULA_BY_VARIABLE = {
    "modality": rfa.FIXED_FORMULA,
    "mas": "correctness ~ failure_positive + C(dataset) + C(underlying_llm)",
    "dataset": "correctness ~ failure_positive + C(mas) + C(underlying_llm)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quadrature-nodes", type=int, default=9)
    parser.add_argument("--sensitivity-quadrature-nodes", type=int, default=15)
    return parser.parse_args()


def stratum_ok(subset: pd.DataFrame) -> bool:
    counts = subset["failure_positive"].value_counts()
    return counts.get(1, 0) >= MIN_GROUP and counts.get(0, 0) >= MIN_GROUP


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    glmm_rows: list[dict[str, object]] = []
    gee_rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for mode in FAILURE_MODES:
        data = rfa.prepare_mode_data(manifest, mode.code)
        for variable in STRATA_VARIABLES:
            formula = FORMULA_BY_VARIABLE[variable]
            for level, subset in data.groupby(variable, observed=True):
                key = f"F-{mode.code}|{variable}={level}"
                if not stratum_ok(subset):
                    skipped.append({"mode": key, "reason": "group_size"})
                    continue
                try:
                    glmm = rfa.fit_frequentist_glmm(
                        mode,
                        subset,
                        formula,
                        args.quadrature_nodes,
                        args.sensitivity_quadrature_nodes,
                    )
                except Exception as exc:
                    skipped.append(
                        {"mode": key, "reason": f"glmm:{type(exc).__name__}"}
                    )
                    continue
                glmm["stratification_variable"] = variable
                glmm["stratum"] = level
                glmm_rows.append(glmm)
                try:
                    gee = rfa.fit_gee(mode, subset, formula)
                except Exception as exc:
                    skipped.append(
                        {"mode": key, "reason": f"gee:{type(exc).__name__}"}
                    )
                    continue
                gee["stratification_variable"] = variable
                gee["stratum"] = level
                gee_rows.append(gee)
        print(f"done F-{mode.code}", flush=True)

    pd.DataFrame(glmm_rows).to_csv(
        args.output_dir / "all_strata_glmm_results.csv", index=False
    )
    pd.DataFrame(gee_rows).to_csv(
        args.output_dir / "all_strata_gee_results.csv", index=False
    )
    pd.DataFrame(skipped).to_csv(
        args.output_dir / "skipped_strata.csv", index=False
    )
    metadata = {
        "analysis_definition": "all-mode stratified sensitivity checks",
        "source": "frozen failure-correctness case manifest",
        "new_model_or_auditor_runs": False,
        "stratification_variables": list(STRATA_VARIABLES),
        "min_group_size": MIN_GROUP,
        "quadrature_nodes": args.quadrature_nodes,
        "sensitivity_quadrature_nodes": args.sensitivity_quadrature_nodes,
        "primary_model": (
            "maximum-likelihood logistic GLMM with dataset:question_ID "
            "random intercept; per-stratum fixed-effect terms dropped as needed"
        ),
        "manifest": str(args.manifest),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
