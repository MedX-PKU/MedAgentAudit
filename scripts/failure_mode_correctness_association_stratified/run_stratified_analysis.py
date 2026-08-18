#!/usr/bin/env python3
"""Fit stratified GLMM and GEE models for all failure modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_SCRIPT_DIR = PROJECT_ROOT / "scripts/failure_mode_correctness_association"
DEFAULT_ANALYSIS_DIR = Path(
    "/mnt/c/Users/LeiGu/Dropbox/[Preprint 2026] "
    "Auditing medical multi-agent AI reveals risks of false consensus/"
    "Preprint/analysis/failure_mode_correctness_association"
)
sys.path.insert(0, str(PRIMARY_SCRIPT_DIR))

import run_failure_correctness_analysis as rfa
from failure_mode_schema import FAILURE_MODES


STRATA_VARIABLES = ("modality", "mas", "dataset", "underlying_llm")
MIN_GROUP = 10
FORMULA_BY_VARIABLE = {
    "modality": rfa.FIXED_FORMULA,
    "mas": "correctness ~ failure_positive + C(dataset) + C(underlying_llm)",
    "dataset": "correctness ~ failure_positive + C(mas) + C(underlying_llm)",
    "underlying_llm": "correctness ~ failure_positive + C(dataset) + C(mas)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "failure_correctness_case_manifest.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "stratified_analysis",
    )
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
    interaction_glmm_rows: list[dict[str, object]] = []
    interaction_gee_rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for mode in FAILURE_MODES:
        data = rfa.prepare_mode_data(manifest, mode.code)
        interaction_data = data.copy()
        interaction_data["failure_x_vqa"] = (
            interaction_data["failure_positive"]
            * interaction_data["modality"].eq("VQA").astype(int)
        )
        interaction_formula = rfa.FIXED_FORMULA + " + failure_x_vqa"
        interaction_glmm = rfa.fit_frequentist_glmm(
            mode,
            interaction_data,
            interaction_formula,
            args.quadrature_nodes,
            args.sensitivity_quadrature_nodes,
            "failure_x_vqa",
        )
        interaction_glmm["comparison"] = "VQA versus QA failure association"
        interaction_glmm_rows.append(interaction_glmm)
        interaction_gee = rfa.fit_gee(
            mode, interaction_data, interaction_formula, "failure_x_vqa"
        )
        interaction_gee["comparison"] = "VQA versus QA failure association"
        interaction_gee_rows.append(interaction_gee)

        for variable in STRATA_VARIABLES:
            formula = FORMULA_BY_VARIABLE[variable]
            for level, subset in data.groupby(variable, observed=True):
                key = f"F-{mode.code}|{variable}={level}"
                if not stratum_ok(subset):
                    skipped.append({"mode": key, "reason": "group_size"})
                    continue
                glmm = rfa.fit_frequentist_glmm(
                    mode,
                    subset,
                    formula,
                    args.quadrature_nodes,
                    args.sensitivity_quadrature_nodes,
                )
                glmm["stratification_variable"] = variable
                glmm["stratum"] = level
                glmm_rows.append(glmm)

                gee = rfa.fit_gee(mode, subset, formula)
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
    rfa.apply_fdr(pd.DataFrame(interaction_glmm_rows)).to_csv(
        args.output_dir / "qa_vqa_interaction_glmm_results.csv", index=False
    )
    rfa.apply_fdr(pd.DataFrame(interaction_gee_rows)).to_csv(
        args.output_dir / "qa_vqa_interaction_gee_results.csv", index=False
    )
    pd.DataFrame(skipped).to_csv(args.output_dir / "skipped_strata.csv", index=False)
    metadata = {
        "analysis_definition": "failure mode–correctness stratified analyses",
        "source": "frozen failure-correctness case manifest",
        "new_mas_llm_or_auditor_runs": False,
        "stratification_variables": list(STRATA_VARIABLES),
        "qa_vqa_interaction_tests": "all ten failure modes; QA reference",
        "interaction_fdr_families": (
            "Benjamini-Hochberg correction across ten modes separately for GLMM and GEE"
        ),
        "min_group_size": MIN_GROUP,
        "quadrature_nodes": args.quadrature_nodes,
        "sensitivity_quadrature_nodes": args.sensitivity_quadrature_nodes,
        "manifest": str(args.manifest),
    }
    (args.output_dir / "model_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
