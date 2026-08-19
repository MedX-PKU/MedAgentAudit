#!/usr/bin/env python3
"""Build validation tables for the failure mode–correctness analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_DIR = Path(
    "/mnt/c/Users/LeiGu/Dropbox/[Preprint 2026] "
    "Auditing medical multi-agent AI reveals risks of false consensus/"
    "Preprint/analysis/failure_mode_correctness_association"
)
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def sign(value: object) -> int:
    if pd.isna(value):
        return 0
    numeric = float(value)
    return int(numeric > 0) - int(numeric < 0)


def build_bootstrap_checks(descriptive: pd.DataFrame) -> pd.DataFrame:
    result = descriptive.copy()
    rd = result["positive_accuracy"] - result["negative_accuracy"]
    rr = result["positive_accuracy"] / result["negative_accuracy"]
    positive_odds = result["positive_correct_n"] / (
        result["positive_n"] - result["positive_correct_n"]
    )
    negative_odds = result["negative_correct_n"] / (
        result["negative_n"] - result["negative_correct_n"]
    )
    odds_ratio = positive_odds / negative_odds
    result["point_estimates_verified"] = (
        np.isclose(result["risk_difference"], rd)
        & np.isclose(result["risk_ratio"], rr)
        & np.isclose(result["odds_ratio"], odds_ratio)
    )
    result["finite_ordered_95_ci"] = True
    for low, high in (
        ("risk_difference_ci_low", "risk_difference_ci_high"),
        ("risk_ratio_ci_low", "risk_ratio_ci_high"),
        ("odds_ratio_ci_low", "odds_ratio_ci_high"),
    ):
        result["finite_ordered_95_ci"] &= (
            np.isfinite(result[low])
            & np.isfinite(result[high])
            & result[low].le(result[high])
        )
    return result[[
        "failure_mode",
        "failure_mode_name",
        "question_clusters",
        "positive_n",
        "positive_correct_n",
        "positive_accuracy",
        "negative_n",
        "negative_correct_n",
        "negative_accuracy",
        "risk_difference",
        "risk_difference_ci_low",
        "risk_difference_ci_high",
        "risk_ratio",
        "risk_ratio_ci_low",
        "risk_ratio_ci_high",
        "odds_ratio",
        "odds_ratio_ci_low",
        "odds_ratio_ci_high",
        "point_estimates_verified",
        "finite_ordered_95_ci",
    ]]


def build_model_comparison(glmm: pd.DataFrame, gee: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "failure_mode",
        "failure_mode_name",
        "status",
        "beta",
        "adjusted_correctness_or",
        "ci_low",
        "ci_high",
        "adjusted_probability_difference",
        "adjusted_probability_difference_ci_low",
        "adjusted_probability_difference_ci_high",
        "p_value",
        "fdr_bh_p_value",
    ]
    result = glmm[columns].merge(
        gee[columns],
        on=["failure_mode", "failure_mode_name"],
        suffixes=("_glmm", "_gee"),
        validate="one_to_one",
    )
    result["direction_consistent"] = [
        sign(left) == sign(right)
        for left, right in zip(result["beta_glmm"], result["beta_gee"])
    ]
    result["raw_significance_consistent"] = (
        result["p_value_glmm"].lt(ALPHA)
        == result["p_value_gee"].lt(ALPHA)
    )
    result["fdr_significance_consistent"] = (
        result["fdr_bh_p_value_glmm"].lt(ALPHA)
        == result["fdr_bh_p_value_gee"].lt(ALPHA)
    )
    result["finite_ordered_probability_difference_95_ci"] = True
    for model in ("glmm", "gee"):
        low = f"adjusted_probability_difference_ci_low_{model}"
        high = f"adjusted_probability_difference_ci_high_{model}"
        result["finite_ordered_probability_difference_95_ci"] &= (
            np.isfinite(result[low])
            & np.isfinite(result[high])
            & result[low].le(result[high])
        )
    return result


def write_report(
    path: Path,
    bootstrap: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    disagreement = comparison.loc[~comparison["fdr_significance_consistent"]]
    lines = [
        "# Failure Mode–Correctness Analysis Checks",
        "",
        "## Question-Cluster Bootstrap",
        "",
        f"All {len(bootstrap)} modes have verified RD, RR, and OR point estimates and finite ordered 95% bootstrap intervals: "
        f"{bool(bootstrap['point_estimates_verified'].all() and bootstrap['finite_ordered_95_ci'].all())}.",
        "",
        "## Adjusted Association Models",
        "",
        f"GLMM and GEE coefficient directions agree for {int(comparison['direction_consistent'].sum())}/{len(comparison)} modes.",
        f"Modes with different FDR classifications: {', '.join(disagreement['failure_mode']) or 'none'}.",
        "All GLMM and GEE adjusted probability differences have finite ordered 95% confidence intervals: "
        f"{bool(comparison['finite_ordered_probability_difference_95_ci'].all())}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    descriptive = read_csv(
        args.analysis_dir / "failure_correctness_descriptive_results.csv"
    )
    glmm = read_csv(args.analysis_dir / "failure_correctness_glmm_results.csv")
    gee = read_csv(args.analysis_dir / "failure_correctness_gee_results.csv")

    bootstrap = build_bootstrap_checks(descriptive)
    comparison = build_model_comparison(glmm, gee)
    output_names = {
        "bootstrap": "failure_correctness_question_cluster_bootstrap_checks.csv",
        "comparison": "failure_correctness_glmm_gee_comparison.csv",
        "report": "failure_correctness_analysis_checks.md",
        "metadata": "failure_correctness_analysis_checks_metadata.json",
    }
    bootstrap.to_csv(args.analysis_dir / output_names["bootstrap"], index=False)
    comparison.to_csv(args.analysis_dir / output_names["comparison"], index=False)
    write_report(args.analysis_dir / output_names["report"], bootstrap, comparison)
    metadata = {
        "analysis_section": "failure mode–correctness associations",
        "source_analysis_directory": str(args.analysis_dir),
        "significance_threshold": ALPHA,
        "outputs": list(output_names.values())[:-1],
    }
    (args.analysis_dir / output_names["metadata"]).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
