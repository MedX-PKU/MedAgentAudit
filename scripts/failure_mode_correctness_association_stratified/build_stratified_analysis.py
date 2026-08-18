#!/usr/bin/env python3
"""Build failure mode–correctness stratified-analysis tables."""

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
STRATIFICATION_VARIABLES = ("dataset", "mas", "underlying_llm")
MIN_GROUP_SIZE = 10
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--stratified-dir", type=Path)
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


def merge_models(
    raw: pd.DataFrame, glmm: pd.DataFrame, gee: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "failure_mode",
        "failure_mode_name",
        "stratification_variable",
        "stratum",
    ]
    raw_columns = keys + [
        "eligible_valid_n",
        "question_clusters",
        "positive_n",
        "negative_n",
        "positive_accuracy",
        "negative_accuracy",
        "risk_difference",
        "risk_ratio",
        "odds_ratio",
    ]
    model_columns = keys + [
        "status",
        "beta",
        "adjusted_correctness_or",
        "ci_low",
        "ci_high",
        "p_value",
    ]
    return raw[raw_columns].merge(
        glmm[model_columns], on=keys, how="left", validate="one_to_one"
    ).merge(
        gee[model_columns],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_glmm", "_gee"),
    )


def build_modality_table(
    merged: pd.DataFrame, pooled_glmm: pd.DataFrame
) -> pd.DataFrame:
    result = merged.loc[merged["stratification_variable"].eq("modality")].copy()
    pooled = pooled_glmm[["failure_mode", "beta"]].rename(
        columns={"beta": "pooled_glmm_beta"}
    )
    result = result.merge(pooled, on="failure_mode", validate="many_to_one")
    result["glmm_direction_vs_pooled"] = [
        "same" if sign(value) == sign(pooled_value) else "reversed"
        for value, pooled_value in zip(result["beta_glmm"], result["pooled_glmm_beta"])
    ]
    result["gee_direction_vs_pooled"] = [
        "same" if sign(value) == sign(pooled_value) else "reversed"
        for value, pooled_value in zip(result["beta_gee"], result["pooled_glmm_beta"])
    ]
    result["glmm_p_below_0_05"] = result["p_value_glmm"].lt(ALPHA)
    result["gee_p_below_0_05"] = result["p_value_gee"].lt(ALPHA)
    return result


def build_direction_table(
    merged: pd.DataFrame, pooled_glmm: pd.DataFrame
) -> pd.DataFrame:
    result = merged.loc[
        merged["stratification_variable"].isin(STRATIFICATION_VARIABLES)
        & merged["positive_n"].ge(MIN_GROUP_SIZE)
        & merged["negative_n"].ge(MIN_GROUP_SIZE)
    ].copy()
    pooled = pooled_glmm[["failure_mode", "beta"]].rename(
        columns={"beta": "pooled_glmm_beta"}
    )
    result = result.merge(pooled, on="failure_mode", validate="many_to_one")
    result["selected_model"] = "GLMM"
    result["selected_beta"] = result["beta_glmm"]
    result["selected_or"] = result["adjusted_correctness_or_glmm"]
    result["selected_ci_low"] = result["ci_low_glmm"]
    result["selected_ci_high"] = result["ci_high_glmm"]
    result["selected_p_value"] = result["p_value_glmm"]

    gee_fallback = (
        ~result["status_glmm"].eq("completed")
        & result["status_gee"].eq("completed")
    )
    medqa_fallback = (
        gee_fallback
        & result["stratification_variable"].eq("dataset")
        & result["stratum"].eq("medqa")
    )
    result.loc[gee_fallback, "selected_model"] = "GEE fallback"
    result.loc[medqa_fallback, "selected_model"] = "GEE (medqa fallback)"
    for target, source in (
        ("selected_beta", "beta_gee"),
        ("selected_or", "adjusted_correctness_or_gee"),
        ("selected_ci_low", "ci_low_gee"),
        ("selected_ci_high", "ci_high_gee"),
        ("selected_p_value", "p_value_gee"),
    ):
        result.loc[gee_fallback, target] = result.loc[gee_fallback, source]

    unavailable = ~result["status_glmm"].eq("completed") & ~gee_fallback
    result.loc[unavailable, "selected_model"] = "not evaluable"
    result.loc[unavailable, [
        "selected_beta",
        "selected_or",
        "selected_ci_low",
        "selected_ci_high",
        "selected_p_value",
    ]] = np.nan
    result["direction_reversal_vs_pooled"] = [
        False if pd.isna(value) else sign(value) != sign(pooled_value)
        for value, pooled_value in zip(
            result["selected_beta"], result["pooled_glmm_beta"]
        )
    ]
    result["selected_p_below_0_05"] = result["selected_p_value"].lt(ALPHA)
    return result


def build_interaction_table(glmm: pd.DataFrame, gee: pd.DataFrame) -> pd.DataFrame:
    keys = ["failure_mode", "failure_mode_name", "comparison"]
    columns = keys + [
        "status",
        "beta",
        "adjusted_correctness_or",
        "ci_low",
        "ci_high",
        "p_value",
        "fdr_bh_p_value",
    ]
    result = glmm[columns].merge(
        gee[columns],
        on=keys,
        suffixes=("_glmm", "_gee"),
        validate="one_to_one",
    )
    result["glmm_fdr_below_0_05"] = result["fdr_bh_p_value_glmm"].lt(ALPHA)
    result["gee_fdr_below_0_05"] = result["fdr_bh_p_value_gee"].lt(ALPHA)
    result["direction_consistent"] = [
        sign(glmm_beta) == sign(gee_beta)
        for glmm_beta, gee_beta in zip(result["beta_glmm"], result["beta_gee"])
    ]
    result["fdr_significance_consistent"] = (
        result["glmm_fdr_below_0_05"] == result["gee_fdr_below_0_05"]
    )
    result["model_specification_sensitive"] = (
        ~result["direction_consistent"]
        | ~result["fdr_significance_consistent"]
    )
    return result


def fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    numeric = float(value)
    if numeric != 0 and abs(numeric) < 0.001:
        return f"{numeric:.2e}"
    return f"{numeric:.3f}"


def display(value: object) -> str:
    return str(value).replace("deepseek-reasoner", "DeepSeek-V3.2-Thinking")


def write_report(
    path: Path,
    modality: pd.DataFrame,
    interactions: pd.DataFrame,
    directions: pd.DataFrame,
) -> None:
    focus = modality.loc[
        modality["stratum"].eq("VQA")
        & modality["failure_mode"].isin(["F-1.2.1", "F-3.1.3"])
    ]
    reversals = directions.loc[directions["direction_reversal_vs_pooled"]]
    supported = reversals.loc[reversals["selected_p_below_0_05"]]
    medqa = directions.loc[
        directions["stratification_variable"].eq("dataset")
        & directions["stratum"].eq("medqa")
    ]
    other_fallbacks = directions.loc[directions["selected_model"].eq("GEE fallback")]
    lines = [
        "# Stratified analyses and cross-stratum comparisons",
        "",
        "## QA/VQA stratification",
        "",
    ]
    for row in focus.itertuples():
        lines.append(
            f"- {row.failure_mode}, VQA: GLMM OR={fmt(row.adjusted_correctness_or_glmm)}, "
            f"P={fmt(row.p_value_glmm)}; GEE OR={fmt(row.adjusted_correctness_or_gee)}, "
            f"P={fmt(row.p_value_gee)}."
        )
    lines.extend([
        "",
        "## QA/VQA interaction tests",
        "",
        "The interaction OR is the VQA failure-positive OR divided by the QA failure-positive OR. "
        "GLMM and GEE P values are BH-FDR corrected across the ten modes separately.",
        "",
    ])
    for row in interactions.itertuples():
        lines.append(
            f"- {row.failure_mode}: GLMM interaction OR={fmt(row.adjusted_correctness_or_glmm)}, "
            f"P={fmt(row.p_value_glmm)}, FDR P={fmt(row.fdr_bh_p_value_glmm)}; "
            f"GEE interaction OR={fmt(row.adjusted_correctness_or_gee)}, "
            f"P={fmt(row.p_value_gee)}, FDR P={fmt(row.fdr_bh_p_value_gee)}."
        )
    sensitive_modes = interactions.loc[
        interactions["model_specification_sensitive"], "failure_mode"
    ]
    lines.append("")
    lines.append(
        "Modes with GLMM–GEE direction or FDR-significance disagreement: "
        f"{', '.join(sensitive_modes) or 'none'}."
    )
    lines.extend([
        "",
        "## Dataset, MAS, and underlying LLM direction checks",
        "",
        f"Eligible strata: {len(directions)}; direction reversals relative to the pooled GLMM: {len(reversals)}; "
        f"reversals with selected-model P below 0.05: {len(supported)}.",
        "",
        f"MedQA GEE fallbacks: {int(medqa['selected_model'].eq('GEE (medqa fallback)').sum())}/{len(medqa)}.",
        f"Other GEE fallbacks: {len(other_fallbacks)}.",
        "",
        "Direction reversals:",
        "",
    ])
    for row in reversals.itertuples():
        lines.append(
            f"- {row.failure_mode}, {row.stratification_variable}={display(row.stratum)}: "
            f"{row.selected_model}, OR={fmt(row.selected_or)}, P={fmt(row.selected_p_value)}."
        )
    lines.extend([
        "",
        "Separate stratum P values do not test whether two strata have different association estimates. "
        "A claim that association estimates differ across strata requires an interaction test.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    stratified_dir = args.stratified_dir or args.analysis_dir / "stratified_analysis"
    stratified_dir.mkdir(parents=True, exist_ok=True)
    raw = read_csv(args.analysis_dir / "failure_correctness_stratified_results.csv")
    pooled_glmm = read_csv(args.analysis_dir / "failure_correctness_glmm_results.csv")
    glmm = read_csv(stratified_dir / "all_strata_glmm_results.csv")
    gee = read_csv(stratified_dir / "all_strata_gee_results.csv")
    interaction_glmm = read_csv(
        stratified_dir / "qa_vqa_interaction_glmm_results.csv"
    )
    interaction_gee = read_csv(
        stratified_dir / "qa_vqa_interaction_gee_results.csv"
    )
    merged = merge_models(raw, glmm, gee)
    modality = build_modality_table(merged, pooled_glmm)
    interactions = build_interaction_table(interaction_glmm, interaction_gee)
    directions = build_direction_table(merged, pooled_glmm)
    modality.to_csv(stratified_dir / "qa_vqa_models.csv", index=False)
    interactions.to_csv(
        stratified_dir / "qa_vqa_interaction_tests.csv", index=False
    )
    directions.to_csv(stratified_dir / "stratified_direction_checks.csv", index=False)
    write_report(
        stratified_dir / "stratified_analysis_summary.md",
        modality,
        interactions,
        directions,
    )
    metadata = {
        "analysis_section": "stratified analyses and cross-stratum comparisons",
        "source_analysis_directory": str(args.analysis_dir),
        "minimum_positive_and_negative_per_stratum": MIN_GROUP_SIZE,
        "direction_reversal_reference": "pooled GLMM failure-positive coefficient",
        "gee_fallback_rule": (
            "use and label stratum GEE when the corresponding GLMM status is not completed"
        ),
        "cross_stratum_comparison_rule": (
            "separate stratum P values do not compare association estimates; "
            "cross-stratum claims require interaction tests"
        ),
        "outputs": [
            "all_strata_glmm_results.csv",
            "all_strata_gee_results.csv",
            "skipped_strata.csv",
            "qa_vqa_models.csv",
            "qa_vqa_interaction_glmm_results.csv",
            "qa_vqa_interaction_gee_results.csv",
            "qa_vqa_interaction_tests.csv",
            "stratified_direction_checks.csv",
            "stratified_analysis_summary.md",
        ],
    }
    (stratified_dir / "stratified_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
