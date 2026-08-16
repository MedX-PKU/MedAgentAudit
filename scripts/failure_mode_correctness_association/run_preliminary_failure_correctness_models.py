#!/usr/bin/env python3
"""Run preliminary descriptive, Bayesian GLMM, and GEE analyses by failure mode."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.multitest import multipletests

from failure_mode_schema import FAILURE_MODES


FIXED_FORMULA = (
    "correctness ~ failure_positive + C(dataset) + C(mas) + C(underlying_llm)"
)


def parse_args() -> argparse.Namespace:
    default_output = Path(
        "/mnt/c/Users/LeiGu/Dropbox/[Preprint 2026] "
        "Auditing medical multi-agent AI reveals risks of false consensus/"
        "Preprint/analysis/failure_mode_correctness_association"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--reuse-existing-glmm",
        action="store_true",
        help="Reuse the existing GLMM CSV while rerunning descriptive and GEE results.",
    )
    return parser.parse_args()


def odds_ratio_from_counts(a: int, b: int, c: int, d: int) -> float:
    values = np.asarray([a, b, c, d], dtype=float)
    if np.any(values == 0):
        values += 0.5
    return float((values[1] * values[2]) / (values[0] * values[3]))


def cluster_bootstrap_difference(
    data: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    clusters = data["question_cluster"].unique()
    grouped = {key: frame for key, frame in data.groupby("question_cluster")}
    estimates: list[float] = []
    for _ in range(replicates):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[key] for key in selected], ignore_index=True)
        means = sample.groupby("failure_positive")["correctness"].mean()
        if 0 in means.index and 1 in means.index:
            estimates.append(float(means.loc[1] - means.loc[0]))
    if not estimates:
        return math.nan, math.nan
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def prepare_mode_data(manifest: pd.DataFrame, code: str) -> pd.DataFrame:
    prefix = f"f_{code.replace('.', '_')}"
    data = manifest.loc[
        manifest[f"{prefix}_state"].isin(["positive", "negative"])
        & manifest["correctness"].notna()
    ].copy()
    data["failure_positive"] = data[f"{prefix}_label"].astype(int)
    data["correctness"] = data["correctness"].astype(int)
    return data


def descriptive_result(
    mode, data: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> dict[str, object]:
    table = pd.crosstab(data["failure_positive"], data["correctness"])
    for exposure in (0, 1):
        for outcome in (0, 1):
            if exposure not in table.index or outcome not in table.columns:
                table.loc[exposure, outcome] = 0
    table = table.sort_index().sort_index(axis=1)
    neg_incorrect, neg_correct = int(table.loc[0, 0]), int(table.loc[0, 1])
    pos_incorrect, pos_correct = int(table.loc[1, 0]), int(table.loc[1, 1])
    pos_total = pos_incorrect + pos_correct
    neg_total = neg_incorrect + neg_correct
    pos_accuracy = pos_correct / pos_total if pos_total else math.nan
    neg_accuracy = neg_correct / neg_total if neg_total else math.nan
    difference = pos_accuracy - neg_accuracy
    ci_low, ci_high = cluster_bootstrap_difference(data, replicates, rng)
    return {
        "failure_mode": f"F-{mode.code}",
        "failure_mode_name": mode.short_name,
        "eligible_valid_n": len(data),
        "question_clusters": data["question_cluster"].nunique(),
        "positive_n": pos_total,
        "positive_correct_n": pos_correct,
        "positive_accuracy": pos_accuracy,
        "negative_n": neg_total,
        "negative_correct_n": neg_correct,
        "negative_accuracy": neg_accuracy,
        "unadjusted_accuracy_difference": difference,
        "cluster_bootstrap_ci_low": ci_low,
        "cluster_bootstrap_ci_high": ci_high,
        "unadjusted_correctness_odds_ratio": odds_ratio_from_counts(
            pos_incorrect, pos_correct, neg_incorrect, neg_correct
        ),
    }


def fit_glmm_vb(mode, data: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        "failure_mode": f"F-{mode.code}",
        "failure_mode_name": mode.short_name,
        "model": "BinomialBayesMixedGLM variational Bayes",
        "status": "not_run",
        "note": "preliminary approximation; confirm with frequentist GLMM",
    }
    if mode.code == "2.2.1":
        result.update(
            status="excluded_original_near_constant_label",
            note="Use revised intermediate-answer-correctness label in the primary model.",
        )
        return result
    try:
        model = BinomialBayesMixedGLM.from_formula(
            FIXED_FORMULA,
            {"question_random_intercept": "0 + C(question_cluster)"},
            data,
            vcp_p=0.5,
            fe_p=10.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = model.fit_vb(
                mean=np.zeros(model.k_fep + model.k_vcp + model.k_vc),
                sd=np.ones(model.k_fep + model.k_vcp + model.k_vc),
                minim_opts={"maxiter": 1000},
            )
        names = list(model.exog_names)
        index = names.index("failure_positive")
        beta = float(fitted.fe_mean[index])
        standard_error = float(fitted.fe_sd[index])
        z_value = beta / standard_error
        p_value = float(2 * st.norm.sf(abs(z_value)))
        design_info = patsy.dmatrix(
            FIXED_FORMULA.split("~", 1)[1], data, return_type="dataframe"
        ).design_info
        data_positive = data.copy()
        data_positive["failure_positive"] = 1
        data_negative = data.copy()
        data_negative["failure_positive"] = 0
        x_positive = patsy.build_design_matrices(
            [design_info], data_positive, return_type="dataframe"
        )[0]
        x_negative = patsy.build_design_matrices(
            [design_info], data_negative, return_type="dataframe"
        )[0]
        positive_probability = float(expit(np.asarray(x_positive) @ fitted.fe_mean).mean())
        negative_probability = float(expit(np.asarray(x_negative) @ fitted.fe_mean).mean())
        optimizer = fitted.optim_retvals
        warning_text = " | ".join(str(item.message) for item in caught)
        result.update(
            status="completed" if optimizer.get("success", False) else "optimizer_not_successful",
            n=len(data),
            question_clusters=data["question_cluster"].nunique(),
            beta=beta,
            posterior_sd=standard_error,
            approximate_z=z_value,
            approximate_p=p_value,
            adjusted_correctness_or=math.exp(beta),
            interval_low=math.exp(beta - 1.96 * standard_error),
            interval_high=math.exp(beta + 1.96 * standard_error),
            fixed_component_positive_probability=positive_probability,
            fixed_component_negative_probability=negative_probability,
            fixed_component_probability_difference=positive_probability
            - negative_probability,
            question_random_intercept_sd=math.exp(float(fitted.vcp_mean[0])),
            optimizer_success=bool(optimizer.get("success", False)),
            optimizer_message=str(optimizer.get("message", "")),
            warnings=warning_text,
        )
    except Exception as exc:  # retain failure transparently in the results
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return result


def fit_gee(mode, data: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        "failure_mode": f"F-{mode.code}",
        "failure_mode_name": mode.short_name,
        "model": "question-clustered GEE with independence working correlation",
        "status": "not_run",
    }
    if mode.code == "2.2.1":
        result.update(
            status="excluded_original_near_constant_label",
            note="Use revised intermediate-answer-correctness label in the primary model.",
        )
        return result
    try:
        model = smf.gee(
            FIXED_FORMULA,
            groups="question_cluster",
            data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        )
        fitted = model.fit(maxiter=200)
        beta = float(fitted.params["failure_positive"])
        standard_error = float(fitted.bse["failure_positive"])
        data_positive = data.copy()
        data_positive["failure_positive"] = 1
        data_negative = data.copy()
        data_negative["failure_positive"] = 0
        positive_probability = float(np.mean(fitted.predict(data_positive)))
        negative_probability = float(np.mean(fitted.predict(data_negative)))
        result.update(
            status="completed" if fitted.converged else "not_converged",
            n=len(data),
            question_clusters=data["question_cluster"].nunique(),
            beta=beta,
            robust_se=standard_error,
            z=float(fitted.tvalues["failure_positive"]),
            p_value=float(fitted.pvalues["failure_positive"]),
            adjusted_correctness_or=math.exp(beta),
            ci_low=math.exp(beta - 1.96 * standard_error),
            ci_high=math.exp(beta + 1.96 * standard_error),
            standardized_positive_probability=positive_probability,
            standardized_negative_probability=negative_probability,
            standardized_probability_difference=positive_probability
            - negative_probability,
            dependence_parameter=0.0,
            converged=bool(fitted.converged),
        )
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return result


def apply_fdr(gee_results: pd.DataFrame) -> pd.DataFrame:
    gee_results["fdr_bh_p_value"] = np.nan
    mask = (gee_results["status"] == "completed") & gee_results["p_value"].notna()
    if mask.any():
        gee_results.loc[mask, "fdr_bh_p_value"] = multipletests(
            gee_results.loc[mask, "p_value"].astype(float), method="fdr_bh"
        )[1]
    return gee_results


def make_forest_plot(results: pd.DataFrame, output_dir: Path) -> None:
    completed = results.loc[results["status"] == "completed"].copy()
    if completed.empty:
        return
    completed = completed.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(completed))
    estimate = completed["adjusted_correctness_or"].astype(float).to_numpy()
    low = completed["ci_low"].astype(float).to_numpy()
    high = completed["ci_high"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.errorbar(
        estimate,
        y,
        xerr=np.vstack([estimate - low, high - estimate]),
        fmt="o",
        color="#315b7d",
        ecolor="#7893aa",
        capsize=3,
    )
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row.failure_mode} {row.failure_mode_name}" for row in completed.itertuples()]
    )
    ax.set_xlabel("Adjusted odds ratio for benchmark answer correctness (GEE)")
    ax.set_title("Preliminary Failure–Correctness Associations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "preliminary_failure_correctness_gee_forest_plot.png", dpi=240)
    plt.close(fig)


def write_report(
    output_dir: Path,
    descriptive: pd.DataFrame,
    glmm: pd.DataFrame,
    gee: pd.DataFrame,
) -> None:
    merged = descriptive.merge(
        glmm[
            [
                "failure_mode",
                "status",
                "adjusted_correctness_or",
                "interval_low",
                "interval_high",
                "fixed_component_probability_difference",
            ]
        ].rename(columns={"status": "glmm_status"}),
        on="failure_mode",
        how="left",
    ).merge(
        gee[
            [
                "failure_mode",
                "status",
                "adjusted_correctness_or",
                "ci_low",
                "ci_high",
                "standardized_probability_difference",
                "p_value",
                "fdr_bh_p_value",
            ]
        ].rename(columns={"status": "gee_status"}),
        on="failure_mode",
        how="left",
        suffixes=("_glmm", "_gee"),
    )
    merged.to_csv(
        output_dir / "preliminary_failure_correctness_combined_results.csv",
        index=False,
    )

    lines = [
        "# Preliminary Failure–Correctness Analysis",
        "",
        "## Scope",
        "",
        "This analysis uses frozen existing MAS and auditor outputs; no MAS or auditor was rerun. Results are preliminary and must not be copied into the manuscript before the final eligibility map, revised F-2.2.1 label, and frequentist GLMM are completed.",
        "",
        "## Model Status",
        "",
        "- Descriptive accuracy differences use question-cluster bootstrap intervals.",
        "- The mixed-effects estimate uses statsmodels BinomialBayesMixedGLM with a question random intercept and variational Bayes. Its interval is an approximate posterior interval, not the final frequentist confidence interval specified in the protocol.",
        "- Question-clustered GEE with an independence working correlation and robust standard errors is reported as a population-averaged sensitivity analysis.",
        "- Original F-2.2.1 is excluded from adjusted models because it is near constant; the revised intermediate-answer-correctness label has not yet been generated.",
        "- F-3.1.1 contains correctness information in its definition, so its association is partly circular and is not an independent validation result.",
        "",
        "## Combined Results",
        "",
        "| Mode | N | Positive accuracy | Negative accuracy | Unadjusted difference | GLMM OR | GEE OR | GEE standardized difference | GEE FDR P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in merged.itertuples():
        def f(value, digits=3):
            return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    row.failure_mode,
                    str(int(row.eligible_valid_n)),
                    f(row.positive_accuracy),
                    f(row.negative_accuracy),
                    f(row.unadjusted_accuracy_difference),
                    f(row.adjusted_correctness_or_glmm),
                    f(row.adjusted_correctness_or_gee),
                    f(row.standardized_probability_difference),
                    f(row.fdr_bh_p_value),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "Because correctness is coded as 1, an odds ratio below 1 or a negative probability difference indicates lower benchmark answer correctness among failure-positive cases. These are observational associations, not causal effects or patient-outcome estimates.",
            "",
            "## Files",
            "",
            "- `failure_correctness_case_manifest.csv.gz`: one row per MAS case with four-state mode labels.",
            "- `failure_correctness_manifest_flow_summary.csv`: planned, observed, and per-mode state counts.",
            "- `preliminary_failure_correctness_descriptive_results.csv`: raw group accuracy and cluster-bootstrap intervals.",
            "- `preliminary_failure_correctness_glmm_vb_results.csv`: preliminary random-intercept GLMM approximation.",
            "- `preliminary_failure_correctness_gee_results.csv`: question-clustered GEE sensitivity results.",
            "- `preliminary_failure_correctness_combined_results.csv`: compact merged table.",
            "- `preliminary_failure_correctness_gee_forest_plot.png`: visual summary of GEE odds ratios.",
        ]
    )
    (output_dir / "preliminary_failure_correctness_analysis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "failure_correctness_case_manifest.csv.gz"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Run build_failure_correctness_case_manifest.py first: {manifest_path}"
        )
    manifest = pd.read_csv(manifest_path, low_memory=False)
    rng = np.random.default_rng(args.seed)
    descriptive_rows: list[dict[str, object]] = []
    glmm_rows: list[dict[str, object]] = []
    gee_rows: list[dict[str, object]] = []
    existing_glmm = None
    glmm_path = args.output_dir / "preliminary_failure_correctness_glmm_vb_results.csv"
    if args.reuse_existing_glmm:
        if not glmm_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing GLMM results: {glmm_path}")
        existing_glmm = pd.read_csv(glmm_path).set_index("failure_mode")
    for mode in FAILURE_MODES:
        print(f"Analyzing F-{mode.code}: {mode.short_name}", flush=True)
        data = prepare_mode_data(manifest, mode.code)
        descriptive_rows.append(
            descriptive_result(mode, data, args.bootstrap_replicates, rng)
        )
        if existing_glmm is None:
            glmm_rows.append(fit_glmm_vb(mode, data))
        else:
            glmm_rows.append(existing_glmm.loc[f"F-{mode.code}"].to_dict())
            glmm_rows[-1]["failure_mode"] = f"F-{mode.code}"
        gee_rows.append(fit_gee(mode, data))

    descriptive = pd.DataFrame(descriptive_rows)
    glmm = pd.DataFrame(glmm_rows)
    gee = apply_fdr(pd.DataFrame(gee_rows))
    descriptive.to_csv(
        args.output_dir / "preliminary_failure_correctness_descriptive_results.csv",
        index=False,
    )
    glmm.to_csv(glmm_path, index=False)
    gee.to_csv(
        args.output_dir / "preliminary_failure_correctness_gee_results.csv",
        index=False,
    )
    make_forest_plot(gee, args.output_dir)
    write_report(args.output_dir, descriptive, glmm, gee)
    metadata = {
        "analysis_status": "preliminary",
        "random_seed": args.seed,
        "cluster_bootstrap_replicates": args.bootstrap_replicates,
        "manifest": str(manifest_path),
        "primary_preliminary_mixed_model": "BinomialBayesMixedGLM variational Bayes",
        "sensitivity_model": "question-clustered GEE with independence working correlation and robust standard errors",
    }
    (args.output_dir / "preliminary_failure_correctness_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
