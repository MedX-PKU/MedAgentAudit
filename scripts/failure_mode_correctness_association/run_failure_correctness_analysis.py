#!/usr/bin/env python3
"""Run the case-level failure-mode and final-correctness analysis."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import patsy
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, root
from scipy.special import expit, logsumexp
from statsmodels.stats.multitest import multipletests

from failure_mode_schema import FAILURE_MODES


FIXED_TERMS = "failure_positive + C(dataset) + C(mas) + C(underlying_llm)"
FIXED_FORMULA = f"correctness ~ {FIXED_TERMS}"
KEY_COLUMNS = ["dataset", "qid", "mas", "underlying_llm"]


def parse_args() -> argparse.Namespace:
    default_output = Path(
        "/mnt/c/Users/LeiGu/Dropbox/[Preprint 2026] "
        "Auditing medical multi-agent AI reveals risks of false consensus/"
        "Preprint/analysis/failure_mode_correctness_association"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--quadrature-nodes", type=int, default=9)
    parser.add_argument("--sensitivity-quadrature-nodes", type=int, default=15)
    return parser.parse_args()


def prepare_mode_data(manifest: pd.DataFrame, code: str) -> pd.DataFrame:
    prefix = f"f_{code.replace('.', '_')}"
    data = manifest.loc[
        manifest[f"{prefix}_state"].isin(["positive", "negative"])
        & manifest["correctness"].notna()
    ].copy()
    data["failure_positive"] = data[f"{prefix}_label"].astype(int)
    data["correctness"] = data["correctness"].astype(int)
    return data


def raw_measures(data: pd.DataFrame) -> dict[str, float | int]:
    grouped = data.groupby("failure_positive")["correctness"].agg(["sum", "count"])
    values: dict[int, tuple[int, int]] = {}
    for exposure in (0, 1):
        values[exposure] = (
            (int(grouped.loc[exposure, "sum"]), int(grouped.loc[exposure, "count"]))
            if exposure in grouped.index
            else (0, 0)
        )
    neg_correct, neg_n = values[0]
    pos_correct, pos_n = values[1]
    pos_accuracy = pos_correct / pos_n if pos_n else math.nan
    neg_accuracy = neg_correct / neg_n if neg_n else math.nan
    risk_difference = pos_accuracy - neg_accuracy
    risk_ratio = pos_accuracy / neg_accuracy if neg_accuracy > 0 else math.nan
    pos_incorrect = pos_n - pos_correct
    neg_incorrect = neg_n - neg_correct
    counts = np.asarray([pos_correct, pos_incorrect, neg_correct, neg_incorrect], float)
    if np.any(counts == 0):
        counts += 0.5
    odds_ratio = (counts[0] * counts[3]) / (counts[1] * counts[2])
    return {
        "positive_n": pos_n,
        "positive_correct_n": pos_correct,
        "positive_accuracy": pos_accuracy,
        "negative_n": neg_n,
        "negative_correct_n": neg_correct,
        "negative_accuracy": neg_accuracy,
        "risk_difference": risk_difference,
        "risk_ratio": risk_ratio,
        "odds_ratio": float(odds_ratio),
    }


def cluster_bootstrap_intervals(
    data: pd.DataFrame, replicates: int, rng: np.random.Generator
) -> dict[str, float]:
    table = (
        data.groupby(["question_cluster", "failure_positive"])["correctness"]
        .agg(["sum", "count"])
        .unstack(fill_value=0)
    )
    clusters = table.index
    arrays = []
    for exposure in (0, 1):
        correct = (
            table[("sum", exposure)].to_numpy(float)
            if ("sum", exposure) in table.columns
            else np.zeros(len(clusters))
        )
        total = (
            table[("count", exposure)].to_numpy(float)
            if ("count", exposure) in table.columns
            else np.zeros(len(clusters))
        )
        arrays.extend([correct, total])
    weights = rng.multinomial(
        len(clusters), np.full(len(clusters), 1 / len(clusters)), size=replicates
    )
    neg_correct, neg_total, pos_correct, pos_total = [weights @ item for item in arrays]
    with np.errstate(divide="ignore", invalid="ignore"):
        neg_risk = neg_correct / neg_total
        pos_risk = pos_correct / pos_total
        rd = pos_risk - neg_risk
        rr = pos_risk / neg_risk
        odds_pos = pos_correct / (pos_total - pos_correct)
        odds_neg = neg_correct / (neg_total - neg_correct)
        odds_ratio = odds_pos / odds_neg

    def interval(values: np.ndarray) -> tuple[float, float]:
        valid = values[np.isfinite(values)]
        return tuple(np.quantile(valid, [0.025, 0.975])) if len(valid) else (math.nan, math.nan)

    rd_ci = interval(rd)
    rr_ci = interval(rr)
    or_ci = interval(odds_ratio)
    return {
        "risk_difference_ci_low": rd_ci[0],
        "risk_difference_ci_high": rd_ci[1],
        "risk_ratio_ci_low": rr_ci[0],
        "risk_ratio_ci_high": rr_ci[1],
        "odds_ratio_ci_low": or_ci[0],
        "odds_ratio_ci_high": or_ci[1],
    }


@dataclass
class GLMMData:
    y: np.ndarray
    x: np.ndarray
    names: list[str]
    group_index: np.ndarray
    starts: np.ndarray
    design_info: patsy.DesignInfo
    frame: pd.DataFrame


def make_glmm_data(data: pd.DataFrame, formula: str) -> GLMMData:
    y_frame, x_frame = patsy.dmatrices(formula, data, return_type="dataframe")
    order = np.argsort(data["question_cluster"].to_numpy(), kind="stable")
    sorted_frame = data.iloc[order].reset_index(drop=True)
    y = np.asarray(y_frame.iloc[order, 0], float)
    x = np.asarray(x_frame.iloc[order], float)
    groups = sorted_frame["question_cluster"].to_numpy()
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    group_index = np.cumsum(np.r_[0, groups[1:] != groups[:-1]])
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Fixed-effect design matrix is rank deficient")
    return GLMMData(
        y=y,
        x=x,
        names=list(x_frame.columns),
        group_index=group_index,
        starts=starts,
        design_info=x_frame.design_info,
        frame=sorted_frame,
    )


def glmm_objective(
    model_data: GLMMData, nodes: np.ndarray, weights: np.ndarray
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Return an adaptive Gauss--Hermite marginal likelihood and score."""
    y = model_data.y
    x = model_data.x
    starts = model_data.starts
    group_index = model_data.group_index
    log_weights = np.log(weights)
    group_count = len(starts)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        beta = parameters[:-1]
        sigma = math.exp(float(parameters[-1]))
        sigma_squared = sigma * sigma
        fixed_predictor = x @ beta

        # The conditional posterior is strictly concave. Vectorized bisection is
        # used instead of undamped Newton updates because all-correct or all-
        # incorrect question clusters can make Newton steps oscillate.
        group_sizes = np.diff(np.r_[starts, len(y)])
        bound = group_sizes * sigma_squared + 1.0
        lower = -bound
        upper = bound
        for _ in range(60):
            modes = (lower + upper) / 2
            eta_at_mode = fixed_predictor + modes[group_index]
            probability_at_mode = expit(eta_at_mode)
            score = (
                np.add.reduceat(y - probability_at_mode, starts)
                - modes / sigma_squared
            )
            lower = np.where(score > 0, modes, lower)
            upper = np.where(score > 0, upper, modes)
        modes = (lower + upper) / 2

        eta_at_mode = fixed_predictor + modes[group_index]
        probability_at_mode = expit(eta_at_mode)
        curvature = (
            np.add.reduceat(
                probability_at_mode * (1 - probability_at_mode), starts
            )
            + 1 / sigma_squared
        )
        scales = np.sqrt(2 / curvature)
        random_intercepts = modes[:, None] + scales[:, None] * nodes[None, :]
        eta = fixed_predictor[:, None] + random_intercepts[group_index]
        probabilities = expit(eta)
        observation_ll = y[:, None] * eta - np.logaddexp(0.0, eta)
        conditional_ll = np.add.reduceat(observation_ll, starts, axis=0)
        log_prior = (
            -0.5 * (random_intercepts / sigma) ** 2
            - math.log(sigma)
            - 0.5 * math.log(2 * math.pi)
        )
        log_jacobian = np.log(scales)[:, None]
        combined = (
            conditional_ll
            + log_prior
            + log_jacobian
            + log_weights[None, :]
            + nodes[None, :] ** 2
        )
        marginal_ll = logsumexp(combined, axis=1)
        posterior_weights = np.exp(combined - marginal_ll[:, None])
        weighted_probabilities = np.sum(
            posterior_weights[group_index] * probabilities, axis=1
        )
        score_beta = x.T @ (y - weighted_probabilities)
        prior_scale_score = (random_intercepts / sigma) ** 2 - 1
        score_log_sigma = np.sum(posterior_weights * prior_scale_score)
        score = np.r_[score_beta, score_log_sigma]
        return -float(marginal_ll.sum()), -score

    return objective


def finite_difference_hessian(
    jacobian: Callable[[np.ndarray], np.ndarray], parameters: np.ndarray
) -> np.ndarray:
    dimension = len(parameters)
    result = np.empty((dimension, dimension), float)
    for column in range(dimension):
        step = 1e-4 * (1 + abs(parameters[column]))
        upper = parameters.copy()
        lower = parameters.copy()
        upper[column] += step
        lower[column] -= step
        result[:, column] = (jacobian(upper) - jacobian(lower)) / (2 * step)
    return (result + result.T) / 2


def invert_information(information: np.ndarray) -> tuple[np.ndarray, float, str]:
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    condition = float(np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues)))
    if np.min(eigenvalues) <= 1e-8:
        inverse = np.linalg.pinv(information, rcond=1e-10)
        return inverse, condition, "pseudo_inverse"
    return np.linalg.inv(information), condition, "inverse"


def marginal_difference(
    parameters: np.ndarray,
    design_info: patsy.DesignInfo,
    frame: pd.DataFrame,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    beta = parameters[:-1]
    sigma = math.exp(float(parameters[-1]))
    probabilities = []
    for exposure in (0, 1):
        counterfactual = frame.copy()
        counterfactual["failure_positive"] = exposure
        if "failure_x_vqa" in counterfactual:
            counterfactual["failure_x_vqa"] = exposure * counterfactual["modality"].eq("VQA").astype(int)
        if "failure_x_medagent" in counterfactual:
            counterfactual["failure_x_medagent"] = exposure * counterfactual["mas"].eq("medagent").astype(int)
        design = np.asarray(
            patsy.build_design_matrices([design_info], counterfactual)[0], float
        )
        eta = (design @ beta)[:, None] + math.sqrt(2) * sigma * nodes[None, :]
        probabilities.append(float(np.mean(expit(eta) @ (weights / math.sqrt(math.pi)))))
    return probabilities[1] - probabilities[0]


def numerical_gradient(function: Callable[[np.ndarray], float], parameters: np.ndarray) -> np.ndarray:
    gradient = np.empty(len(parameters), float)
    for index in range(len(parameters)):
        step = 1e-5 * (1 + abs(parameters[index]))
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        gradient[index] = (function(upper) - function(lower)) / (2 * step)
    return gradient


def initial_parameters(model_data: GLMMData) -> np.ndarray:
    try:
        fitted = sm.GLM(
            model_data.y, model_data.x, family=sm.families.Binomial()
        ).fit(maxiter=200)
        beta = np.asarray(fitted.params, float)
    except Exception:
        beta = np.zeros(model_data.x.shape[1])
    return np.r_[beta, math.log(0.5)]


def fit_frequentist_glmm(
    mode,
    data: pd.DataFrame,
    formula: str,
    quadrature_nodes: int,
    sensitivity_nodes: int,
    target_term: str = "failure_positive",
) -> dict[str, object]:
    base = {
        "failure_mode": f"F-{mode.code}",
        "failure_mode_name": mode.short_name,
        "model": "maximum-likelihood logistic GLMM with question random intercept",
        "formula": formula,
        "target_term": target_term,
        "n": len(data),
        "question_clusters": data["question_cluster"].nunique(),
        "status": "not_run",
    }
    try:
        model_data = make_glmm_data(data, formula)
        nodes, weights = hermgauss(quadrature_nodes)
        objective = glmm_objective(model_data, nodes, weights)
        start = initial_parameters(model_data)
        bounds = [(None, None)] * (len(start) - 1) + [(-7, 3)]
        fitted = minimize(
            lambda p: objective(p)[0],
            start,
            jac=lambda p: objective(p)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-6, "maxls": 50},
        )
        # L-BFGS-B can stop on a negligible likelihood change while a small
        # residual score remains. A local root solve removes that residual and
        # makes the reported score norm an explicit convergence diagnostic.
        refined = root(
            lambda p: objective(p)[1],
            fitted.x,
            method="hybr",
            options={"xtol": 1e-9, "maxfev": 1000},
        )
        parameters = fitted.x
        if (
            refined.success
            and -7 <= refined.x[-1] <= 3
            and np.max(np.abs(objective(refined.x)[1]))
            < np.max(np.abs(objective(parameters)[1]))
        ):
            parameters = refined.x
        gradient_norm = float(np.max(np.abs(objective(parameters)[1])))
        target_index = model_data.names.index(target_term)
        hessian = finite_difference_hessian(lambda p: objective(p)[1], parameters)
        covariance, condition_number, covariance_method = invert_information(hessian)
        standard_error = math.sqrt(float(covariance[target_index, target_index]))
        beta = float(parameters[target_index])
        z_value = beta / standard_error
        pd_function = lambda p: marginal_difference(
            p, model_data.design_info, model_data.frame, nodes, weights
        )
        probability_difference = pd_function(parameters)
        pd_gradient = numerical_gradient(pd_function, parameters)
        pd_variance = float(pd_gradient @ covariance @ pd_gradient)
        pd_standard_error = math.sqrt(max(pd_variance, 0.0))

        sensitivity_x = parameters.copy()
        sensitivity_success = True
        sensitivity_message = "same_as_primary"
        if sensitivity_nodes != quadrature_nodes:
            nodes_s, weights_s = hermgauss(sensitivity_nodes)
            objective_s = glmm_objective(model_data, nodes_s, weights_s)
            fitted_s = minimize(
                lambda p: objective_s(p)[0],
                parameters,
                jac=lambda p: objective_s(p)[1],
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-6, "maxls": 50},
            )
            refined_s = root(
                lambda p: objective_s(p)[1],
                fitted_s.x,
                method="hybr",
                options={"xtol": 1e-9, "maxfev": 1000},
            )
            sensitivity_x = (
                refined_s.x
                if refined_s.success and -7 <= refined_s.x[-1] <= 3
                else fitted_s.x
            )
            sensitivity_success = bool(fitted_s.success)
            sensitivity_message = (
                f"optimizer: {fitted_s.message}; score refinement: {refined_s.message}"
            )
        sensitivity_beta = float(sensitivity_x[target_index])
        base.update(
            status="completed"
            if fitted.success and gradient_norm < 1e-4
            else "score_not_converged",
            optimizer_success=bool(fitted.success),
            optimizer_message=(
                f"optimizer: {fitted.message}; score refinement: {refined.message}"
            ),
            iterations=int(fitted.nit + refined.nfev),
            log_likelihood=-float(objective(parameters)[0]),
            fixed_effect_count=model_data.x.shape[1],
            beta=beta,
            standard_error=standard_error,
            z_value=z_value,
            p_value=float(2 * st.norm.sf(abs(z_value))),
            adjusted_correctness_or=math.exp(beta),
            ci_low=math.exp(beta - 1.96 * standard_error),
            ci_high=math.exp(beta + 1.96 * standard_error),
            adjusted_probability_difference=probability_difference,
            adjusted_probability_difference_se=pd_standard_error,
            adjusted_probability_difference_ci_low=probability_difference - 1.96 * pd_standard_error,
            adjusted_probability_difference_ci_high=probability_difference + 1.96 * pd_standard_error,
            question_random_intercept_sd=math.exp(float(parameters[-1])),
            gradient_infinity_norm=gradient_norm,
            hessian_condition_number=condition_number,
            covariance_method=covariance_method,
            quadrature_nodes=quadrature_nodes,
            sensitivity_quadrature_nodes=sensitivity_nodes,
            sensitivity_optimizer_success=sensitivity_success,
            sensitivity_optimizer_message=sensitivity_message,
            sensitivity_beta=sensitivity_beta,
            sensitivity_adjusted_correctness_or=math.exp(sensitivity_beta),
            quadrature_beta_difference=sensitivity_beta - beta,
        )
    except Exception as exc:
        base.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return base


def fit_gee(mode, data: pd.DataFrame, formula: str = FIXED_FORMULA, target_term: str = "failure_positive") -> dict[str, object]:
    base = {
        "failure_mode": f"F-{mode.code}",
        "failure_mode_name": mode.short_name,
        "formula": formula,
        "target_term": target_term,
        "n": len(data),
        "question_clusters": data["question_cluster"].nunique(),
        "status": "not_run",
    }
    try:
        fitted = smf.gee(
            formula,
            groups="question_cluster",
            data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        ).fit(maxiter=300)
        beta = float(fitted.params[target_term])
        standard_error = float(fitted.bse[target_term])
        positive = data.copy()
        negative = data.copy()
        positive["failure_positive"] = 1
        negative["failure_positive"] = 0
        for frame, exposure in ((positive, 1), (negative, 0)):
            if "failure_x_vqa" in frame:
                frame["failure_x_vqa"] = exposure * frame["modality"].eq("VQA").astype(int)
            if "failure_x_medagent" in frame:
                frame["failure_x_medagent"] = exposure * frame["mas"].eq("medagent").astype(int)
        probability_difference = float(fitted.predict(positive).mean() - fitted.predict(negative).mean())
        base.update(
            status="completed" if fitted.converged else "not_converged",
            beta=beta,
            robust_se=standard_error,
            z_value=float(fitted.tvalues[target_term]),
            p_value=float(fitted.pvalues[target_term]),
            adjusted_correctness_or=math.exp(beta),
            ci_low=math.exp(beta - 1.96 * standard_error),
            ci_high=math.exp(beta + 1.96 * standard_error),
            standardized_probability_difference=probability_difference,
            converged=bool(fitted.converged),
        )
    except Exception as exc:
        base.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return base


def stratified_rows(mode, data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variable in ("dataset", "mas", "underlying_llm", "modality"):
        for level, subset in data.groupby(variable, observed=True):
            result = raw_measures(subset)
            rows.append(
                {
                    "failure_mode": f"F-{mode.code}",
                    "failure_mode_name": mode.short_name,
                    "stratification_variable": variable,
                    "stratum": level,
                    "eligible_valid_n": len(subset),
                    "question_clusters": subset["question_cluster"].nunique(),
                    **result,
                }
            )
    return rows


def flow_rows(manifest: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in FAILURE_MODES:
        prefix = f"f_{mode.code.replace('.', '_')}"
        counts = manifest[f"{prefix}_state"].value_counts()
        for state in ("positive", "negative", "unknown", "not_applicable", "not_audited"):
            rows.append(
                {
                    "failure_mode": f"F-{mode.code}",
                    "failure_mode_name": mode.short_name,
                    "state": state,
                    "count": int(counts.get(state, 0)),
                }
            )
    return rows


def apply_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["fdr_bh_p_value"] = np.nan
    mask = frame["status"].eq("completed") & frame["p_value"].notna()
    if mask.any():
        frame.loc[mask, "fdr_bh_p_value"] = multipletests(
            frame.loc[mask, "p_value"].astype(float), method="fdr_bh"
        )[1]
    return frame


def write_report(
    output_dir: Path,
    descriptive: pd.DataFrame,
    glmm: pd.DataFrame,
    gee: pd.DataFrame,
    interactions: pd.DataFrame,
) -> None:
    combined = descriptive.merge(
        glmm[[
            "failure_mode", "status", "adjusted_correctness_or", "ci_low", "ci_high",
            "adjusted_probability_difference", "adjusted_probability_difference_ci_low",
            "adjusted_probability_difference_ci_high", "p_value", "fdr_bh_p_value",
            "question_random_intercept_sd", "gradient_infinity_norm", "quadrature_beta_difference",
        ]].rename(columns={"status": "glmm_status"}), on="failure_mode", how="left"
    ).merge(
        gee[["failure_mode", "status", "adjusted_correctness_or", "ci_low", "ci_high", "standardized_probability_difference", "p_value"]]
        .rename(columns={"status": "gee_status"}), on="failure_mode", how="left", suffixes=("_glmm", "_gee")
    )
    combined.to_csv(output_dir / "failure_correctness_combined_results.csv", index=False)

    def fmt(value: object, digits: int = 3) -> str:
        if pd.isna(value):
            return "NA"
        numeric = float(value)
        if numeric != 0 and abs(numeric) < 10 ** (-digits):
            return f"{numeric:.2e}"
        return f"{numeric:.{digits}f}"

    lines = [
        "# Failure Mode–Correctness Analysis",
        "",
        "This analysis uses frozen MAS and automated-auditor logs. No MAS, underlying LLM, or auditor was rerun.",
        "",
        "## Primary results",
        "",
        "| Mode | N | Positive accuracy | Negative accuracy | Raw risk difference | Adjusted OR (95% CI) | Adjusted probability difference (95% CI) | FDR P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined.itertuples():
        adjusted_or = f"{fmt(row.adjusted_correctness_or_glmm)} ({fmt(row.ci_low_glmm)} to {fmt(row.ci_high_glmm)})"
        adjusted_pd = f"{fmt(row.adjusted_probability_difference)} ({fmt(row.adjusted_probability_difference_ci_low)} to {fmt(row.adjusted_probability_difference_ci_high)})"
        lines.append(
            f"| {row.failure_mode} | {int(row.eligible_valid_n)} | {fmt(row.positive_accuracy)} | {fmt(row.negative_accuracy)} | {fmt(row.risk_difference)} | {adjusted_or} | {adjusted_pd} | {fmt(row.fdr_bh_p_value)} |"
        )
    lines.extend([
        "",
        "## Interpretation boundaries",
        "",
        "- Odds ratios below 1 and negative probability differences indicate lower benchmark-answer correctness in failure-positive cases after adjustment.",
        "- These estimates are observational associations and do not establish that a failure mode caused an incorrect answer.",
        "- F-2.2.1 uses only the original case-level auditor label. Intermediate answers and ground truth do not enter the exposure definition; final correctness is the outcome.",
        "- F-2.2.1 is near constant and is retained as a descriptive interaction pattern. Its adjusted estimate is a sensitivity result, not primary support for clinical risk.",
        "- F-3.1.1 does not use benchmark ground truth in its label, but its definition requires the auditor to judge minority correctness and current-decision incorrectness. Its association is therefore definition-aligned rather than a fully external validation.",
        "- F-3.2.1 applies only to 545 auditable multi-round cases; framework-specific estimates with very few positives are not interpreted.",
        "- GEE estimates, stratified raw results, QA/VQA and MAS interactions, and quadrature diagnostics are provided as separate files.",
        "",
        "## Interaction checks",
        "",
    ])
    if interactions.empty:
        lines.append("No interaction model completed.")
    else:
        for row in interactions.itertuples():
            lines.append(
                f"- {row.failure_mode}, {row.comparison}: interaction OR {fmt(row.adjusted_correctness_or)} (95% CI {fmt(row.ci_low)} to {fmt(row.ci_high)}), P={fmt(row.p_value)}."
            )
    (output_dir / "failure_correctness_analysis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "failure_correctness_case_manifest.csv.gz"
    if not manifest_path.exists():
        raise FileNotFoundError("Build the case manifest first")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    if len(manifest) != 14370 or manifest.duplicated(KEY_COLUMNS).any():
        raise ValueError("Final manifest does not contain 14,370 unique cases")
    manifest.to_csv(
        args.output_dir / "failure_correctness_analysis_manifest.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(flow_rows(manifest)).to_csv(
        args.output_dir / "failure_mode_flow_table.csv", index=False
    )

    rng = np.random.default_rng(args.seed)
    descriptive_rows: list[dict[str, object]] = []
    stratified: list[dict[str, object]] = []
    glmm_rows: list[dict[str, object]] = []
    gee_rows: list[dict[str, object]] = []
    interaction_rows: list[dict[str, object]] = []
    stratified_model_rows: list[dict[str, object]] = []
    interaction_gee_rows: list[dict[str, object]] = []
    stratified_gee_rows: list[dict[str, object]] = []

    for mode in FAILURE_MODES:
        print(f"Analyzing F-{mode.code}: {mode.short_name}", flush=True)
        data = prepare_mode_data(manifest, mode.code)
        raw = raw_measures(data)
        descriptive_rows.append({
            "failure_mode": f"F-{mode.code}",
            "failure_mode_name": mode.short_name,
            "eligible_valid_n": len(data),
            "question_clusters": data["question_cluster"].nunique(),
            **raw,
            **cluster_bootstrap_intervals(data, args.bootstrap_replicates, rng),
        })
        stratified.extend(stratified_rows(mode, data))
        glmm_rows.append(fit_frequentist_glmm(
            mode, data, FIXED_FORMULA, args.quadrature_nodes,
            args.sensitivity_quadrature_nodes,
        ))
        gee_rows.append(fit_gee(mode, data))

        if mode.code == "1.2.1":
            interaction_data = data.copy()
            interaction_data["failure_x_vqa"] = interaction_data["failure_positive"] * interaction_data["modality"].eq("VQA").astype(int)
            formula = FIXED_FORMULA + " + failure_x_vqa"
            result = fit_frequentist_glmm(
                mode, interaction_data, formula, args.quadrature_nodes,
                args.sensitivity_quadrature_nodes, "failure_x_vqa",
            )
            result["comparison"] = "VQA versus QA failure association"
            interaction_rows.append(result)
            result = fit_gee(mode, interaction_data, formula, "failure_x_vqa")
            result["comparison"] = "VQA versus QA failure association"
            interaction_gee_rows.append(result)
            for level, subset in data.groupby("modality"):
                result = fit_frequentist_glmm(
                    mode, subset, FIXED_FORMULA, args.quadrature_nodes,
                    args.sensitivity_quadrature_nodes,
                )
                result["stratification_variable"] = "modality"
                result["stratum"] = level
                stratified_model_rows.append(result)
                result = fit_gee(mode, subset)
                result["stratification_variable"] = "modality"
                result["stratum"] = level
                stratified_gee_rows.append(result)

        if mode.code == "2.1.1":
            interaction_data = data.copy()
            interaction_data["failure_x_medagent"] = interaction_data["failure_positive"] * interaction_data["mas"].eq("medagent").astype(int)
            formula = FIXED_FORMULA + " + failure_x_medagent"
            result = fit_frequentist_glmm(
                mode, interaction_data, formula, args.quadrature_nodes,
                args.sensitivity_quadrature_nodes, "failure_x_medagent",
            )
            result["comparison"] = "MedAgents versus MDAgents failure association"
            interaction_rows.append(result)
            result = fit_gee(mode, interaction_data, formula, "failure_x_medagent")
            result["comparison"] = "MedAgents versus MDAgents failure association"
            interaction_gee_rows.append(result)
            for level, subset in data.groupby("mas"):
                result = fit_frequentist_glmm(
                    mode, subset, FIXED_FORMULA, args.quadrature_nodes,
                    args.sensitivity_quadrature_nodes,
                )
                result["stratification_variable"] = "mas"
                result["stratum"] = level
                stratified_model_rows.append(result)
                result = fit_gee(mode, subset)
                result["stratification_variable"] = "mas"
                result["stratum"] = level
                stratified_gee_rows.append(result)

    descriptive = pd.DataFrame(descriptive_rows)
    glmm = apply_fdr(pd.DataFrame(glmm_rows))
    gee = apply_fdr(pd.DataFrame(gee_rows))
    interactions = pd.DataFrame(interaction_rows)
    descriptive.to_csv(args.output_dir / "failure_correctness_descriptive_results.csv", index=False)
    pd.DataFrame(stratified).to_csv(args.output_dir / "failure_correctness_stratified_results.csv", index=False)
    glmm.to_csv(args.output_dir / "failure_correctness_glmm_results.csv", index=False)
    gee.to_csv(args.output_dir / "failure_correctness_gee_results.csv", index=False)
    interactions.to_csv(args.output_dir / "failure_correctness_interaction_results.csv", index=False)
    pd.DataFrame(stratified_model_rows).to_csv(args.output_dir / "failure_correctness_stratified_glmm_results.csv", index=False)
    pd.DataFrame(interaction_gee_rows).to_csv(args.output_dir / "failure_correctness_interaction_gee_results.csv", index=False)
    pd.DataFrame(stratified_gee_rows).to_csv(args.output_dir / "failure_correctness_stratified_gee_results.csv", index=False)
    write_report(args.output_dir, descriptive, glmm, gee, interactions)
    metadata = {
        "analysis_status": "complete",
        "analysis_definition": "original case-level auditor labels",
        "source": "frozen existing MAS and automated-auditor logs",
        "new_model_or_auditor_runs": False,
        "random_seed": args.seed,
        "cluster_bootstrap_replicates": args.bootstrap_replicates,
        "primary_model": "maximum-likelihood logistic GLMM with dataset:question_ID random intercept",
        "quadrature_nodes": args.quadrature_nodes,
        "sensitivity_quadrature_nodes": args.sensitivity_quadrature_nodes,
        "sensitivity_model": "question-clustered GEE with robust standard errors",
        "manifest": str(manifest_path),
    }
    (args.output_dir / "failure_correctness_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
