#!/usr/bin/env python3
"""Create stratified summaries, quality checks, and a Chinese interpretation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from failure_mode_schema import FAILURE_MODES


def parse_args() -> argparse.Namespace:
    default_output = Path(
        "/mnt/c/Users/LeiGu/Dropbox/[Preprint 2026] "
        "Auditing medical multi-agent AI reveals risks of false consensus/"
        "Preprint/analysis/failure_mode_correctness_association"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def mode_data(manifest: pd.DataFrame, code: str) -> pd.DataFrame:
    prefix = f"f_{code.replace('.', '_')}"
    data = manifest.loc[
        manifest[f"{prefix}_state"].isin(["positive", "negative"])
        & manifest["correctness"].notna()
    ].copy()
    data["failure_positive"] = data[f"{prefix}_label"].astype(int)
    data["correctness"] = data["correctness"].astype(int)
    return data


def build_stratified_results(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    factors = {
        "dataset": "dataset_display",
        "MAS": "mas_display",
        "underlying_LLM": "underlying_llm_display",
        "modality": "modality",
    }
    for mode in FAILURE_MODES:
        data = mode_data(manifest, mode.code)
        for factor, column in factors.items():
            for level, group in data.groupby(column, dropna=False):
                positive = group.loc[group["failure_positive"] == 1]
                negative = group.loc[group["failure_positive"] == 0]
                positive_accuracy = (
                    positive["correctness"].mean() if len(positive) else np.nan
                )
                negative_accuracy = (
                    negative["correctness"].mean() if len(negative) else np.nan
                )
                rows.append(
                    {
                        "failure_mode": f"F-{mode.code}",
                        "failure_mode_name": mode.short_name,
                        "stratification_factor": factor,
                        "stratum": level,
                        "eligible_n": len(group),
                        "positive_n": len(positive),
                        "positive_correct_n": int(positive["correctness"].sum()),
                        "positive_accuracy": positive_accuracy,
                        "negative_n": len(negative),
                        "negative_correct_n": int(negative["correctness"].sum()),
                        "negative_accuracy": negative_accuracy,
                        "accuracy_difference": positive_accuracy - negative_accuracy
                        if len(positive) and len(negative)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def quality_checks(
    manifest: pd.DataFrame, glmm: pd.DataFrame, gee: pd.DataFrame
) -> dict[str, object]:
    repeat_distribution = (
        manifest.groupby("question_cluster").size().value_counts().sort_index()
    )
    mode_checks: dict[str, object] = {}
    for mode in FAILURE_MODES:
        prefix = f"f_{mode.code.replace('.', '_')}"
        counts = manifest[f"{prefix}_state"].value_counts().to_dict()
        mode_checks[f"F-{mode.code}"] = {
            "states": {key: int(value) for key, value in counts.items()},
            "state_total": int(sum(counts.values())),
            "state_total_equals_manifest": int(sum(counts.values())) == len(manifest),
        }
    completed_glmm = glmm.loc[glmm["status"] == "completed"]
    completed_gee = gee.loc[gee["status"] == "completed"]
    return {
        "manifest_rows": len(manifest),
        "unique_case_key_rows": int(
            manifest[["dataset", "qid", "mas", "underlying_llm"]]
            .drop_duplicates()
            .shape[0]
        ),
        "unique_question_clusters": int(manifest["question_cluster"].nunique()),
        "question_repeat_distribution": {
            str(key): int(value) for key, value in repeat_distribution.items()
        },
        "missing_correctness": int(manifest["correctness"].isna().sum()),
        "glmm_completed": int(len(completed_glmm)),
        "glmm_optimizer_success": int(
            completed_glmm["optimizer_success"].eq(True).sum()
        ),
        "gee_completed": int(len(completed_gee)),
        "gee_converged": int(
            completed_gee["converged"].eq(True).sum()
        ),
        "completed_glmm_finite_estimates": bool(
            np.isfinite(completed_glmm["adjusted_correctness_or"]).all()
        ),
        "completed_gee_finite_estimates": bool(
            np.isfinite(completed_gee["adjusted_correctness_or"]).all()
        ),
        "failure_modes": mode_checks,
    }


def fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_p(value: object) -> str:
    if pd.isna(value):
        return "NA"
    number = float(value)
    return f"{number:.2e}" if number < 0.001 else f"{number:.3f}"


def direction_summary(stratified: pd.DataFrame, code: str) -> str:
    subset = stratified.loc[
        (stratified["failure_mode"] == code)
        & stratified["accuracy_difference"].notna()
        & (stratified["positive_n"] >= 10)
        & (stratified["negative_n"] >= 10)
    ]
    if subset.empty:
        return "没有同时含至少10个 positive 和10个 negative 的可比较 strata"
    negative = int((subset["accuracy_difference"] < 0).sum())
    positive = int((subset["accuracy_difference"] > 0).sum())
    zero = int((subset["accuracy_difference"] == 0).sum())
    return f"{negative}/{len(subset)} 个可比较 strata 为负向，{positive} 个为正向，{zero} 个为零"


def write_chinese_report(
    output_dir: Path,
    combined: pd.DataFrame,
    stratified: pd.DataFrame,
    checks: dict[str, object],
) -> None:
    by_mode = combined.set_index("failure_mode")
    lines = [
        "# Failure Mode 与 Benchmark Correctness 初步分析报告",
        "",
        "## 1. 分析定位",
        "",
        "本报告使用冻结的既有 MAS logs 和 automated-auditor outputs，没有重新运行 MAS、LLM 或 auditor。结果用于判断 failure–correctness joint analysis 的初步方向，不是可直接复制到稿件中的最终模型结果。正式稿仍需完成修订后的 F-2.2.1 标签，并用 protocol 规定的 frequentist mixed-effects logistic regression 复核。",
        "",
        "## 2. 数据质量",
        "",
        f"- Case manifest 包含 {checks['manifest_rows']:,} 行，唯一 case keys 也是 {checks['unique_case_key_rows']:,}，没有重复 case key。",
        f"- 共 {checks['unique_question_clusters']} 道不同 benchmark questions；595 道各重复24次，五道各重复18次。",
        f"- 无法确定 final correctness 的 cases：{checks['missing_correctness']}。",
        f"- 九个可建模模式的 VB-GLMM optimizer 均成功：{checks['glmm_optimizer_success']}/{checks['glmm_completed']}。",
        f"- 九个可建模模式的 question-clustered GEE 均收敛：{checks['gee_converged']}/{checks['gee_completed']}。",
        "- Original repetition of initial views (F-2.2.1) 因接近常量而未进入 adjusted models。",
        "",
        "## 3. 主要结果",
        "",
        "Outcome 编码为 correctness = 1，因此 OR < 1 或 probability difference < 0 表示 failure-positive cases 的 benchmark correctness 较低。",
        "",
        "| Failure mode | N | Positive accuracy | Negative accuracy | Raw difference | VB-GLMM OR | GEE OR (95% CI) | GEE adjusted difference | FDR P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined.itertuples():
        gee_interval = (
            "NA"
            if pd.isna(row.adjusted_correctness_or_gee)
            else f"{row.adjusted_correctness_or_gee:.3f} ({row.ci_low:.3f}--{row.ci_high:.3f})"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row.failure_mode,
                    f"{int(row.eligible_valid_n):,}",
                    fmt(row.positive_accuracy),
                    fmt(row.negative_accuracy),
                    fmt(row.unadjusted_accuracy_difference),
                    fmt(row.adjusted_correctness_or_glmm),
                    gee_interval,
                    fmt(row.standardized_probability_difference),
                    fmt_p(row.fdr_bh_p_value),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 4. 当前最稳妥的判断",
            "",
            "### 4.1 三种分析方向一致且与较低 correctness 相关",
            "",
            "以下模式在 raw accuracy difference、VB-GLMM 和 cluster-robust GEE 中方向一致：",
            "",
            "- factual hallucinations (F-1.1.1)：positive accuracy 55.1%，negative accuracy 76.3%；GEE adjusted OR 0.251，adjusted probability difference -25.4 percentage points。",
            "- failure to activate specialist knowledge (F-2.1.2)：68.6% vs 75.8%；GEE adjusted OR 0.666，adjusted difference -6.9 points。",
            "- unresolved conflicts (F-2.2.2)：57.5% vs 73.6%；GEE adjusted OR 0.542，adjusted difference -11.3 points。",
            "- authority bias (F-3.1.2)：61.8% vs 76.9%；GEE adjusted OR 0.360，adjusted difference -18.0 points。",
            "- contradiction neglect (F-3.1.3)：53.0% vs 73.7%；GEE adjusted OR 0.546，adjusted difference -11.1 points。",
            "",
            "这些结果支持将上述模式描述为与 benchmark answer correctness 存在负向统计关联，但不能写成这些模式导致错误。",
            "",
            "### 4.2 当前没有一致关联的模式",
            "",
            "- modality neglect (F-1.2.1)：raw difference 为 +1.4 points，question-cluster bootstrap interval 跨0；VB-GLMM OR 为0.746，但 GEE OR为0.955（95% CI 0.615--1.481，FDR P = 0.836）。不同方法不一致，当前不能声称它与 final correctness 存在稳定关联。",
            "- role-task mismatch (F-2.1.1)：70.1% vs 72.0%；VB-GLMM OR 0.954，GEE OR 0.956（95% CI 0.679--1.345，FDR P = 0.836）。当前结果接近无关联。",
            "",
            "### 4.3 不能按普通模式解释的结果",
            "",
            "- Original repetition of initial views (F-2.2.1)：positive 组 accuracy 反而更高（72.3% vs 58.7%），但 negative 只有155例。该结果说明原标签混入大量正确意见重申，也可能受到 workflow selection 影响；不能解释为重复具有保护作用。必须完成基于 intermediate-answer correctness 的修订标签后再建模。",
            "- minority suppression (F-3.1.1)：positive accuracy 27.4%，negative accuracy 75.0%，关联最强；但标签定义本身要求正确 minority view 被错误 consensus 覆盖，因此包含 correctness information。该结果部分由定义决定，不能作为独立验证。",
            "- self-contradiction across rounds (F-3.2.1)：38.6% vs 53.8%，GEE OR 0.315（95% CI 0.183--0.544），但仅有545例、315道题。方向为负，估计需要在最终 frequentist GLMM 和其他敏感性分析中复核。",
            "",
            "## 5. 分层方向核查",
            "",
        ]
    )
    for mode in FAILURE_MODES:
        lines.append(
            f"- F-{mode.code} {mode.short_name}：{direction_summary(stratified, f'F-{mode.code}')}。"
        )
    lines.extend(
        [
            "",
            "这里的 strata 汇总合并了 dataset、MAS、underlying LLM 和 modality 四类分层，仅用于发现明显方向冲突；完整分层分子和分母见 CSV。",
            "",
            "## 6. 当前不能写入稿件的内容",
            "",
            "1. 不能把 VB-GLMM 的 approximate posterior interval 写成 protocol 预定的 frequentist 95% confidence interval。",
            "2. 不能把 association 写成 failure mode 导致 final error。",
            "3. 不能把 benchmark correctness 改写为患者结局或真实临床安全性。",
            "4. 不能报告 original F-2.2.1 的 adjusted effect；必须先生成修订标签。",
            "5. 不能把 F-3.1.1 的强关联当作独立的 taxonomy validation。",
            "6. 不能因 F-1.2.1 或 F-2.1.1 未显示稳定关联而事后删除模式；应如实报告其 prevalence 与 outcome association 回答不同问题。",
            "",
            "## 7. 下一步",
            "",
            "1. 完成 F-2.2.1 intermediate-answer parser、coverage audit 和修订标签；",
            "2. 使用 frequentist random-intercept logistic GLMM 复核九种现有模式和修订后的 F-2.2.1；",
            "3. 对 question-level random effect 与 failure exposure 可能相关的问题增加 question fixed-effects 或 Mundlak sensitivity analysis；",
            "4. 检查各模式在 dataset、MAS、LLM 和 QA/VQA 中的异质性；",
            "5. 将最终 effect sizes、intervals 和有效分母映射回正文结论。",
        ]
    )
    (output_dir / "preliminary_failure_correctness_interpretation_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(
        args.output_dir / "failure_correctness_case_manifest.csv.gz", low_memory=False
    )
    combined = pd.read_csv(
        args.output_dir / "preliminary_failure_correctness_combined_results.csv"
    )
    glmm = pd.read_csv(
        args.output_dir / "preliminary_failure_correctness_glmm_vb_results.csv"
    )
    gee = pd.read_csv(
        args.output_dir / "preliminary_failure_correctness_gee_results.csv"
    )
    stratified = build_stratified_results(manifest)
    stratified.to_csv(
        args.output_dir
        / "preliminary_failure_correctness_stratified_descriptive_results.csv",
        index=False,
    )
    checks = quality_checks(manifest, glmm, gee)
    (args.output_dir / "preliminary_failure_correctness_quality_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_chinese_report(args.output_dir, combined, stratified, checks)


if __name__ == "__main__":
    main()
