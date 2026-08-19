#!/usr/bin/env python3
"""Plot the phase 2 F-2.1.2 QA/VQA round-step trajectory.

The output is a vector PDF with editable text and paths for placement in the
phase 2 Adobe Illustrator composition.

Run from the code repository root:
    uv run python scripts/plot_phase2_f212_qa_vqa_trajectory.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
import numpy as np

from audit_drawing import collect_group_stage_stats, process_audit_data_to_df


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "logs" / "audit_results" / "20260302"
DEFAULT_OUTPUT_PATH = Path(
    "/mnt/c/Users/LeiGu/Dropbox/[Preprint 2026] "
    "Auditing medical multi-agent AI reveals risks of false consensus/"
    "Preprint/figure/figure_in_body_text/phase2/subfigure/"
    "phase2_panel_e_2_1_2_qa_vqa_trajectory.pdf"
)

FAILURE_MODE = "2.1.2"
QA_DATASETS = {"MedQA", "PubMedQA", "MedXpertQA"}
VQA_DATASETS = {"PathVQA", "VQA-RAD", "SLAKE"}
EXPECTED_DATASETS = QA_DATASETS | VQA_DATASETS
TASK_FAMILIES = ["Text QA", "Medical VQA"]
STAGE_ORDER = [
    "R1-Analysis",
    "R1-Review",
    "R2-Analysis",
    "R2-Review",
    "R3-Analysis",
    "R3-Review",
]
DISPLAY_STAGE_ORDER = [stage.replace("-", "\n", 1) for stage in STAGE_ORDER]
COLORS = {"Text QA": "#1B9E77", "Medical VQA": "#7570B3"}
MARKERS = {"Text QA": "o", "Medical VQA": "s"}
ARIAL_FONT_PATHS = [
    Path("/mnt/c/Windows/Fonts/arial.ttf"),
    Path("/mnt/c/Windows/Fonts/arialbd.ttf"),
    Path("/mnt/c/Windows/Fonts/ariali.ttf"),
    Path("/mnt/c/Windows/Fonts/arialbi.ttf"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def configure_vector_output() -> None:
    missing_fonts = [path for path in ARIAL_FONT_PATHS if not path.is_file()]
    if missing_fonts:
        raise FileNotFoundError(f"Missing Windows Arial fonts: {missing_fonts}")
    for font_path in ARIAL_FONT_PATHS:
        font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.weight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def prepare_data(input_dir: Path):
    audit_data = process_audit_data_to_df(input_dir)
    mode_data = audit_data.loc[audit_data["failure_mode"].eq(FAILURE_MODE)].copy()
    if mode_data.empty:
        raise ValueError(f"No F-{FAILURE_MODE} audit entries were found in {input_dir}")

    datasets = set(mode_data["dataset"].dropna().unique())
    if datasets != EXPECTED_DATASETS:
        missing = sorted(EXPECTED_DATASETS - datasets)
        unexpected = sorted(datasets - EXPECTED_DATASETS)
        raise ValueError(
            f"Dataset mismatch for F-{FAILURE_MODE}; missing={missing}, "
            f"unexpected={unexpected}"
        )

    mode_data["task_family"] = np.select(
        [
            mode_data["dataset"].isin(QA_DATASETS),
            mode_data["dataset"].isin(VQA_DATASETS),
        ],
        TASK_FAMILIES,
        default="",
    )
    mode_data = mode_data.loc[mode_data["round_stage"].isin(STAGE_ORDER)].copy()
    observed_stages = set(mode_data["round_stage"].unique())
    if observed_stages != set(STAGE_ORDER):
        raise ValueError(
            f"Round-step mismatch for F-{FAILURE_MODE}; "
            f"observed={sorted(observed_stages)}"
        )
    return mode_data


def build_trajectory(mode_data):
    stats, _ = collect_group_stage_stats(
        mode_data,
        "task_family",
        TASK_FAMILIES,
        STAGE_ORDER,
        seed_offset=21200,
    )
    trajectory = {}
    for task_family in TASK_FAMILIES:
        cells = [stats[(task_family, stage)] for stage in STAGE_ORDER]
        trajectory[task_family] = {
            "rate": np.asarray([cell["failure_rate"] for cell in cells], float),
            "ci_low": np.asarray([cell["ci_95"][0] for cell in cells], float),
            "ci_high": np.asarray([cell["ci_95"][1] for cell in cells], float),
            "failed": np.asarray([cell["failed_count"] for cell in cells], int),
            "total": np.asarray([cell["total_count"] for cell in cells], int),
        }
    return trajectory


def plot_trajectory(trajectory, output_path: Path) -> None:
    configure_vector_output()
    x_values = np.arange(len(STAGE_ORDER), dtype=float)
    fig, axis = plt.subplots(figsize=(9.6, 4.1))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")

    for x_value in x_values:
        axis.axvline(
            x_value,
            color="#E5E7EB",
            linestyle="--",
            linewidth=0.75,
            alpha=0.45,
            zorder=0,
        )

    for task_family in TASK_FAMILIES:
        series = trajectory[task_family]
        color = COLORS[task_family]
        marker = MARKERS[task_family]
        axis.fill_between(
            x_values,
            series["ci_low"],
            series["ci_high"],
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        axis.scatter(
            x_values,
            series["rate"],
            color=color,
            s=105,
            marker=marker,
            edgecolors="white",
            linewidth=1.4,
            label=task_family,
            zorder=5,
        )
        for stage_index in range(len(STAGE_ORDER) - 1):
            axis.annotate(
                "",
                xy=(x_values[stage_index + 1], series["rate"][stage_index + 1]),
                xytext=(x_values[stage_index], series["rate"][stage_index]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 2.1,
                    "shrinkA": 7,
                    "shrinkB": 7,
                    "mutation_scale": 13,
                },
                zorder=4,
            )

    all_ci_high = np.concatenate(
        [trajectory[task_family]["ci_high"] for task_family in TASK_FAMILIES]
    )
    y_max = 5 * np.ceil((float(np.nanmax(all_ci_high)) * 1.08) / 5)
    y_max = max(10.0, y_max)

    axis.set_xlim(-0.18, len(STAGE_ORDER) - 0.82)
    axis.set_ylim(0, y_max)
    axis.set_xticks(x_values)
    axis.set_xticklabels(DISPLAY_STAGE_ORDER, fontsize=8, color="#6F7190")
    axis.set_yticks(np.arange(0, y_max + 0.01, 5))
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    axis.set_ylabel("Failure Rate (%)", fontsize=10, color="#111827")
    axis.grid(axis="y", color="#D5D8EA", linewidth=0.9, alpha=0.95)
    axis.tick_params(axis="x", width=0.8, length=3.5, color="#6F7190")
    axis.tick_params(
        axis="y",
        width=0.8,
        length=3.5,
        color="#6F7190",
        labelcolor="#6F7190",
        labelsize=8,
    )
    for spine_name in ("left", "bottom"):
        axis.spines[spine_name].set_color("#B9BDD8")
        axis.spines[spine_name].set_linewidth(0.9)

    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncol=2,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.4,
        borderaxespad=0,
    )
    fig.text(0.012, 0.975, "e", ha="left", va="top", fontsize=14, fontweight="bold")
    fig.text(
        0.055,
        0.975,
        "F-2.1.2",
        ha="left",
        va="top",
        fontsize=12,
        fontstyle="italic",
        color="#B43C2B",
    )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.84, bottom=0.20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format="pdf",
        dpi=600,
        bbox_inches="tight",
        transparent=False,
        metadata={
            "Title": "Phase 2 F-2.1.2 QA and VQA trajectory",
            "Creator": "Matplotlib vector PDF for Adobe Illustrator editing",
        },
    )
    plt.close(fig)


def print_summary(trajectory) -> None:
    print("F-2.1.2 task-family round-step statistics:")
    for task_family in TASK_FAMILIES:
        print(f"  {task_family}")
        for stage, rate, failed, total, low, high in zip(
            STAGE_ORDER,
            trajectory[task_family]["rate"],
            trajectory[task_family]["failed"],
            trajectory[task_family]["total"],
            trajectory[task_family]["ci_low"],
            trajectory[task_family]["ci_high"],
        ):
            print(
                f"    {stage}: {rate:.2f}% ({failed}/{total}), "
                f"95% CI {low:.2f}–{high:.2f}"
            )


def main() -> None:
    args = parse_args()
    mode_data = prepare_data(args.input_dir)
    trajectory = build_trajectory(mode_data)
    print_summary(trajectory)
    plot_trajectory(trajectory, args.output)
    print(f"Saved editable vector PDF: {args.output}")


if __name__ == "__main__":
    main()
