"""
This script is in Phase 1. It calculates the accuracy of Single LLM and MAS
at the dataset-LLM granularity, writes summary tables, and plots the
Single-LLM-vs-best-MAS comparison for each dataset.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))

from medagentaudit.utils.json_utils import load_jsonl
from medagentaudit.utils.logger import DualLogger


single_llm_dir = project_root / "logs" / "single_llm" / "20260302"
mas_dir = project_root / "logs" / "mas_collaboration_results_audit"
output_dir = project_root / "logs" / "accuracy_stats"

output_single_llm_file = output_dir / "single_llm_accuracy_stats.csv"
output_mas_file = output_dir / "mas_accuracy_stats.csv"
output_best_mas_file = output_dir / "best_mas_accuracy_stats.csv"
output_plot_png_file = output_dir / "single_llm_vs_best_mas_accuracy.png"
output_plot_pdf_file = output_dir / "single_llm_vs_best_mas_accuracy.pdf"

COLOR_BASELINE = "#B0B5B9"
COLOR_IMPROVED = "#6B8EAD"
COLOR_NO_IMPROVEMENT = "#C6878F"
TEXT_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"
MAX_X_AXIS_TICK_COUNT = 8

TEXT_DATASET_ORDER = ["medqa", "pubmedqa", "medxpertqa-text"]
VISION_DATASET_ORDER = ["pathvqa", "slake", "vqa-rad"]
LLM_ORDER = [
    "deepseek-reasoner",
    "gemini-3-flash-preview",
    "gpt-5.2",
    "qwen3-8b",
    "glm-4.6v",
    "qwen3-vl-8b-thinking",
]

DATASET_DISPLAY_MAP = {
    "medqa": "MedQA",
    "pubmedqa": "PubMedQA",
    "medxpertqa-text": "MedXpertQA",
    "pathvqa": "PathVQA",
    "slake": "SLAKE",
    "vqa-rad": "VQA-RAD",
}

DATASET_MODALITY_MAP = {
    "medqa": "Text-based",
    "pubmedqa": "Text-based",
    "medxpertqa-text": "Text-based",
    "pathvqa": "Vision-based",
    "slake": "Vision-based",
    "vqa-rad": "Vision-based",
}

LLM_DISPLAY_MAP = {
    "deepseek-reasoner": "DeepSeek-V3.2-Thinking",
    "gemini-3-flash-preview": "Gemini-3-Flash-Preview",
    "gpt-5.2": "GPT-5.2",
    "qwen3-8b": "Qwen3-8B",
    "glm-4.6v": "GLM-4.6V",
    "qwen3-vl-8b-thinking": "Qwen3-VL-8B-Thinking",
}


def get_metadata(file_path_str, object_type):
    """Retrieve dataset / LLM / MAS names from a file stem."""
    parts = file_path_str.lower().split("_")
    if object_type == "mas":
        found_mas = parts[0]
        found_dataset = parts[1]
        found_llm = "_".join(parts[2:])
        return found_dataset, found_llm, found_mas

    found_dataset = parts[0]
    found_llm = "_".join(parts[1:])
    return found_dataset, found_llm


def normalize_answer(answer):
    if answer is None:
        return ""
    return str(answer).strip().lower()


def calculate_accuracy(jsonl_file):
    case_num = 0
    correct_num = 0
    data = load_jsonl(jsonl_file)
    for json_record in data:
        case_num += 1
        if normalize_answer(json_record.get("ground_truth")) == normalize_answer(json_record.get("predicted_answer")):
            correct_num += 1

    accuracy = correct_num / case_num if case_num > 0 else 0.0
    return case_num, correct_num, accuracy


def dataset_sort_key(dataset):
    if dataset in TEXT_DATASET_ORDER:
        return (0, TEXT_DATASET_ORDER.index(dataset))
    if dataset in VISION_DATASET_ORDER:
        return (1, VISION_DATASET_ORDER.index(dataset))
    return (2, dataset)


def llm_sort_key(llm):
    if llm in LLM_ORDER:
        return (0, LLM_ORDER.index(llm))
    return (1, llm)


def collect_single_llm_stats():
    stats_single_llm = {}
    all_jsonl_files_single_llm = sorted(single_llm_dir.glob("*.jsonl"))
    print(f"Found {len(all_jsonl_files_single_llm)} Single LLM result files.")

    for jsonl_file in all_jsonl_files_single_llm:
        print(f"Processing Single LLM file: {jsonl_file.name}")
        dataset, llm = get_metadata(jsonl_file.stem, "single_llm")
        case_num, correct_num, acc = calculate_accuracy(jsonl_file)
        print(f"Dataset: {dataset}, LLM: {llm}, Accuracy: {acc:.4f} ({correct_num}/{case_num})")
        stats_single_llm[(dataset, llm)] = {
            "dataset": dataset,
            "llm": llm,
            "case_num": case_num,
            "correct_num": correct_num,
            "accuracy": acc,
        }

    return stats_single_llm


def collect_mas_stats():
    stats_mas = []
    all_jsonl_files_mas = sorted(mas_dir.glob("*.jsonl"))
    print(f"Found {len(all_jsonl_files_mas)} MAS result files.")

    for jsonl_file in all_jsonl_files_mas:
        print(f"Processing MAS file: {jsonl_file.name}")
        dataset, llm, mas = get_metadata(jsonl_file.stem, "mas")
        case_num, correct_num, acc = calculate_accuracy(jsonl_file)
        print(
            f"Dataset: {dataset}, LLM: {llm}, MAS: {mas}, "
            f"Accuracy: {acc:.4f} ({correct_num}/{case_num})"
        )
        stats_mas.append(
            {
                "dataset": dataset,
                "llm": llm,
                "mas": mas,
                "case_num": case_num,
                "correct_num": correct_num,
                "accuracy": acc,
            }
        )

    return stats_mas


def write_single_llm_csv(single_llm_stats):
    sorted_records = sorted(
        single_llm_stats.values(),
        key=lambda item: (dataset_sort_key(item["dataset"]), llm_sort_key(item["llm"])),
    )

    with open(output_single_llm_file, "w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["Dataset", "LLM", "Case_Num", "Correct_Num", "Accuracy"])
        for record in sorted_records:
            writer.writerow(
                [
                    record["dataset"],
                    record["llm"],
                    record["case_num"],
                    record["correct_num"],
                    f'{record["accuracy"]:.4f}',
                ]
            )

    print(f"Single LLM accuracy stats written to {output_single_llm_file}")


def write_mas_csv(mas_stats):
    sorted_records = sorted(
        mas_stats,
        key=lambda item: (
            dataset_sort_key(item["dataset"]),
            llm_sort_key(item["llm"]),
            item["mas"],
        ),
    )

    with open(output_mas_file, "w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["Dataset", "LLM", "MAS", "Case_Num", "Correct_Num", "Accuracy"])
        for record in sorted_records:
            writer.writerow(
                [
                    record["dataset"],
                    record["llm"],
                    record["mas"],
                    record["case_num"],
                    record["correct_num"],
                    f'{record["accuracy"]:.4f}',
                ]
            )

    print(f"MAS accuracy stats written to {output_mas_file}")


def select_best_mas_records(single_llm_stats, mas_stats):
    grouped_mas_stats = defaultdict(list)
    for record in mas_stats:
        grouped_mas_stats[(record["dataset"], record["llm"])].append(record)

    best_mas_records = []
    for key, single_record in sorted(
        single_llm_stats.items(),
        key=lambda item: (dataset_sort_key(item[0][0]), llm_sort_key(item[0][1])),
    ):
        dataset, llm = key
        mas_candidates = sorted(grouped_mas_stats.get((dataset, llm), []), key=lambda item: item["mas"])
        if not mas_candidates:
            print(f"Warning: missing MAS results for dataset={dataset}, llm={llm}.")
            continue

        best_mas_record = max(
            mas_candidates,
            key=lambda item: (item["accuracy"], item["correct_num"], item["case_num"]),
        )
        improvement = best_mas_record["accuracy"] - single_record["accuracy"]

        best_mas_records.append(
            {
                "dataset": dataset,
                "llm": llm,
                "single_case_num": single_record["case_num"],
                "single_correct_num": single_record["correct_num"],
                "single_accuracy": single_record["accuracy"],
                "best_mas": best_mas_record["mas"],
                "best_mas_case_num": best_mas_record["case_num"],
                "best_mas_correct_num": best_mas_record["correct_num"],
                "best_mas_accuracy": best_mas_record["accuracy"],
                "improvement": improvement,
            }
        )

    return best_mas_records


def write_best_mas_csv(best_mas_records):
    with open(output_best_mas_file, "w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            [
                "Dataset",
                "LLM",
                "Single_Case_Num",
                "Single_Correct_Num",
                "Single_Accuracy",
                "Best_MAS",
                "Best_MAS_Case_Num",
                "Best_MAS_Correct_Num",
                "Best_MAS_Accuracy",
                "Improvement",
            ]
        )
        for record in best_mas_records:
            writer.writerow(
                [
                    record["dataset"],
                    record["llm"],
                    record["single_case_num"],
                    record["single_correct_num"],
                    f'{record["single_accuracy"]:.4f}',
                    record["best_mas"],
                    record["best_mas_case_num"],
                    record["best_mas_correct_num"],
                    f'{record["best_mas_accuracy"]:.4f}',
                    f'{record["improvement"]:.4f}',
                ]
            )

    print(f"Best-MAS summary written to {output_best_mas_file}")


def format_accuracy_label(acc_percent):
    rounded_value = round(acc_percent)
    if abs(acc_percent - rounded_value) < 0.15:
        return f"{rounded_value:.0f}%"
    return f"{acc_percent:.1f}%"


def get_display_positions(single_acc_percent, mas_acc_percent):
    single_label = format_accuracy_label(single_acc_percent)
    mas_label = format_accuracy_label(mas_acc_percent)
    if single_label == mas_label:
        jitter = 0.45
        if mas_acc_percent > single_acc_percent:
            return single_acc_percent - jitter, mas_acc_percent + jitter
        if mas_acc_percent < single_acc_percent:
            return single_acc_percent + jitter, mas_acc_percent - jitter
        return single_acc_percent - jitter, mas_acc_percent + jitter
    return single_acc_percent, mas_acc_percent


def get_axis_config(values):
    min_value = min(values)
    max_value = max(values)

    padded_left = max(0, min_value - 4)
    padded_right = min(100, max_value + 3)

    if padded_right - padded_left < 10:
        midpoint = (padded_left + padded_right) / 2
        padded_left = max(0, midpoint - 5)
        padded_right = min(100, midpoint + 5)

    tick_step_candidates = [5, 10, 20, 25, 50]
    for tick_step in tick_step_candidates:
        left_bound = max(0, math.floor(padded_left / tick_step) * tick_step)
        right_bound = min(100, math.ceil(padded_right / tick_step) * tick_step)
        tick_count = int((right_bound - left_bound) / tick_step) + 1

        if tick_count <= MAX_X_AXIS_TICK_COUNT:
            return int(left_bound), int(right_bound), tick_step

    return 0, 100, tick_step_candidates[-1]


def build_dataset_rows(dataset_names):
    known_dataset_names = set(dataset_names)
    rows = []

    text_row = [dataset for dataset in TEXT_DATASET_ORDER if dataset in known_dataset_names]
    vision_row = [dataset for dataset in VISION_DATASET_ORDER if dataset in known_dataset_names]

    remaining = sorted(
        [dataset for dataset in dataset_names if dataset not in set(TEXT_DATASET_ORDER + VISION_DATASET_ORDER)],
        key=dataset_sort_key,
    )

    if text_row:
        rows.append(text_row)
    if vision_row:
        rows.append(vision_row)
    while remaining:
        rows.append(remaining[:3])
        remaining = remaining[3:]

    return rows


def plot_single_dataset_comparison(ax, dataset, dataset_records):
    y_positions = list(range(len(dataset_records)))
    llm_labels = [LLM_DISPLAY_MAP.get(record["llm"], record["llm"]) for record in dataset_records]

    displayed_single_values = []
    displayed_mas_values = []

    for y_position, record in zip(y_positions, dataset_records):
        single_acc_percent = record["single_accuracy"] * 100
        mas_acc_percent = record["best_mas_accuracy"] * 100
        single_plot_value, mas_plot_value = get_display_positions(single_acc_percent, mas_acc_percent)

        displayed_single_values.append(single_plot_value)
        displayed_mas_values.append(mas_plot_value)

        is_improved = record["improvement"] > 0
        mas_color = COLOR_IMPROVED if is_improved else COLOR_NO_IMPROVEMENT

        ax.plot(
            [single_plot_value, mas_plot_value],
            [y_position, y_position],
            color=mas_color,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            single_plot_value,
            y_position,
            s=220,
            color=COLOR_BASELINE,
            edgecolors="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            mas_plot_value,
            y_position,
            s=220,
            color=mas_color,
            edgecolors="white",
            linewidth=1.2,
            zorder=4,
        )

        if mas_plot_value >= single_plot_value:
            single_text_x = single_plot_value - 0.65
            mas_text_x = mas_plot_value + 0.65
            single_alignment = "right"
            mas_alignment = "left"
        else:
            single_text_x = single_plot_value + 0.65
            mas_text_x = mas_plot_value - 0.65
            single_alignment = "left"
            mas_alignment = "right"

        ax.text(
            single_text_x,
            y_position,
            format_accuracy_label(single_acc_percent),
            ha=single_alignment,
            va="center",
            fontsize=12,
            color=TEXT_COLOR,
        )
        ax.text(
            mas_text_x,
            y_position,
            format_accuracy_label(mas_acc_percent),
            ha=mas_alignment,
            va="center",
            fontsize=12,
            color=TEXT_COLOR,
        )

    axis_min, axis_max, tick_step = get_axis_config(displayed_single_values + displayed_mas_values)
    x_ticks = list(range(axis_min, axis_max + tick_step, tick_step))
    ax.set_xlim(axis_min, axis_max)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick}%" for tick in x_ticks])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(llm_labels)
    ax.invert_yaxis()

    dataset_display_name = DATASET_DISPLAY_MAP.get(dataset, dataset)
    dataset_modality = DATASET_MODALITY_MAP.get(dataset)
    if dataset_modality:
        ax.set_title(f"{dataset_display_name} ({dataset_modality})", fontsize=18, pad=10)
    else:
        ax.set_title(dataset_display_name, fontsize=18, pad=10)

    ax.set_xlabel("Accuracy (%)", fontsize=16)
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)


def plot_single_llm_vs_best_mas(best_mas_records):
    if not best_mas_records:
        print("No best-MAS summary records found. Skipping plotting.")
        return

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 1.2,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    dataset_to_records = defaultdict(list)
    for record in best_mas_records:
        dataset_to_records[record["dataset"]].append(record)

    for dataset_records in dataset_to_records.values():
        dataset_records.sort(key=lambda item: llm_sort_key(item["llm"]))

    rows = build_dataset_rows(sorted(dataset_to_records.keys(), key=dataset_sort_key))
    nrows = len(rows)
    ncols = max(len(row) for row in rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.7 * ncols, 4.1 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")

    for row_index, row_datasets in enumerate(rows):
        for col_index in range(ncols):
            axis = axes[row_index][col_index]
            if col_index >= len(row_datasets):
                axis.axis("off")
                continue

            dataset = row_datasets[col_index]
            plot_single_dataset_comparison(axis, dataset, dataset_to_records[dataset])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLOR_BASELINE,
            markeredgecolor=COLOR_BASELINE,
            markersize=11,
            label="Single LLM Baseline",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLOR_IMPROVED,
            markeredgecolor=COLOR_IMPROVED,
            markersize=11,
            label="Best Multi-Agent System (Improvement)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLOR_NO_IMPROVEMENT,
            markeredgecolor=COLOR_NO_IMPROVEMENT,
            markersize=11,
            label="Best Multi-Agent System (No Improvement / Drop)",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#DDDDDD",
        handletextpad=0.6,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.87, wspace=0.55, hspace=0.46)

    fig.savefig(output_plot_png_file, dpi=300, bbox_inches="tight")
    fig.savefig(output_plot_pdf_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot written to {output_plot_png_file}")
    print(f"Comparison plot written to {output_plot_pdf_file}")


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    terminal_log_file = output_dir / "accuracy_calculation.log"
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    single_llm_stats = collect_single_llm_stats()
    mas_stats = collect_mas_stats()

    write_single_llm_csv(single_llm_stats)
    write_mas_csv(mas_stats)

    best_mas_records = select_best_mas_records(single_llm_stats, mas_stats)
    write_best_mas_csv(best_mas_records)
    plot_single_llm_vs_best_mas(best_mas_records)


if __name__ == "__main__":
    main()
