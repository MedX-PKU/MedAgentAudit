'''
./scripts/audit_drawing.py
This script visualizes the failure rates for each failure mode across different rounds and stages,
based on the audit results from multi-agent systems.
It answers 3 core questions regarding the evolution of failure modes:
1. MAS frameworks evolution
2. Medical scenarios/datasets evolution
3. Base LLMs evolution for Medical QA & VQA
'''
import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))

AUDIT_CONFIG = {
    "1.1.1": {"status_key": "factual_hallucination_status", "name": "Factual Hallucination", "valid_rounds": [1, 2, 3]},
    "1.2.1": {"status_key": "modality_neglect_status", "name": "Modality Neglect", "valid_rounds": [1, 2, 3]},
    "2.1.1": {"status_key": "role_task_alignment", "name": "Role Assignment Mismatch", "valid_rounds": [1]},
    "2.1.2": {"status_key": "knowledge_activation_status", "name": "Knowledge Activation Fail", "valid_rounds": [1, 2, 3]},
    "2.2.1": {"status_key": "interaction_redundancy", "name": "Interaction Redundancy", "valid_rounds": [1, 2, 3]},
    "2.2.2": {"status_key": "conflict_resolution_status", "name": "Unresolved Conflicts", "valid_rounds": [1, 2, 3]},
    "3.1.1": {"status_key": "suppression_status", "name": "Minority Suppression", "valid_rounds": [1, 2, 3]},
    "3.1.2": {"status_key": "authority_bias_status", "name": "Authority Bias", "valid_rounds": [1, 2, 3]},
    "3.1.3": {"status_key": "neglect_of_conflict_status", "name": "Neglect of Contradictions", "valid_rounds": [1, 2, 3]},
    "3.2.1": {"status_key": "inter_round_consistency_status", "name": "Inter-round Inconsistency", "valid_rounds": [2, 3]},
}

LOG_KEY_MAP = {
    "1.1.1": "1_1_1_factual_hallucination",
    "1.2.1": "1_2_1_neglect_or_misinterpretation_of_modality_info",
    "2.1.1": "2_1_1_role_assignment",
    "2.1.2": "2_1_2_domain_specific_knowledge_activation",
    "2.2.1": "2_2_1_repetition_of_initial_views",
    "2.2.2": "2_2_2_unresolved_conflicts",
    "3.1.1": "3_1_1_suppression_of_minority_views",
    "3.1.2": "3_1_2_authority_bias",
    "3.1.3": "3_1_3_neglect_of_contradictions",
    "3.2.1": "3_2_1_self_contradiction_when_decision",
}

STAGE_ORDER = ["role_assignment", "analysis", "synthesis", "review", "decision"]

STAGE_DISPLAY = {
    "role_assignment": "Assign",
    "analysis": "Analysis",
    "synthesis": "Synthesis",
    "review": "Review",
    "decision": "Decision",
}

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BASE_SEED = 20260307
LLM_PRIORITY = [
    "DeepSeek-V3.2-Thinking",
    "GPT-5.2",
    "Gemini-3-Flash-Preview",
    "Qwen3-8B",
    "GLM-4.6V",
    "Qwen3-VL-8B-Thinking",
]


def compute_bootstrap_failure_ci(failure_values, num_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_BASE_SEED):
    """
    Compute a deterministic percentile bootstrap 95% confidence interval (%) for
    the failure rate within a single Panel A cell.
    """
    values = np.asarray(failure_values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(num_resamples, values.size), replace=True)
    resampled_rates = resampled.mean(axis=1) * 100
    ci_lower, ci_upper = np.percentile(resampled_rates, [2.5, 97.5])
    return float(ci_lower), float(ci_upper)


def compute_adaptive_rate_axis(max_rate):
    """
    Choose a compact, human-friendly y-axis ceiling for failure-rate panels.
    Example: max=45 -> top=50 instead of leaving a large empty band up to 100.
    """
    if not np.isfinite(max_rate) or max_rate <= 0:
        return 0.0, 5.0, 1.0

    padded_max = float(max_rate) * 1.08

    if padded_max <= 8:
        tick_step = 2.0
    elif padded_max <= 60:
        tick_step = 5.0
    else:
        tick_step = 10.0

    axis_top = tick_step * np.ceil(padded_max / tick_step)
    axis_top = min(100.0, max(axis_top, tick_step * 2))

    if axis_top <= max_rate and axis_top < 100:
        axis_top = min(100.0, axis_top + tick_step)

    return 0.0, float(axis_top), float(tick_step)


def process_audit_data_to_df(audit_results_path: Path):
    """
    Parse JSONL logs and construct a structured Pandas DataFrame.
    """
    records = []
    jsonl_files = list(audit_results_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} files. Processing into DataFrame...")

    mas_map = {
        "colacare": "ColaCare",
        "healthcareagent": "HealthcareAgent",
        "mac": "MAC",
        "mdagents": "MDAgents",
        "medagent": "MedAgent",
        "reconcile": "ReConcile",
    }

    llm_map = {
        "deepseek-reasoner": "DeepSeek-V3.2-Thinking",
        "gpt-5.2": "GPT-5.2",
        "gemini-3-flash-preview": "Gemini-3-Flash-Preview",
        "qwen3-8b": "Qwen3-8B",
        "glm-4.6v": "GLM-4.6V",
        "qwen3-vl-8b-thinking": "Qwen3-VL-8B-Thinking",
    }

    for jsonl_file in tqdm(jsonl_files, desc="Scanning Logs"):
        if "errors" in jsonl_file.name:
            continue

        stem = jsonl_file.stem
        parts = stem.split('_')
        if len(parts) < 3:
            continue

        raw_mas = parts[0]
        mas = mas_map.get(raw_mas.lower(), raw_mas)

        dataset = parts[1]
        if dataset.lower() == "medxpertqa-text":
            dataset = "MedXpertQA"

        raw_llm = "_".join(parts[2:])
        llm = llm_map.get(raw_llm.lower(), raw_llm)

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as file_handle:
                for line in file_handle:
                    if not line.strip():
                        continue

                    case_data = json.loads(line)

                    audit_data = None
                    if "case_history" in case_data and "audit" in case_data["case_history"]:
                        audit_data = case_data["case_history"]["audit"]

                    if not audit_data:
                        continue

                    rounds = audit_data.get("rounds", [])

                    for round_data in rounds:
                        round_num = round_data.get("round")
                        if not round_num:
                            continue

                        for code, config in AUDIT_CONFIG.items():
                            if round_num not in config["valid_rounds"]:
                                continue

                            log_key = LOG_KEY_MAP[code]
                            entries = round_data.get(log_key, [])

                            if isinstance(entries, dict):
                                entries = [entries]
                            if not entries:
                                continue

                            status_key = config["status_key"]

                            for entry in entries:
                                step = entry.get("step", "unknown").lower()
                                if step not in STAGE_ORDER:
                                    continue

                                result_obj = entry.get("audit_result", {})
                                status_val = result_obj.get(status_key)
                                is_failed = int(str(status_val) == "1" or status_val is True)

                                round_stage = f"R{round_num}-{STAGE_DISPLAY.get(step, step)}"

                                records.append(
                                    {
                                        "failure_mode": code,
                                        "mas": mas,
                                        "dataset": dataset,
                                        "llm": llm,
                                        "round_num": round_num,
                                        "step": step,
                                        "round_stage": round_stage,
                                        "failed": is_failed,
                                    }
                                )
        except Exception as exc:
            print(f"Error processing {jsonl_file}: {exc}")
            continue

    df = pd.DataFrame(records)
    return df


def build_group_stage_stats(df_subset, group_key, group_order, x_order, seed_offset):
    """
    Aggregate failure-rate statistics for a group-by-stage panel.
    """
    panel_stats = {}
    max_ci_upper = 0.0

    for group_idx, group in enumerate(group_order):
        for stage_idx, stage in enumerate(x_order):
            cell_values = df_subset.loc[
                (df_subset[group_key] == group) & (df_subset['round_stage'] == stage),
                'failed',
            ].dropna().astype(float).to_numpy()

            if cell_values.size == 0:
                continue

            failure_rate = float(cell_values.mean() * 100)
            ci_lower, ci_upper = compute_bootstrap_failure_ci(
                cell_values,
                seed=BOOTSTRAP_BASE_SEED + seed_offset + group_idx * max(len(x_order), 1) + stage_idx,
            )
            panel_stats[(group, stage)] = {
                'failure_rate': failure_rate,
                'failed_count': int(cell_values.sum()),
                'total_count': int(cell_values.size),
                'ci_95': (ci_lower, ci_upper),
            }
            max_ci_upper = max(
                max_ci_upper,
                failure_rate,
                float(ci_upper) if np.isfinite(ci_upper) else failure_rate,
            )

    return panel_stats, max_ci_upper


def extract_panel_series(panel_stats, group, x_order):
    """
    Convert panel stats into aligned y/CI arrays for plotting.
    """
    y_vals = []
    ci_lower = []
    ci_upper = []

    for stage in x_order:
        cell_stats = panel_stats.get((group, stage))
        if cell_stats is None:
            y_vals.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            continue

        y_vals.append(cell_stats['failure_rate'])
        lower, upper = cell_stats['ci_95']
        ci_lower.append(lower)
        ci_upper.append(upper)

    return (
        np.asarray(y_vals, dtype=float),
        np.asarray(ci_lower, dtype=float),
        np.asarray(ci_upper, dtype=float),
    )


def plot_series_with_ci(ax, x_numeric, y_vals, ci_lower, ci_upper, color, label):
    """
    Plot a line with large circular markers and a semi-transparent CI ribbon.
    """
    masked_y = np.ma.masked_invalid(y_vals)
    if masked_y.count() == 0:
        return

    masked_lower = np.ma.masked_invalid(ci_lower)
    masked_upper = np.ma.masked_invalid(ci_upper)
    ax.fill_between(
        x_numeric,
        masked_lower,
        masked_upper,
        color=color,
        alpha=0.14,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x_numeric,
        masked_y,
        color=color,
        linewidth=2.4,
        marker='o',
        markersize=9,
        markerfacecolor=color,
        markeredgecolor='white',
        markeredgewidth=1.6,
        label=label,
        zorder=3,
    )


def style_line_panel(
    ax,
    x_numeric,
    x_labels,
    title,
    y_label,
    y_axis_top,
    tick_step,
    panel_title_fontsize,
    panel_title_pad,
    panel_title_y,
    panel_title_linespacing,
    common_axis_label_fontsize,
    common_tick_fontsize,
    legend_loc='upper left',
    legend_ncol=1,
):
    """
    Apply the shared line-chart style used by Panels B-D.
    """
    for x_val in x_numeric:
        ax.axvline(x_val, color='#E8E9EE', linewidth=1.0, zorder=0)

    ax.set_title(
        title,
        fontweight='bold',
        fontsize=panel_title_fontsize,
        loc='left',
        pad=panel_title_pad,
        y=panel_title_y,
        linespacing=panel_title_linespacing,
    )
    ax.set_xlim(x_numeric[0] - 0.15, x_numeric[-1] + 0.15)
    ax.set_xticks(x_numeric)
    ax.set_xticklabels(
        x_labels,
        rotation=0 if len(x_labels) <= 4 else 18,
        ha='center',
        fontweight='bold',
        fontsize=common_tick_fontsize,
    )
    ax.set_ylabel(
        y_label,
        fontweight='bold',
        fontsize=common_axis_label_fontsize,
    )
    ax.set_ylim(0.0, y_axis_top)
    ax.set_yticks(np.arange(0.0, y_axis_top + 0.001, tick_step))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.tick_params(axis='x', labelsize=common_tick_fontsize, width=1.4, length=6, pad=8)
    ax.tick_params(axis='y', labelsize=common_tick_fontsize, width=1.4, length=6)
    ax.grid(axis='y', color='#E8E9EE', linestyle='-', linewidth=0.9, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#3F3F46')
    ax.spines['bottom'].set_color('#3F3F46')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            facecolor='white',
            edgecolor='#C7CAD3',
            borderpad=0.6,
            handlelength=1.8,
            columnspacing=1.0,
            handletextpad=0.6,
        )
        legend.get_frame().set_linewidth(1.0)


def format_stats_cell(cell_stats):
    """
    Render a compact multi-line cell string for appendix tables.
    """
    if cell_stats is None:
        return "N/A"

    ci_lower, ci_upper = cell_stats['ci_95']
    return (
        f"{cell_stats['failure_rate']:.1f}%\n"
        f"{cell_stats['failed_count']}/{cell_stats['total_count']}\n"
        f"CI [{ci_lower:.1f}, {ci_upper:.1f}]"
    )


def render_stats_table(
    ax,
    title,
    row_order,
    col_order,
    panel_stats,
    row_display_map=None,
    row_color_map=None,
    col_display_labels=None,
):
    """
    Render a standalone appendix table with the numeric values removed from the main figure.
    """
    if not row_order:
        ax.axis('off')
        ax.text(0.5, 0.5, "No Data Available", ha='center', va='center', fontsize=13)
        return

    row_display_map = row_display_map or {}
    row_color_map = row_color_map or {}
    col_display_labels = col_display_labels or col_order
    row_positions = np.arange(len(row_order))[::-1]
    col_positions = np.arange(len(col_order))

    ax.set_xlim(-0.5, len(col_order) - 0.5)
    ax.set_ylim(-0.5, len(row_order) - 0.5)

    for row_idx, y_val in enumerate(row_positions):
        bg_color = '#FBFCFE' if row_idx % 2 == 0 else '#F4F6FA'
        ax.add_patch(
            mpatches.Rectangle(
                (-0.5, y_val - 0.5),
                len(col_order),
                1.0,
                facecolor=bg_color,
                edgecolor='none',
                zorder=0,
            )
        )

    for x_edge in np.arange(-0.5, len(col_order) + 0.5, 1.0):
        ax.axvline(x_edge, color='#D7D9E0', linewidth=1.0, zorder=1)
    for y_edge in np.arange(-0.5, len(row_order) + 0.5, 1.0):
        ax.axhline(y_edge, color='#D7D9E0', linewidth=1.0, zorder=1)

    for row_idx, row in enumerate(row_order):
        y_val = row_positions[row_idx]
        for x_val, stage in zip(col_positions, col_order):
            cell_stats = panel_stats.get((row, stage))
            ax.text(
                x_val,
                y_val,
                format_stats_cell(cell_stats),
                ha='center',
                va='center',
                fontsize=9.3,
                color='#111827' if cell_stats is not None else '#9CA3AF',
                linespacing=1.15,
                zorder=2,
            )

    ax.set_xticks(col_positions)
    ax.set_xticklabels(col_display_labels, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=10)

    ax.set_yticks(row_positions)
    ax.set_yticklabels(
        [row_display_map.get(row, row) for row in row_order],
        fontsize=12,
        fontweight='bold',
    )
    ax.tick_params(axis='y', length=0, pad=10)
    for tick_label, row in zip(ax.get_yticklabels(), row_order):
        tick_label.set_color(row_color_map.get(row, '#111827'))

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    ax.set_title(title, loc='left', fontsize=15, fontweight='bold', pad=18)


def save_figure_outputs(fig, output_prefix: Path, dpi=600):
    """
    Save both PDF and PNG outputs for a figure.
    """
    pdf_path = output_prefix.parent / f"{output_prefix.name}.pdf"
    png_path = output_prefix.parent / f"{output_prefix.name}.png"
    fig.savefig(pdf_path, dpi=dpi, format='pdf', bbox_inches='tight', transparent=False)
    fig.savefig(png_path, dpi=max(300, dpi // 2), format='png', bbox_inches='tight', transparent=False)
    return pdf_path, png_path


def plot_failure_mode_comprehensive(code, df_mode, output_dir):
    """
    Generate the redesigned comprehensive figure and separate appendix tables.
    """
    config = AUDIT_CONFIG[code]
    mode_name = config["name"]

    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found.")
        return

    unique_round_stages = df_mode[['round_num', 'step', 'round_stage']].drop_duplicates()
    unique_round_stages['step_idx'] = unique_round_stages['step'].map({step: idx for idx, step in enumerate(STAGE_ORDER)})
    unique_round_stages = unique_round_stages.sort_values(['round_num', 'step_idx'])
    x_order = unique_round_stages['round_stage'].tolist()

    if not x_order:
        return

    df_mode = df_mode.copy()
    df_mode['round_stage'] = pd.Categorical(df_mode['round_stage'], categories=x_order, ordered=True)

    common_axis_label_fontsize = 14
    common_tick_fontsize = 13
    common_colorbar_label_fontsize = 14
    common_colorbar_tick_fontsize = 12
    panel_title_fontsize = 18
    panel_title_pad = 16
    panel_title_y = 1.03
    panel_title_linespacing = 1.12

    plt.rcParams.update(
        {
            'font.family': 'Arial',
            'font.sans-serif': ['Arial'],
            'axes.linewidth': 1.2,
            'axes.labelsize': common_axis_label_fontsize,
            'axes.titlesize': panel_title_fontsize,
            'xtick.labelsize': common_tick_fontsize,
            'ytick.labelsize': common_tick_fontsize,
            'legend.fontsize': common_tick_fontsize,
            'legend.title_fontsize': common_tick_fontsize,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )

    fig = plt.figure(figsize=(23.4, 14.8))
    fig.patch.set_facecolor('white')

    gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.34, height_ratios=[1.0, 1.0])

    qa_datasets = ["MedQA", "PubMedQA", "MedXpertQA"]
    vqa_datasets = ["PathVQA", "VQA-RAD", "SLAKE"]
    dataset_color_map = {
        "MedQA": "#2F5D8A",
        "PubMedQA": "#5C84B1",
        "MedXpertQA": "#8FB3D9",
        "PathVQA": "#6F7D8C",
        "VQA-RAD": "#A45D78",
        "SLAKE": "#D1845D",
    }
    model_color_map = {
        "DeepSeek-V3.2-Thinking": "#2F5D8A",
        "GPT-5.2": "#2F8A74",
        "Gemini-3-Flash-Preview": "#A16E98",
        "Qwen3-8B": "#C67A3B",
        "GLM-4.6V": "#6F7681",
        "Qwen3-VL-8B-Thinking": "#9C4F73",
    }
    model_display_names = {
        "DeepSeek-V3.2-Thinking": "DeepSeek",
        "Gemini-3-Flash-Preview": "Gemini",
        "Qwen3-8B": "Qwen",
        "Qwen3-VL-8B-Thinking": "Qwen-VL",
    }
    display_x_order = [stage_label.replace("Svnthesis", "Synthesis") for stage_label in x_order]
    x_numeric = np.arange(len(x_order), dtype=float)

    mas_priority = ["ColaCare", "HealthcareAgent", "MAC", "MDAgents", "MedAgent", "ReConcile"]
    mas_present = df_mode['mas'].dropna().unique().tolist()
    mas_order = [mas for mas in mas_priority if mas in mas_present]
    mas_order.extend(sorted(mas for mas in mas_present if mas not in mas_order))

    # =========================================================
    # Panel A: square-cell grey heatmap
    # =========================================================
    ax_a = fig.add_subplot(gs[0, 0])

    num_stages = len(x_order)
    num_mas = len(mas_order)
    x_step = 1.16
    y_step = 1.16
    x_positions = np.arange(num_stages) * x_step
    y_positions = np.arange(num_mas)[::-1] * y_step
    y_map = dict(zip(mas_order, y_positions))
    panel_a_stats, _ = build_group_stage_stats(df_mode, 'mas', mas_order, x_order, seed_offset=0)

    cmap_a = mcolors.LinearSegmentedColormap.from_list(
        "failure_mode_greys",
        ['#EFEFF6', '#777792'],
    )
    norm_a = mcolors.Normalize(vmin=0, vmax=100)

    for y_val in y_positions:
        ax_a.hlines(
            y_val,
            xmin=x_positions[0],
            xmax=x_positions[-1],
            color='#D3D3D3',
            linewidth=3,
            zorder=1,
        )

    box_width = 0.94
    box_height = 0.94
    value_y_offset = 0.17
    count_y_offset = 0.00
    ci_y_offset = -0.17

    for mas in mas_order:
        y_val = y_map[mas]
        for x_val, stage in zip(x_positions, x_order):
            cell_stats = panel_a_stats.get((mas, stage))

            if cell_stats is None:
                box = mpatches.Rectangle(
                    (x_val - box_width / 2, y_val - box_height / 2),
                    box_width,
                    box_height,
                    facecolor='#F8F9FA',
                    edgecolor='#A0A0A0',
                    linewidth=1.5,
                    linestyle='--',
                    hatch='//////',
                    zorder=2,
                )
                ax_a.add_patch(box)
                ax_a.text(
                    x_val,
                    y_val,
                    "N/A",
                    ha='center',
                    va='center',
                    color='#A0A0A0',
                    fontsize=10,
                    fontweight='bold',
                    zorder=3,
                )
            else:
                val = cell_stats['failure_rate']
                failed_count = cell_stats['failed_count']
                total_count = cell_stats['total_count']
                ci_lower, ci_upper = cell_stats['ci_95']
                bg_color = cmap_a(norm_a(val))
                box = mpatches.Rectangle(
                    (x_val - box_width / 2, y_val - box_height / 2),
                    box_width,
                    box_height,
                    facecolor=bg_color,
                    edgecolor='#D4D6DE',
                    linewidth=1.5,
                    zorder=2,
                )
                ax_a.add_patch(box)

                luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                text_color = 'white' if luminance < 0.65 else 'black'
                ax_a.text(
                    x_val,
                    y_val + value_y_offset,
                    f"{val:.1f}%",
                    ha='center',
                    va='center',
                    color=text_color,
                    fontsize=10.5,
                    fontweight='bold',
                    zorder=3,
                )
                ax_a.text(
                    x_val,
                    y_val + count_y_offset,
                    f"{failed_count}/{total_count}",
                    ha='center',
                    va='center',
                    color=text_color,
                    fontsize=9.2,
                    fontweight='bold',
                    zorder=3,
                )
                ax_a.text(
                    x_val,
                    y_val + ci_y_offset,
                    f"CI {ci_lower:.1f}–{ci_upper:.1f}",
                    ha='center',
                    va='center',
                    color=text_color,
                    fontsize=8.0,
                    fontweight='bold',
                    zorder=3,
                )

    ax_a.set_xlim(x_positions[0] - box_width / 2 - 0.08, x_positions[-1] + box_width / 2 + 0.10)
    ax_a.set_ylim(y_positions[-1] - box_height / 2 - 0.06, y_positions[0] + box_height / 2 + 0.08)
    ax_a.set_aspect('equal', adjustable='box')
    ax_a.set_anchor('NW')
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(
        display_x_order,
        rotation=30,
        rotation_mode='anchor',
        ha='center',
        fontweight='bold',
        fontsize=common_tick_fontsize,
    )
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels(mas_order, fontweight='bold', fontsize=common_tick_fontsize)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax_a.spines[spine].set_visible(False)

    ax_a.tick_params(axis='x', length=0, pad=12)
    ax_a.tick_params(axis='y', length=0)

    sm_a = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm_a)
    sm_a.set_array([])
    cbar_a = plt.colorbar(sm_a, ax=ax_a, pad=0.02, shrink=0.7, aspect=15)
    cbar_a.set_label(
        'Failure Rate (%)',
        rotation=270,
        labelpad=20,
        fontsize=common_colorbar_label_fontsize,
        fontweight='bold',
    )
    cbar_a.ax.tick_params(labelsize=common_colorbar_tick_fontsize)
    cbar_a.outline.set_visible(False)

    ax_a.set_title(
        "A. Failure rates across multi-agent architectures\nand collaboration stages.",
        fontweight='bold',
        fontsize=panel_title_fontsize,
        loc='left',
        pad=panel_title_pad,
        y=panel_title_y,
        linespacing=panel_title_linespacing,
    )

    # =========================================================
    # Panel B: dataset trajectories
    # =========================================================
    datasets_present = df_mode['dataset'].dropna().unique().tolist()
    qa_present = [dataset for dataset in qa_datasets if dataset in datasets_present]
    vqa_present = [dataset for dataset in vqa_datasets if dataset in datasets_present]
    panel_b_datasets = qa_present + vqa_present
    ax_b = fig.add_subplot(gs[0, 1])
    panel_b_stats, max_b_upper = build_group_stage_stats(
        df_mode,
        'dataset',
        panel_b_datasets,
        x_order,
        seed_offset=5000,
    )
    for dataset in panel_b_datasets:
        y_vals, ci_lower, ci_upper = extract_panel_series(panel_b_stats, dataset, x_order)
        plot_series_with_ci(
            ax_b,
            x_numeric,
            y_vals,
            ci_lower,
            ci_upper,
            dataset_color_map[dataset],
            dataset,
        )
    _, y_max_b, tick_step_b = compute_adaptive_rate_axis(max_b_upper)
    style_line_panel(
        ax_b,
        x_numeric,
        display_x_order,
        "B. Stage-wise failure-rate trajectories across\ntext and visual medical datasets.",
        "Failure Rate (%)",
        y_max_b,
        tick_step_b,
        panel_title_fontsize,
        panel_title_pad,
        panel_title_y,
        panel_title_linespacing,
        common_axis_label_fontsize,
        common_tick_fontsize,
        legend_loc='upper left',
        legend_ncol=2,
    )

    # =========================================================
    # Panels C & D: LLM trajectories
    # =========================================================
    df_c = df_mode[df_mode['dataset'].isin(qa_datasets)].copy()
    model_values_c = df_c['llm'].dropna().unique().tolist()
    models_c = [llm for llm in LLM_PRIORITY if llm in model_values_c]
    models_c.extend(sorted(llm for llm in model_values_c if llm not in models_c))
    ax_c = fig.add_subplot(gs[1, 0])
    panel_c_stats, max_c_upper = build_group_stage_stats(
        df_c,
        'llm',
        models_c,
        x_order,
        seed_offset=10000,
    )
    for model in models_c:
        y_vals, ci_lower, ci_upper = extract_panel_series(panel_c_stats, model, x_order)
        plot_series_with_ci(
            ax_c,
            x_numeric,
            y_vals,
            ci_lower,
            ci_upper,
            model_color_map.get(model, '#4B5563'),
            model_display_names.get(model, model),
        )
    _, y_max_c, tick_step_c = compute_adaptive_rate_axis(max_c_upper)
    style_line_panel(
        ax_c,
        x_numeric,
        display_x_order,
        "C. Absolute failure-rate trajectories for large language models in text QA tasks.",
        "Failure Rate (%)",
        y_max_c,
        tick_step_c,
        panel_title_fontsize,
        panel_title_pad,
        panel_title_y,
        panel_title_linespacing,
        common_axis_label_fontsize,
        common_tick_fontsize,
        legend_loc='upper left',
        legend_ncol=1,
    )

    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    model_values_d = df_d['llm'].dropna().unique().tolist()
    models_d = [llm for llm in LLM_PRIORITY if llm in model_values_d]
    models_d.extend(sorted(llm for llm in model_values_d if llm not in models_d))
    ax_d = fig.add_subplot(gs[1, 1])
    panel_d_stats, max_d_upper = build_group_stage_stats(
        df_d,
        'llm',
        models_d,
        x_order,
        seed_offset=15000,
    )
    for model in models_d:
        y_vals, ci_lower, ci_upper = extract_panel_series(panel_d_stats, model, x_order)
        plot_series_with_ci(
            ax_d,
            x_numeric,
            y_vals,
            ci_lower,
            ci_upper,
            model_color_map.get(model, '#4B5563'),
            model_display_names.get(model, model),
        )
    _, y_max_d, tick_step_d = compute_adaptive_rate_axis(max_d_upper)
    style_line_panel(
        ax_d,
        x_numeric,
        display_x_order,
        "D. Absolute failure-rate trajectories for vision-language models in medical VQA tasks.",
        "Failure Rate (%)",
        y_max_d,
        tick_step_d,
        panel_title_fontsize,
        panel_title_pad,
        panel_title_y,
        panel_title_linespacing,
        common_axis_label_fontsize,
        common_tick_fontsize,
        legend_loc='upper left',
        legend_ncol=1,
    )

    plt.subplots_adjust(left=0.07, right=0.985, top=0.955, bottom=0.07)
    main_pdf_path, main_png_path = save_figure_outputs(fig, output_dir / f"failure_mode_{code}")
    plt.close()
    print(f"Saved redesigned figure: {main_pdf_path}")
    print(f"Saved redesigned figure preview: {main_png_path}")

    appendix_fig = plt.figure(figsize=(17.5, 12.5))
    appendix_fig.patch.set_facecolor('white')
    appendix_fig.suptitle(
        f"Supplementary Numeric Tables for Failure Mode {code}",
        fontsize=18,
        fontweight='bold',
        y=0.985,
    )
    appendix_gs = GridSpec(
        3,
        1,
        figure=appendix_fig,
        height_ratios=[
            max(2.4, 0.55 * max(len(panel_b_datasets), 1)),
            max(2.1, 0.55 * max(len(models_c), 1)),
            max(2.1, 0.55 * max(len(models_d), 1)),
        ],
        hspace=0.54,
    )
    ax_b_table = appendix_fig.add_subplot(appendix_gs[0, 0])
    render_stats_table(
        ax_b_table,
        "Panel B. Dataset-level failure rates, counts, and 95% confidence intervals.",
        panel_b_datasets,
        x_order,
        panel_b_stats,
        row_color_map=dataset_color_map,
        col_display_labels=display_x_order,
    )
    ax_c_table = appendix_fig.add_subplot(appendix_gs[1, 0])
    render_stats_table(
        ax_c_table,
        "Panel C. Text-QA model failure rates, counts, and 95% confidence intervals.",
        models_c,
        x_order,
        panel_c_stats,
        row_display_map=model_display_names,
        row_color_map=model_color_map,
        col_display_labels=display_x_order,
    )
    ax_d_table = appendix_fig.add_subplot(appendix_gs[2, 0])
    render_stats_table(
        ax_d_table,
        "Panel D. Medical-VQA model failure rates, counts, and 95% confidence intervals.",
        models_d,
        x_order,
        panel_d_stats,
        row_display_map=model_display_names,
        row_color_map=model_color_map,
        col_display_labels=display_x_order,
    )
    appendix_pdf_path, appendix_png_path = save_figure_outputs(
        appendix_fig,
        output_dir / f"failure_mode_{code}_appendix_tables",
    )
    plt.close(appendix_fig)
    print(f"Saved appendix tables: {appendix_pdf_path}")
    print(f"Saved appendix tables preview: {appendix_png_path}")


def main():
    input_base_dir = project_root / "logs" / "audit_results" / "20260302"
    output_dir = project_root / "logs" / "audit_results" / "figures_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {output_dir}")

    df_global = process_audit_data_to_df(input_base_dir)

    print("\nGenerating Plots...")
    for code in AUDIT_CONFIG.keys():
        df_mode = df_global[df_global['failure_mode'] == code]
        plot_failure_mode_comprehensive(code, df_mode, output_dir)

    print("\nVisualization Audit Completed.")


if __name__ == "__main__":
    main()
