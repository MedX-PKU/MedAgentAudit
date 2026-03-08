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
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
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


def plot_failure_mode_comprehensive(code, df_mode, output_dir):
    """
    Generate the final comprehensive figure.
    - Panel A uses the V3 MAS cognitive pipeline style.
    - Panel B uses the V4 dataset modality divergence style.
    - Panels C and D use the V4 kinematic degradation style.
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
    common_tick_fontsize = 14
    common_colorbar_label_fontsize = 14
    common_colorbar_tick_fontsize = 12
    panel_b_xtick_pad = 18

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica'],
            'axes.linewidth': 1.2,
            'axes.labelsize': common_axis_label_fontsize,
            'axes.titlesize': 16,
            'xtick.labelsize': common_tick_fontsize,
            'ytick.labelsize': common_tick_fontsize,
            'legend.fontsize': common_tick_fontsize,
            'legend.title_fontsize': common_tick_fontsize,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )

    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor('white')

    gs = GridSpec(2, 2, figure=fig, wspace=0.24, hspace=0.15, height_ratios=[1.10, 1.20])
    title_y = 1.03

    qa_datasets = ["MedQA", "PubMedQA", "MedXpertQA"]
    vqa_datasets = ["PathVQA", "VQA-RAD", "SLAKE"]

    mas_priority = ["ColaCare", "HealthcareAgent", "MAC", "MDAgents", "MedAgent", "ReConcile"]
    mas_present = df_mode['mas'].dropna().unique().tolist()
    mas_order = [mas for mas in mas_priority if mas in mas_present]
    mas_order.extend(sorted(mas for mas in mas_present if mas not in mas_order))

    delta_cmap = plt.cm.coolwarm
    delta_norm = mcolors.CenteredNorm(vcenter=0, halfrange=20)

    # =========================================================
    # Panel A: MAS Cognitive Pipeline Breakdown (V3 style)
    # =========================================================
    ax_a = fig.add_subplot(gs[0, 0])

    num_stages = len(x_order)
    num_mas = len(mas_order)
    x_step = 1.16
    y_step = 1.16
    x_positions = np.arange(num_stages) * x_step
    y_positions = np.arange(num_mas)[::-1] * y_step
    y_map = dict(zip(mas_order, y_positions))
    panel_a_stats = {}

    for mas_idx, mas in enumerate(mas_order):
        for stage_idx, stage in enumerate(x_order):
            cell_values = df_mode.loc[
                (df_mode['mas'] == mas) & (df_mode['round_stage'] == stage),
                'failed',
            ].dropna().astype(float).to_numpy()

            if cell_values.size == 0:
                continue

            panel_a_stats[(mas, stage)] = {
                'failure_rate': float(cell_values.mean() * 100),
                'failed_count': int(cell_values.sum()),
                'total_count': int(cell_values.size),
                'ci_95': compute_bootstrap_failure_ci(
                    cell_values,
                    seed=BOOTSTRAP_BASE_SEED + mas_idx * max(num_stages, 1) + stage_idx,
                ),
            }

    cmap_a = sns.color_palette("flare", as_cmap=True)
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
    box_rounding = 0.22
    value_y_offset = 0.17
    count_y_offset = 0.00
    ci_y_offset = -0.17

    for mas in mas_order:
        y_val = y_map[mas]
        for x_val, stage in zip(x_positions, x_order):
            cell_stats = panel_a_stats.get((mas, stage))

            if cell_stats is None:
                box = mpatches.FancyBboxPatch(
                    (x_val - box_width / 2, y_val - box_height / 2),
                    box_width,
                    box_height,
                    boxstyle=f"round,pad=0,rounding_size={box_rounding}",
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
                box = mpatches.FancyBboxPatch(
                    (x_val - box_width / 2, y_val - box_height / 2),
                    box_width,
                    box_height,
                    boxstyle=f"round,pad=0,rounding_size={box_rounding}",
                    facecolor=bg_color,
                    edgecolor='white',
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
        x_order,
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

    na_patch = mpatches.Patch(
        facecolor='#F8F9FA',
        edgecolor='#A0A0A0',
        linestyle='--',
        hatch='//////',
        label='Structurally N/A',
    )
    ax_a.legend(
        handles=[na_patch],
        loc='lower left',
        bbox_to_anchor=(0.0, 0.985),
        borderaxespad=0,
        frameon=False,
        fontsize=14,
    )
    ax_a.set_title(
        "A. MAS Cognitive Pipeline Breakdown",
        fontweight='bold',
        loc='left',
        pad=18,
        y=title_y,
    )

    # =========================================================
    # Panel B: Modality Divergence Breakdown (V4 style)
    # =========================================================
    datasets_present = df_mode['dataset'].dropna().unique().tolist()
    qa_present = [dataset for dataset in qa_datasets if dataset in datasets_present]
    vqa_present = [dataset for dataset in vqa_datasets if dataset in datasets_present]
    panel_b_datasets = qa_present + vqa_present
    panel_b_row_count = max(len(panel_b_datasets), 1)

    gs_b = gs[0, 1].subgridspec(
        2,
        1,
        height_ratios=[1.90, max(0.60, 0.17 * panel_b_row_count)],
        hspace=0.30,
    )
    ax_b = fig.add_subplot(gs_b[0, 0])
    ax_b_table = fig.add_subplot(gs_b[1, 0], sharex=ax_b)
    ax_b.set_zorder(2)
    ax_b_table.set_zorder(1)
    ax_b_table.set_facecolor('none')

    x_numeric = np.arange(len(x_order))
    qa_colors = ["#c6dbef", "#6baed6", "#2171b5"]
    vqa_colors = ["#fdd0a2", "#fd8d3c", "#d94801"]
    qa_color_map = dict(zip(qa_datasets, qa_colors))
    vqa_color_map = dict(zip(vqa_datasets, vqa_colors))

    panel_b_stats = {}

    for dataset_idx, dataset in enumerate(panel_b_datasets):
        for stage_idx, stage in enumerate(x_order):
            cell_values = df_mode.loc[
                (df_mode['dataset'] == dataset) & (df_mode['round_stage'] == stage),
                'failed',
            ].dropna().astype(float).to_numpy()

            if cell_values.size == 0:
                continue

            panel_b_stats[(dataset, stage)] = {
                'failure_rate': float(cell_values.mean() * 100),
                'failed_count': int(cell_values.sum()),
                'total_count': int(cell_values.size),
                'ci_95': compute_bootstrap_failure_ci(
                    cell_values,
                    seed=BOOTSTRAP_BASE_SEED + 5000 + dataset_idx * max(len(x_order), 1) + stage_idx,
                ),
            }

    qa_bottom = np.zeros(len(x_order))
    for dataset in qa_present:
        y_vals = np.array(
            [panel_b_stats.get((dataset, stage), {}).get('failure_rate', 0.0) for stage in x_order],
            dtype=float,
        )
        ax_b.fill_between(
            x_numeric,
            qa_bottom,
            qa_bottom + y_vals,
            color=qa_color_map[dataset],
            alpha=0.92,
            edgecolor='white',
            linewidth=1.2,
            zorder=2,
        )
        qa_bottom += y_vals

    vqa_top = np.zeros(len(x_order))
    for dataset in vqa_present:
        y_vals = np.array(
            [panel_b_stats.get((dataset, stage), {}).get('failure_rate', 0.0) for stage in x_order],
            dtype=float,
        )
        ax_b.fill_between(
            x_numeric,
            vqa_top,
            vqa_top - y_vals,
            color=vqa_color_map[dataset],
            alpha=0.92,
            edgecolor='white',
            linewidth=1.2,
            zorder=2,
        )
        vqa_top -= y_vals

    for x_val in x_numeric:
        ax_b.axvline(x_val, color='#D8DEE6', linestyle='--', linewidth=0.9, alpha=0.45, zorder=1)

    ax_b.axhline(0, color='black', linewidth=2, zorder=5)
    ax_b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y_val, pos: f"{abs(y_val):.0f}%"))
    ax_b.set_xlim(x_numeric[0] - 0.25, x_numeric[-1] + 0.25)
    ax_b.set_ylabel(
        "Accumulated Failure Representation",
        fontweight='bold',
        fontsize=common_axis_label_fontsize,
    )
    ax_b.set_title(
        "B. Modality Divergence: Text (QA) vs Vision (VQA) Breakdown",
        fontweight='bold',
        loc='left',
        pad=18,
        y=title_y,
    )
    ax_b.set_xticks(x_numeric)
    ax_b.set_xticklabels(
        x_order,
        rotation=30,
        rotation_mode='anchor',
        ha='center',
        fontweight='bold',
        fontsize=common_tick_fontsize,
    )
    ax_b.xaxis.set_ticks_position('bottom')
    ax_b.tick_params(
        axis='x',
        which='major',
        bottom=True,
        labelbottom=True,
        length=6,
        width=1.4,
        color='black',
        direction='out',
        pad=panel_b_xtick_pad,
    )
    ax_b.tick_params(axis='y', labelsize=common_tick_fontsize)
    ax_b.grid(axis='y', linestyle=':', alpha=0.25)
    ax_b.spines['bottom'].set_visible(True)
    ax_b.spines['bottom'].set_linewidth(1.5)
    ax_b.spines['bottom'].set_color('black')

    max_extent = max(
        float(qa_bottom.max()) if qa_bottom.size else 0.0,
        float(np.abs(vqa_top.min())) if vqa_top.size else 0.0,
    )
    if max_extent > 0:
        ax_b.set_ylim(-max_extent * 1.05, max_extent * 1.05)
    else:
        ax_b.set_ylim(-5, 5)

    table_row_order = qa_present + vqa_present
    row_color_map = {**qa_color_map, **vqa_color_map}
    row_positions = {}
    row_spacing = 1.26
    group_gap = 0.58
    table_top_offset = 0.72
    table_top_padding = 1.08
    table_bottom_padding = 0.48
    current_y = -table_top_offset

    for dataset in qa_present:
        row_positions[dataset] = current_y
        current_y -= row_spacing

    if qa_present and vqa_present:
        current_y -= group_gap

    for dataset in vqa_present:
        row_positions[dataset] = current_y
        current_y -= row_spacing

    table_cell_fontsize = 8.1 if len(x_order) <= 8 else 7.4 if len(x_order) <= 11 else 6.8
    if len(table_row_order) >= 6:
        table_cell_fontsize -= 0.4
    table_label_fontsize = 10.8 if len(table_row_order) <= 4 else 10.2
    label_transform = mtransforms.blended_transform_factory(ax_b_table.transAxes, ax_b_table.transData)
    label_x = -0.04
    table_x_min = x_numeric[0] - 0.22
    table_x_max = x_numeric[-1] + 0.22

    for x_val in x_numeric:
        ax_b_table.axvline(x_val, color='#E5E7EB', linestyle=':', linewidth=0.9, zorder=0)

    for dataset in table_row_order:
        ax_b_table.hlines(
            row_positions[dataset],
            xmin=table_x_min,
            xmax=table_x_max,
            color='#F1F5F9',
            linewidth=1.0,
            linestyle=':',
            zorder=0,
        )

    if qa_present and vqa_present:
        group_divider_y = (row_positions[qa_present[-1]] + row_positions[vqa_present[0]]) / 2
        ax_b_table.hlines(
            group_divider_y,
            xmin=table_x_min,
            xmax=table_x_max,
            color='#CBD5E1',
            linewidth=1.0,
            linestyle='--',
            zorder=1,
        )

    for dataset in table_row_order:
        y_val = row_positions[dataset]
        ax_b_table.text(
            label_x,
            y_val,
            dataset,
            transform=label_transform,
            ha='right',
            va='center',
            fontsize=table_label_fontsize,
            fontweight='bold',
            color=row_color_map[dataset],
            clip_on=False,
        )

        for x_val, stage in zip(x_numeric, x_order):
            cell_stats = panel_b_stats.get((dataset, stage))

            if cell_stats is None:
                cell_text = "N/A"
                cell_color = '#9CA3AF'
            else:
                ci_lower, ci_upper = cell_stats['ci_95']
                cell_text = (
                    f"{cell_stats['failure_rate']:.1f}% "
                    f"({cell_stats['failed_count']}/{cell_stats['total_count']})\n"
                    f"[{ci_lower:.1f}–{ci_upper:.1f}]"
                )
                cell_color = '#111827'

            ax_b_table.text(
                x_val,
                y_val,
                cell_text,
                ha='center',
                va='center',
                fontsize=table_cell_fontsize,
                color=cell_color,
                linespacing=1.04,
                zorder=2,
            )

    ax_b_table.set_yticks([])
    ax_b_table.set_xticks(x_numeric)
    ax_b_table.tick_params(axis='x', top=False, labeltop=False, bottom=False, labelbottom=False, length=0)

    if table_row_order:
        table_y_min = min(row_positions.values()) - table_bottom_padding
        table_y_max = max(row_positions.values()) + table_top_padding
    else:
        table_y_min, table_y_max = -0.5, 0.5
    ax_b_table.set_ylim(table_y_min, table_y_max)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax_b_table.spines[spine].set_visible(False)

    # =========================================================
    # Helper: Kinematic Acceleration Vector Plot (V4 style)
    # =========================================================
    def plot_kinematic_panel(subspec, df_subset, title, palette_name, seed_offset):
        model_display_names = {
            "DeepSeek-V3.2-Thinking": "DeepSeek",
            "Gemini-3-Flash-Preview": "Gemini",
            "Qwen3-8B": "Qwen",
            "Qwen3-VL-8B-Thinking": "Qwen",
        }
        model_values = df_subset['llm'].dropna().unique().tolist() if not df_subset.empty else []
        models = [llm for llm in LLM_PRIORITY if llm in model_values]
        models.extend(sorted(llm for llm in model_values if llm not in models))
        display_x_order = [stage_label.replace("Svnthesis", "Synthesis") for stage_label in x_order]

        table_row_count = max(len(models), 1)
        panel_gs = subspec.subgridspec(
            2,
            1,
            height_ratios=[1.92, max(0.60, 0.17 * table_row_count)],
            hspace=0.30,
        )
        ax = fig.add_subplot(panel_gs[0, 0])
        ax_table = fig.add_subplot(panel_gs[1, 0], sharex=ax)
        ax.set_zorder(2)
        ax_table.set_zorder(1)
        ax_table.set_facecolor('none')

        if df_subset.empty:
            ax.text(0.5, 0.5, "No Data Available", ha='center', va='center')
            ax.axis('off')
            ax_table.axis('off')
            return ax, ax_table

        panel_stats = {}
        max_failure_rate = 0.0
        for model_idx, model in enumerate(models):
            for stage_idx, stage in enumerate(x_order):
                cell_values = df_subset.loc[
                    (df_subset['llm'] == model) & (df_subset['round_stage'] == stage),
                    'failed',
                ].dropna().astype(float).to_numpy()

                if cell_values.size == 0:
                    continue

                failure_rate = float(cell_values.mean() * 100)
                panel_stats[(model, stage)] = {
                    'failure_rate': failure_rate,
                    'failed_count': int(cell_values.sum()),
                    'total_count': int(cell_values.size),
                    'ci_95': compute_bootstrap_failure_ci(
                        cell_values,
                        seed=BOOTSTRAP_BASE_SEED + seed_offset + model_idx * max(len(x_order), 1) + stage_idx,
                    ),
                }
                max_failure_rate = max(max_failure_rate, failure_rate)

        base_colors = sns.color_palette(palette_name, n_colors=max(len(models), 1))
        color_map = dict(zip(models, base_colors))
        y_min, y_max, tick_step = compute_adaptive_rate_axis(max_failure_rate)

        for x_val in x_numeric:
            ax.axvline(x_val, color='#E5E7EB', linestyle='--', linewidth=0.9, alpha=0.55, zorder=1)

        for idx, model in enumerate(models):
            y_vals = np.array(
                [panel_stats.get((model, stage), {}).get('failure_rate', np.nan) for stage in x_order],
                dtype=float,
            )

            valid_mask = ~np.isnan(y_vals)
            if valid_mask.any():
                ax.scatter(
                    x_numeric[valid_mask],
                    y_vals[valid_mask],
                    color=color_map[model],
                    s=115,
                    zorder=5,
                    edgecolors='white',
                    lw=2,
                )

            for stage_idx in range(len(x_order) - 1):
                if np.isnan(y_vals[stage_idx]) or np.isnan(y_vals[stage_idx + 1]):
                    continue

                delta = y_vals[stage_idx + 1] - y_vals[stage_idx]
                segment_color = delta_cmap(delta_norm(delta))
                segment_lw = 2 + abs(delta) / 5.0

                ax.annotate(
                    "",
                    xy=(x_numeric[stage_idx + 1], y_vals[stage_idx + 1]),
                    xytext=(x_numeric[stage_idx], y_vals[stage_idx]),
                    arrowprops={
                        'arrowstyle': "->",
                        'color': segment_color,
                        'lw': segment_lw,
                        'shrinkA': 8,
                        'shrinkB': 8,
                        'mutation_scale': 17,
                    },
                    zorder=3,
                )

        ax.set_title(title, fontweight='bold', loc='left', pad=15)
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(
            display_x_order,
            rotation=30,
            rotation_mode='anchor',
            ha='right',
            fontweight='bold',
            fontsize=common_tick_fontsize,
        )
        ax.set_xlim(x_numeric[0] - 0.2, x_numeric[-1] + 0.2)
        ax.set_ylabel(
            "Absolute Failure Rate (%)",
            fontweight='bold',
            fontsize=common_axis_label_fontsize,
        )
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.001, tick_step))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(axis='x', labelsize=common_tick_fontsize, width=1.4, length=6, pad=8)
        ax.tick_params(axis='y', labelsize=common_tick_fontsize, width=1.4, length=6)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        for tick_label in ax.get_xticklabels():
            tick_label.set_clip_on(False)
            tick_label.set_zorder(6)

        row_positions = {}
        row_spacing = 1.26
        table_top_offset = 0.72
        table_top_padding = 1.18
        table_bottom_padding = 0.48
        current_y = -table_top_offset

        for model in models:
            row_positions[model] = current_y
            current_y -= row_spacing

        table_cell_fontsize = 8.1 if len(x_order) <= 8 else 7.4
        if len(models) >= 6:
            table_cell_fontsize -= 0.4
        table_label_fontsize = 10.8 if len(models) <= 4 else 10.2
        label_transform = mtransforms.blended_transform_factory(ax_table.transAxes, ax_table.transData)
        label_x = -0.04

        for x_val in x_numeric:
            ax_table.axvline(x_val, color='#E5E7EB', linestyle=':', linewidth=0.9, zorder=0)

        for model in models:
            y_val = row_positions[model]
            display_model = model_display_names.get(model, model)
            ax_table.hlines(
                y_val,
                xmin=x_numeric[0] - 0.22,
                xmax=x_numeric[-1] + 0.22,
                color='#F1F5F9',
                linewidth=1.0,
                zorder=0,
            )
            ax_table.text(
                label_x,
                y_val,
                display_model,
                transform=label_transform,
                ha='right',
                va='center',
                fontsize=table_label_fontsize,
                fontweight='bold',
                color=color_map[model],
                clip_on=False,
            )

            for x_val, stage in zip(x_numeric, x_order):
                cell_stats = panel_stats.get((model, stage))
                if cell_stats is None:
                    cell_text = "N/A"
                    cell_color = '#9CA3AF'
                else:
                    ci_lower, ci_upper = cell_stats['ci_95']
                    cell_text = (
                        f"{cell_stats['failure_rate']:.1f}% ({cell_stats['failed_count']}/{cell_stats['total_count']})\n"
                        f"CI [{ci_lower:.1f}–{ci_upper:.1f}]"
                    )
                    cell_color = '#111827'

                ax_table.text(
                    x_val,
                    y_val,
                    cell_text,
                    ha='center',
                    va='center',
                    fontsize=table_cell_fontsize,
                    color=cell_color,
                    linespacing=1.04,
                    zorder=2,
                )

        ax_table.set_yticks([])
        ax_table.set_xticks(x_numeric)
        ax_table.tick_params(axis='x', top=False, labeltop=False, bottom=False, labelbottom=False, length=0)

        if row_positions:
            table_y_min = min(row_positions.values()) - table_bottom_padding
            table_y_max = max(row_positions.values()) + table_top_padding
        else:
            table_y_min, table_y_max = -0.5, 0.5
        ax_table.set_ylim(table_y_min, table_y_max)

        for spine in ['top', 'right', 'left', 'bottom']:
            ax_table.spines[spine].set_visible(False)

        return ax, ax_table

    # =========================================================
    # Panels C & D: LLM Kinematic Degradation (V4 style)
    # =========================================================
    df_c = df_mode[df_mode['dataset'].isin(qa_datasets)].copy()
    ax_c, _ = plot_kinematic_panel(gs[1, 0], df_c, "C. LLM Kinetic Degradation: Text QA", "Set1", seed_offset=10000)

    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    ax_d, _ = plot_kinematic_panel(gs[1, 1], df_d, "D. LLM Kinetic Degradation: Vision VQA", "Dark2", seed_offset=15000)

    plt.subplots_adjust(left=0.065, right=0.905, top=0.985, bottom=0.055)
    fig.canvas.draw()

    kinematic_axes = [axis for axis in (ax_c, ax_d) if axis is not None]
    kinematic_top = max(axis.get_position().y1 for axis in kinematic_axes)
    kinematic_height = max(axis.get_position().height for axis in kinematic_axes)
    cbar_height = kinematic_height * 0.82
    cbar_bottom = kinematic_top - cbar_height
    cbar_left = ax_d.get_position().x1 + 0.015
    cax_delta = fig.add_axes([cbar_left, cbar_bottom, 0.015, cbar_height])
    sm_delta = plt.cm.ScalarMappable(cmap=delta_cmap, norm=delta_norm)
    sm_delta.set_array([])
    cbar_delta = plt.colorbar(sm_delta, cax=cax_delta)
    cbar_delta.set_label(
        'Stage-to-Stage Change ($\\Delta$ %)',
        rotation=270,
        labelpad=20,
        fontweight='bold',
        fontsize=common_colorbar_label_fontsize,
    )
    cbar_delta.ax.tick_params(labelsize=common_colorbar_tick_fontsize)
    cbar_delta.outline.set_visible(False)

    filename = f"Failure_Mode_{code}_Nature_Evolution.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Saved Nature-quality comprehensive figure: {save_path}")


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
