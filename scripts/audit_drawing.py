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

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica'],
            'axes.linewidth': 1.2,
            'axes.labelsize': 13,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'legend.title_fontsize': 12,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('white')
    plt.suptitle(
        f"Failure Mode {code}: {mode_name}\nSystemic Dynamics and Divergence of Cognitive Collapse",
        fontsize=26,
        fontweight='bold',
        y=1.01,
        ha='center',
    )

    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.4)

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
    x_step = 1.30
    y_step = 1.08
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

    box_width = 1.12
    box_height = 0.92
    box_rounding = 0.24
    value_y_offset = 0.20
    count_y_offset = 0.00
    ci_y_offset = -0.20

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

    ax_a.set_xlim(x_positions[0] - box_width / 2 - 0.2, x_positions[-1] + box_width / 2 + 0.2)
    ax_a.set_ylim(y_positions[-1] - box_height / 2 - 0.12, y_positions[0] + box_height / 2 + 0.12)
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(x_order, rotation=30, rotation_mode='anchor', ha='center', fontweight='bold', fontsize=14)
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels(mas_order, fontweight='bold', fontsize=14)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax_a.spines[spine].set_visible(False)

    ax_a.tick_params(axis='x', length=0, pad=20)
    ax_a.tick_params(axis='y', length=0)

    sm_a = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm_a)
    sm_a.set_array([])
    cbar_a = plt.colorbar(sm_a, ax=ax_a, pad=0.02, shrink=0.7, aspect=15)
    cbar_a.set_label('Failure Rate (%)', rotation=270, labelpad=20, fontsize=14, fontweight='bold')
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
    ax_a.set_title("A. MAS Cognitive Pipeline Breakdown", fontweight='bold', loc='left', pad=25)

    # =========================================================
    # Panel B: Modality Divergence Breakdown (V4 style)
    # =========================================================
    ax_b = fig.add_subplot(gs[0, 1])

    df_b = df_mode.groupby(['dataset', 'round_stage'], observed=False)['failed'].mean().reset_index()
    df_b['failure_rate'] = df_b['failed'] * 100

    x_numeric = np.arange(len(x_order))
    qa_colors = ["#c6dbef", "#6baed6", "#2171b5"]
    vqa_colors = ["#fdd0a2", "#fd8d3c", "#d94801"]

    qa_bottom = np.zeros(len(x_order))
    for idx, dataset in enumerate(qa_datasets):
        if dataset not in df_b['dataset'].unique():
            continue
        y_vals = (
            df_b[df_b['dataset'] == dataset]
            .set_index('round_stage')
            .reindex(x_order)['failure_rate']
            .fillna(0)
            .values
        )
        ax_b.fill_between(
            x_numeric,
            qa_bottom,
            qa_bottom + y_vals,
            color=qa_colors[idx],
            alpha=0.9,
            label=f"QA: {dataset}",
            edgecolor='white',
            lw=1,
        )
        qa_bottom += y_vals

    vqa_top = np.zeros(len(x_order))
    for idx, dataset in enumerate(vqa_datasets):
        if dataset not in df_b['dataset'].unique():
            continue
        y_vals = (
            df_b[df_b['dataset'] == dataset]
            .set_index('round_stage')
            .reindex(x_order)['failure_rate']
            .fillna(0)
            .values
        )
        ax_b.fill_between(
            x_numeric,
            vqa_top,
            vqa_top - y_vals,
            color=vqa_colors[idx],
            alpha=0.9,
            label=f"VQA: {dataset}",
            edgecolor='white',
            lw=1,
        )
        vqa_top -= y_vals

    ax_b.axhline(0, color='black', linewidth=2, zorder=5)
    ax_b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y_val, pos: f"{abs(int(y_val))}%"))
    ax_b.set_xticks(x_numeric)
    ax_b.set_xticklabels(x_order, rotation=0, rotation_mode='anchor', ha='center', fontweight='bold')
    ax_b.set_ylabel("Accumulated Failure Representation", fontweight='bold')
    ax_b.set_title("B. Modality Divergence: Text (QA) vs Vision (VQA) Breakdown", fontweight='bold', loc='left', pad=15)
    ax_b.grid(axis='x', linestyle='--', alpha=0.3)

    max_extent = max(float(qa_bottom.max()), float(np.abs(vqa_top.min())))
    if max_extent > 0:
        ax_b.set_ylim(-max_extent * 1.1, max_extent * 1.1)

    handles_b, labels_b = ax_b.get_legend_handles_labels()
    if handles_b:
        ax_b.legend(handles_b[::-1], labels_b[::-1], loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

    # =========================================================
    # Helper: Kinematic Acceleration Vector Plot (V4 style)
    # =========================================================
    def plot_kinematic_vectors(ax, df_subset, title, palette_name):
        if df_subset.empty:
            ax.text(0.5, 0.5, "No Data Available", ha='center', va='center')
            ax.axis('off')
            return

        df_agg = df_subset.groupby(['llm', 'round_stage'], observed=False)['failed'].mean().reset_index()
        df_agg['failure_rate'] = df_agg['failed'] * 100

        models = sorted(df_agg['llm'].dropna().unique())
        base_colors = sns.color_palette(palette_name, n_colors=len(models))

        for idx, model in enumerate(models):
            y_vals = (
                df_agg[df_agg['llm'] == model]
                .set_index('round_stage')
                .reindex(x_order)['failure_rate']
                .to_numpy()
            )

            valid_mask = ~np.isnan(y_vals)
            if valid_mask.any():
                ax.scatter(
                    x_numeric[valid_mask],
                    y_vals[valid_mask],
                    color=base_colors[idx],
                    s=100,
                    zorder=5,
                    edgecolors='white',
                    lw=2,
                    label=model,
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
                        'mutation_scale': 15,
                    },
                    zorder=3,
                )

        ax.set_title(title, fontweight='bold', loc='left', pad=15)
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(x_order, rotation=0, rotation_mode='anchor', ha='center', fontweight='bold')
        ax.set_ylabel("Absolute Failure Rate (%)", fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Base LLM", frameon=False, loc='upper left')

    # =========================================================
    # Panels C & D: LLM Kinematic Degradation (V4 style)
    # =========================================================
    ax_c = fig.add_subplot(gs[1, 0])
    df_c = df_mode[df_mode['dataset'].isin(qa_datasets)].copy()
    plot_kinematic_vectors(ax_c, df_c, "C. LLM Kinetic Degradation: Text QA", "Set1")

    ax_d = fig.add_subplot(gs[1, 1])
    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    plot_kinematic_vectors(ax_d, df_d, "D. LLM Kinetic Degradation: Vision VQA", "Dark2")

    cax_delta = fig.add_axes([0.92, 0.15, 0.015, 0.25])
    sm_delta = plt.cm.ScalarMappable(cmap=delta_cmap, norm=delta_norm)
    sm_delta.set_array([])
    cbar_delta = plt.colorbar(sm_delta, cax=cax_delta)
    cbar_delta.set_label('Stage-to-Stage Change ($\\Delta$ %)', rotation=270, labelpad=20, fontweight='bold')
    cbar_delta.outline.set_visible(False)

    plt.subplots_adjust(right=0.9, top=0.92, bottom=0.08)

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
