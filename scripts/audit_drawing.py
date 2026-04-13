'''
./scripts/audit_drawing.py
Build publication figures for the audit logs stored under `logs/audit_results/`.

The script turns per-case JSONL audit traces into a long Pandas table and then
renders one figure per failure mode. Each figure answers three analysis
questions:
1. How the error changes across multi-agent system (MAS) architectures.
2. How the error changes across medical datasets / modalities.
3. How the error changes across base models in text QA and medical VQA.

Reproduction notes
------------------
1. The default input directory is `logs/audit_results/20260302`.
2. Output figures are written to `logs/audit_results/figures_final`.
3. Run a single figure with:
   `uv run python scripts/audit_drawing.py 1.1.1`
4. Run all registered failure modes with:
   `uv run python scripts/audit_drawing.py`

The Phase I figures for `1.1.1` and `1.2.1` are handled by a dedicated
renderer because those panel sets received substantial paper-specific layout
tuning. The remaining failure modes use the more generic comprehensive
renderer below.

Reuse guide
-----------
1. If a new failure mode can share the standard 2 x 2 layout, update
   `plot_failure_mode_comprehensive()`.
2. If a new failure mode needs panel-specific geometry or typography changes,
   copy the `plot_failure_mode_111_enhanced()` pattern into a new dedicated
   renderer and route that code explicitly in `main()`.
3. Keep raw identifier normalization in `process_audit_data_to_df()`, but keep
   manuscript-facing short names inside the renderer that owns the figure.
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

# `AUDIT_CONFIG` maps a human-readable failure-mode code used in the paper to:
# - the status field stored in each JSON audit result
# - the paper-facing title
# - the rounds in which the metric is defined
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

# Each failure mode is stored in a different key inside the raw JSONL audit
# object. This map isolates that schema detail from the plotting logic.
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

# Canonical within-round step order used everywhere for sorting and plotting.
STAGE_ORDER = ["role_assignment", "analysis", "synthesis", "review", "decision"]

STAGE_DISPLAY = {
    "role_assignment": "Assign",
    "analysis": "Analysis",
    "synthesis": "Synthesis",
    "review": "Review",
    "decision": "Decision",
}

# Some failure modes are only interpreted on a subset of within-round steps in
# the manuscript. For 2.2.2, Phase II excludes synthesis nodes and keeps only
# the discussion-loop analysis / review steps.
FAILURE_MODE_STEP_FILTERS = {
    "2.2.2": {"analysis", "review"},
}

# All confidence intervals are deterministic bootstrap intervals so repeated
# runs produce the exact same numbers and figure geometry.
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

PALETTE_1 = ["#EFEFF6", "#D9D9EA", "#C4C3DE", "#AFADD2", "#777792"]
PALETTE_2 = ["#2B7BBA", "#2B7BBA", "#6BA3CF", "#ABCBE4", "#EAF3FA"]
PALETTE_3 = ["#AC3827", "#F5533A", "#F88574", "#FBB7AE", "#FEEAE7"]
PALETTE_4 = ["#E9E5E6", "#E8D3D2", "#E7C5C4", "#EBBCB9", "#DEB4B5", "#C89A9C"]
PALETTE_5 = ["#F4F1D0", "#F4F1D0", "#F4F1D0", "#DCD2A1", "#D0C78E", "#C5AC45"]
PALETTE_6 = ["#E4E9E3", "#DAE4D9", "#CBE0D1", "#BBD4BF", "#AFCDB1", "#7FAB87"]

ENHANCED_DATASET_COLOR_MAP = {
    "MedQA": "#CCCBE8",
    "PubMedQA": "#C1D5E5",
    "MedXpertQA": "#E3BDB7",
    "PathVQA": "#FAC5AE",
    "VQA-RAD": "#F3E5B0",
    "SLAKE": "#B4EDBE",
}

ENHANCED_MODEL_COLOR_MAP = {
    "DeepSeek-V3.2-Thinking": PALETTE_1[-1],
    "GPT-5.2": PALETTE_2[0],
    "Gemini-3-Flash-Preview": PALETTE_3[1],
    "Qwen3-8B": PALETTE_5[-1],
    "GLM-4.6V": PALETTE_6[-1],
    "Qwen3-VL-8B-Thinking": PALETTE_5[-1],
}

SERIES_MARKERS = ["o", "s", "^", "D", "P", "X"]


def compute_bootstrap_failure_ci(failure_values, num_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_BASE_SEED):
    """
    Compute a deterministic percentile bootstrap 95% confidence interval (%).

    `failure_values` is expected to be a binary vector for a single cell, where
    each element corresponds to one audited case at one round-step location.
    The returned interval is expressed in percentage points so it can be drawn
    directly on the figure without further scaling.
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

    We keep the lower bound fixed at 0 because all panels visualize absolute
    failure rates, then add a small amount of headroom so the top marker / CI
    does not visually collide with the panel title or legend.
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


def collect_group_stage_stats(df_subset, group_column, groups, x_order, seed_offset):
    """
    Collect per-group, per-stage failure statistics with deterministic bootstrap CIs.

    The return value is a dictionary keyed by `(group_name, round_stage)` so the
    same structure can be reused by the heatmap-like Panel A, the stacked area
    Panel B, the arrow panels C/D, and the appendix tables.
    """
    stats = {}
    max_failure_rate = 0.0

    for group_idx, group in enumerate(groups):
        for stage_idx, stage in enumerate(x_order):
            cell_values = df_subset.loc[
                (df_subset[group_column] == group) & (df_subset['round_stage'] == stage),
                'failed',
            ].dropna().astype(float).to_numpy()

            if cell_values.size == 0:
                continue

            failure_rate = float(cell_values.mean() * 100)
            stats[(group, stage)] = {
                'failure_rate': failure_rate,
                'failed_count': int(cell_values.sum()),
                'total_count': int(cell_values.size),
                'ci_95': compute_bootstrap_failure_ci(
                    cell_values,
                    seed=BOOTSTRAP_BASE_SEED + seed_offset + group_idx * max(len(x_order), 1) + stage_idx,
                ),
            }
            max_failure_rate = max(max_failure_rate, failure_rate)

    return stats, max_failure_rate


def apply_failure_mode_step_filter(code, df_mode):
    """Restrict a failure mode to the manuscript-approved audited steps."""
    allowed_steps = FAILURE_MODE_STEP_FILTERS.get(code)
    if not allowed_steps:
        return df_mode
    return df_mode[df_mode["step"].isin(allowed_steps)].copy()


def save_figure_variants(fig, output_dir, stem):
    """
    Save a figure as both PDF and PNG into the requested output directory.

    PDF is the paper-facing artifact. PNG is used for quick visual inspection
    during iterative layout work.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, dpi=600, format='pdf', bbox_inches='tight', transparent=False)
    fig.savefig(png_path, dpi=500, format='png', bbox_inches='tight', transparent=False)
    return pdf_path, png_path


def process_audit_data_to_df(audit_results_path: Path):
    """
    Parse JSONL logs and construct a structured Pandas DataFrame.

    Expected filename pattern:
        `<mas>_<dataset>_<llm>.jsonl`

    Expected JSONL content:
    - one case per line
    - each case contains `case_history.audit.rounds`
    - each round stores one or more failure-mode-specific audit entries

    Output columns are normalized so downstream plotting code does not need to
    know the raw JSON schema or filename aliases.
    """
    records = []
    jsonl_files = list(audit_results_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} files. Processing into DataFrame...")

    # Normalize raw file-name identifiers to the paper-facing system names used
    # throughout the figure. Reuse by editing this map first instead of
    # changing labels deeper in the plotting code.
    mas_map = {
        "colacare": "ColaCare",
        "healthcareagent": "HealthcareAgent",
        "mac": "MAC",
        "mdagents": "MDAgents",
        "medagents": "MedAgents",
        "reconcile": "ReConcile",
    }

    # Keep the exact logged model identifiers here; legend-facing short names
    # are defined later in each renderer so manuscript naming can change
    # without touching the ingestion layer.
    llm_map = {
        "deepseek-reasoner": "DeepSeek-V3.2-Thinking",
        "gpt-5.2": "GPT-5.2",
        "gemini-3-flash-preview": "Gemini-3-Flash-Preview",
        "qwen3-8b": "Qwen3-8B",
        "glm-4.6v": "GLM-4.6V",
        "qwen3-vl-8b-thinking": "Qwen3-VL-8B-Thinking",
    }

    for jsonl_file in tqdm(jsonl_files, desc="Scanning Logs"):
        # Error files do not contain usable per-case audit traces.
        if "errors" in jsonl_file.name:
            continue

        stem = jsonl_file.stem
        parts = stem.split('_')
        if len(parts) < 3:
            continue

        # System / dataset / model identifiers are encoded in the filename and
        # normalized here to match paper-facing names.
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
                                # The raw logs use a mixture of string and
                                # boolean statuses. We collapse everything to a
                                # strict binary failure indicator here.
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


def plot_failure_mode_111_enhanced(code, df_mode, output_dir):
    """
    Generate the paper-tuned Phase I figure and its appendix tables.

    This renderer is intentionally more bespoke than the generic figure builder:
    - Panel a uses six mini-matrices (2 columns x 3 rows) so the MAS section
      fills the top-left quadrant efficiently.
    - Panel b uses mirrored cumulative stacked areas to contrast text QA
      datasets (positive direction) against VQA datasets (negative direction).
    - Panels c/d show absolute failure rates, 95% CIs, and step-to-step change
      vectors in a single view.
    - Separate appendix tables preserve exact numbers without overloading the
      main figure.

    Reuse this function only when another failure mode also needs paper-level
    manual layout edits, such as uneven panel widths, panel-letter-only titles,
    or custom legend / colorbar alignment. If a new mode only needs standard
    labels, colors, or ordering updates, prefer editing the comprehensive path
    instead of cloning this bespoke renderer.
    """
    config = AUDIT_CONFIG[code]
    mode_name = config["name"]

    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found.")
        return

    df_mode = apply_failure_mode_step_filter(code, df_mode)
    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found after step filtering.")
        return

    # Build the x-axis directly from the data so the figure remains valid even
    # if future audits contain fewer rounds or only a subset of steps.
    unique_round_stages = df_mode[['round_num', 'step', 'round_stage']].drop_duplicates()
    unique_round_stages['step_idx'] = unique_round_stages['step'].map({step: idx for idx, step in enumerate(STAGE_ORDER)})
    unique_round_stages = unique_round_stages.sort_values(['round_num', 'step_idx'])
    x_order = unique_round_stages['round_stage'].tolist()

    if not x_order:
        return

    df_mode = df_mode.copy()
    df_mode['round_stage'] = pd.Categorical(df_mode['round_stage'], categories=x_order, ordered=True)

    qa_datasets = ["MedQA", "PubMedQA", "MedXpertQA"]
    vqa_datasets = ["PathVQA", "VQA-RAD", "SLAKE"]
    datasets_present = df_mode['dataset'].dropna().unique().tolist()
    qa_present = [dataset for dataset in qa_datasets if dataset in datasets_present]
    vqa_present = [dataset for dataset in vqa_datasets if dataset in datasets_present]
    panel_b_datasets = qa_present + vqa_present

    # Paper-facing short names used only in legends for the bespoke Phase I
    # figure. The statistics still key off the full normalized model IDs.
    model_display_names = {
        "DeepSeek-V3.2-Thinking": "DeepSeek-V3.2",
        "GPT-5.2": "GPT-5.2",
        "Gemini-3-Flash-Preview": "Gemini-3-Flash",
        "Qwen3-8B": "Qwen-3",
        "GLM-4.6V": "GLM-4.6V",
        "Qwen3-VL-8B-Thinking": "Qwen-3VL",
    }

    display_x_order = [stage_label.replace("Svnthesis", "Synthesis") for stage_label in x_order]
    x_numeric = np.arange(len(x_order))
    delta_cmap = plt.cm.coolwarm
    delta_norm = mcolors.CenteredNorm(vcenter=0, halfrange=10)

    # Keep exported PDF/SVG text editable in Illustrator. For reuse, typography
    # defaults that should affect the whole bespoke figure belong here instead
    # of being scattered panel by panel.
    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
            'font.weight': 'normal',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none',
            'text.usetex': False,
            'axes.linewidth': 1.4,
            'axes.labelsize': 18,
            'axes.labelweight': 'normal',
            'axes.titlesize': 24,
            'axes.titleweight': 'normal',
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 14,
            'legend.title_fontsize': 14,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )

    def build_series_arrays(panel_stats, series_name):
        # Convert sparse `(series, stage) -> stats` dictionaries into aligned
        # NumPy arrays so CI bands and arrows can be drawn stage by stage.
        y_vals = []
        ci_lower = []
        ci_upper = []
        for stage in x_order:
            cell_stats = panel_stats.get((series_name, stage))
            if cell_stats is None:
                y_vals.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
                continue
            lower, upper = cell_stats['ci_95']
            y_vals.append(cell_stats['failure_rate'])
            ci_lower.append(lower)
            ci_upper.append(upper)
        return np.asarray(y_vals, dtype=float), np.asarray(ci_lower, dtype=float), np.asarray(ci_upper, dtype=float)

    def plot_dataset_divergence_panel(ax, panel_stats):
        def style_decorated_axes(target_ax, x_pad=8, add_y_minor=True):
            # Apply the lighter axis treatment requested for the final paper
            # figure so panel b matches panels c/d stylistically.
            for spine_name in ['left', 'bottom']:
                target_ax.spines[spine_name].set_linewidth(1.0)
                target_ax.spines[spine_name].set_color('#334155')

            target_ax.tick_params(
                axis='x',
                which='major',
                width=0.95,
                length=4.2,
                pad=x_pad,
                direction='out',
                color='#475569',
            )
            target_ax.tick_params(
                axis='y',
                which='major',
                width=0.95,
                length=4.2,
                direction='out',
                color='#475569',
            )
            if add_y_minor:
                target_ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                target_ax.tick_params(
                    axis='y',
                    which='minor',
                    width=0.75,
                    length=2.4,
                    direction='out',
                    color='#64748B',
                )

        # Text QA datasets accumulate upward from zero, while VQA datasets
        # accumulate downward. The mirrored layout makes the modality split
        # visible at a glance without using separate subplots.
        qa_bottom = np.zeros(len(x_order))
        for dataset in qa_present:
            y_vals = np.array(
                [panel_stats.get((dataset, stage), {}).get('failure_rate', 0.0) for stage in x_order],
                dtype=float,
            )
            ax.fill_between(
                x_numeric,
                qa_bottom,
                qa_bottom + y_vals,
                color=ENHANCED_DATASET_COLOR_MAP[dataset],
                alpha=0.92,
                edgecolor='white',
                linewidth=1.5,
                zorder=2,
            )
            qa_bottom += y_vals

        vqa_top = np.zeros(len(x_order))
        for dataset in vqa_present:
            y_vals = np.array(
                [panel_stats.get((dataset, stage), {}).get('failure_rate', 0.0) for stage in x_order],
                dtype=float,
            )
            ax.fill_between(
                x_numeric,
                vqa_top,
                vqa_top - y_vals,
                color=ENHANCED_DATASET_COLOR_MAP[dataset],
                alpha=0.92,
                edgecolor='white',
                linewidth=1.5,
                zorder=2,
            )
            vqa_top -= y_vals

        for x_val in x_numeric:
            ax.axvline(x_val, color='#D8DEE6', linestyle='--', linewidth=0.8, alpha=0.42, zorder=1)

        ax.axhline(0, color='#334155', linewidth=1.2, zorder=5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y_val, pos: f"{abs(y_val):.0f}%"))
        ax.set_xlim(x_numeric[0] - 0.25, x_numeric[-1] + 0.25)
        ax.set_ylabel("Accumulated Failure Representation", fontweight='normal', fontsize=18)
        # In-figure panel titles are intentionally reduced to bare panel
        # letters. The longer panel names are documented outside the figure in
        # `prompt/appendix/重构audit_stas_figure/audit_stas_figure的4个panel的名称.md`.
        ax.set_title(
            "b",
            fontweight='bold',
            fontsize=24,
            loc='left',
            pad=16,
            y=1.02,
            linespacing=1.08,
        )
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(
            display_x_order,
            rotation=0,
            ha='center',
            fontweight='normal',
            fontsize=16,
        )
        ax.grid(axis='y', linestyle=':', alpha=0.25)
        style_decorated_axes(ax, x_pad=8, add_y_minor=True)

        max_extent = max(
            float(qa_bottom.max()) if qa_bottom.size else 0.0,
            float(np.abs(vqa_top.min())) if vqa_top.size else 0.0,
        )
        if max_extent > 0:
            ax.set_ylim(-max_extent * 1.08, max_extent * 1.08)
        else:
            ax.set_ylim(-5, 5)

        handles = [
            mpatches.Patch(facecolor=ENHANCED_DATASET_COLOR_MAP[dataset], edgecolor='none', label=dataset)
            for dataset in panel_b_datasets
        ]
        if handles:
            ax.legend(
                handles=handles,
                loc='upper left',
                bbox_to_anchor=(0.0, 1.015),
                frameon=True,
                fancybox=False,
                framealpha=0.96,
                facecolor='white',
                edgecolor='#CBD5E1',
                ncol=2,
                borderpad=0.55,
                labelspacing=0.55,
                handletextpad=0.65,
                columnspacing=1.6,
                handlelength=1.6,
            )

    def plot_kinematic_panel(ax, panel_stats, series_order, color_map, title, display_names=None):
        # Panels c/d encode three quantities at once:
        # 1. marker position = absolute failure rate
        # 2. translucent ribbon = 95% CI
        # 3. arrow color / thickness = signed step-to-step change magnitude
        max_panel_rate = max(
            (panel_stats[(series, stage)]['failure_rate'] for series in series_order for stage in x_order if (series, stage) in panel_stats),
            default=0.0,
        )
        y_min, y_max, tick_step = compute_adaptive_rate_axis(max_panel_rate)

        for x_val in x_numeric:
            ax.axvline(x_val, color='#E5E7EB', linestyle='--', linewidth=0.8, alpha=0.48, zorder=0)

        handles = []
        labels = []
        prepared_series = []

        for idx, series_name in enumerate(series_order):
            y_vals, ci_lower, ci_upper = build_series_arrays(panel_stats, series_name)
            valid_mask = ~np.isnan(y_vals)
            if not valid_mask.any():
                continue

            color = color_map[series_name]
            marker = SERIES_MARKERS[idx % len(SERIES_MARKERS)]
            display_label = display_names.get(series_name, series_name) if display_names else series_name

            prepared_series.append(
                {
                    'idx': idx,
                    'series_name': series_name,
                    'display_label': display_label,
                    'color': color,
                    'marker': marker,
                    'y_vals': y_vals,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'valid_mask': valid_mask,
                }
            )

        for series in prepared_series:
            if np.count_nonzero(series['valid_mask']) >= 2:
                # Draw CI underneath the arrows and markers so the central
                # trajectory remains visually dominant.
                ax.fill_between(
                    x_numeric[series['valid_mask']],
                    series['ci_lower'][series['valid_mask']],
                    series['ci_upper'][series['valid_mask']],
                    color=series['color'],
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )

        for series in prepared_series:
            scatter = ax.scatter(
                x_numeric[series['valid_mask']],
                series['y_vals'][series['valid_mask']],
                color=series['color'],
                s=320,
                zorder=6,
                edgecolors='white',
                lw=1.8,
                marker=series['marker'],
                label=series['display_label'],
            )
            handles.append(scatter)
            labels.append(series['display_label'])

            for stage_idx in range(len(x_order) - 1):
                if np.isnan(series['y_vals'][stage_idx]) or np.isnan(series['y_vals'][stage_idx + 1]):
                    continue

                delta = series['y_vals'][stage_idx + 1] - series['y_vals'][stage_idx]
                segment_color = delta_cmap(delta_norm(delta))
                segment_lw = 2.2 + abs(delta) / 5.5

                # No base line is drawn below the arrows: the arrow itself is
                # the only trajectory cue, which avoids implying an additional
                # independent series.
                ax.annotate(
                    "",
                    xy=(x_numeric[stage_idx + 1], series['y_vals'][stage_idx + 1]),
                    xytext=(x_numeric[stage_idx], series['y_vals'][stage_idx]),
                    arrowprops={
                        'arrowstyle': "->",
                        'color': segment_color,
                        'lw': segment_lw,
                        'shrinkA': 9,
                        'shrinkB': 9,
                        'mutation_scale': 18,
                    },
                    zorder=5,
                )

        ax.set_title(
            title,
            fontweight='bold',
            fontsize=22,
            loc='left',
            pad=16,
            y=1.02,
            linespacing=1.08,
        )
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(
            display_x_order,
            rotation=0,
            ha='center',
            fontweight='normal',
            fontsize=14,
        )
        ax.set_xlim(x_numeric[0] - 0.2, x_numeric[-1] + 0.2)
        ax.set_ylabel("Absolute Failure Rate (%)", fontweight='normal', fontsize=18)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.001, tick_step))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis='y', linestyle=':', alpha=0.35)
        for spine_name in ['left', 'bottom']:
            ax.spines[spine_name].set_linewidth(1.0)
            ax.spines[spine_name].set_color('#334155')
        ax.tick_params(
            axis='x',
            which='major',
            width=0.95,
            length=4.2,
            pad=8,
            direction='out',
            color='#475569',
        )
        ax.tick_params(
            axis='y',
            which='major',
            width=0.95,
            length=4.2,
            direction='out',
            color='#475569',
        )
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(
            axis='y',
            which='minor',
            width=0.75,
            length=2.4,
            direction='out',
            color='#64748B',
        )

        if handles:
            legend_upper_left_codes = {"2.2.2", "3.1.1", "3.1.2"}
            legend_loc = 'upper left' if code in legend_upper_left_codes else 'lower left'
            legend_anchor = (0.0, 0.98) if code in legend_upper_left_codes else (0.0, 0.02)
            ax.legend(
                handles,
                labels,
                loc=legend_loc,
                bbox_to_anchor=legend_anchor,
                frameon=True,
                fancybox=False,
                framealpha=0.96,
                facecolor='white',
                edgecolor='#CBD5E1',
                ncol=2,
                borderpad=0.70,
                labelspacing=0.90,
                handletextpad=0.65,
                handlelength=1.8,
                columnspacing=1.6,
                markerscale=0.78,
            )

    def draw_appendix_table(ax, title, row_order, color_map, panel_stats, display_names=None):
        # The appendix tables intentionally mirror the same ordering as the
        # main figure, making it easy to cross-reference exact values.
        row_count = max(len(row_order), 1)
        row_step = 1.24
        y_positions = np.arange(row_count)[::-1] * row_step
        header_y = y_positions[0] + 0.92 if row_order else 0.92
        label_x = -1.55
        table_left = -0.5
        table_right = len(x_order) - 0.5

        ax.set_title(title, fontweight='normal', fontsize=22, loc='left', pad=14)

        for x_val in np.arange(-0.5, len(x_order) + 0.5, 1.0):
            ax.vlines(x_val, ymin=-0.65, ymax=header_y - 0.35, color='#E2E8F0', linewidth=0.9, zorder=0)

        ax.hlines(header_y - 0.35, xmin=table_left, xmax=table_right, color='#CBD5E1', linewidth=1.2, zorder=0)

        for x_val, stage_label in zip(x_numeric, display_x_order):
            ax.text(
                x_val,
                header_y,
                stage_label,
                ha='center',
                va='bottom',
                fontsize=12.5,
                fontweight='normal',
                rotation=28,
                rotation_mode='anchor',
            )

        for idx, row_name in enumerate(row_order):
            y_val = y_positions[idx]
            display_label = display_names.get(row_name, row_name) if display_names else row_name
            ax.hlines(y_val - row_step / 2, xmin=table_left, xmax=table_right, color='#F1F5F9', linewidth=0.9, zorder=0)
            ax.text(
                label_x,
                y_val,
                display_label,
                ha='right',
                va='center',
                fontsize=13.0,
                fontweight='normal',
                color=color_map[row_name],
            )

            for x_val, stage in zip(x_numeric, x_order):
                cell_stats = panel_stats.get((row_name, stage))
                if cell_stats is None:
                    cell_text = "N/A"
                    cell_color = '#9CA3AF'
                else:
                    ci_lower, ci_upper = cell_stats['ci_95']
                    cell_text = (
                        f"{cell_stats['failure_rate']:.1f}%\n"
                        f"{cell_stats['failed_count']}/{cell_stats['total_count']}\n"
                        f"[{ci_lower:.1f}, {ci_upper:.1f}]"
                    )
                    cell_color = '#111827'

                ax.text(
                    x_val,
                    y_val,
                    cell_text,
                    ha='center',
                    va='center',
                    fontsize=10.0,
                    color=cell_color,
                    linespacing=1.05,
                )

        if row_order:
            y_bottom = y_positions[-1] - row_step / 2 - 0.08
            y_top = header_y + 0.38
        else:
            y_bottom, y_top = -0.65, 1.25

        ax.set_xlim(label_x - 0.18, table_right + 0.25)
        ax.set_ylim(y_bottom, y_top)
        ax.axis('off')

    mas_priority = ["ColaCare", "HealthcareAgent", "MAC", "MDAgents", "MedAgents", "ReConcile"]
    mas_present = df_mode['mas'].dropna().unique().tolist()
    mas_order = [mas for mas in mas_priority if mas in mas_present]
    mas_order.extend(sorted(mas for mas in mas_present if mas not in mas_order))

    # Precompute all panel statistics before any plotting. This keeps the
    # rendering layer mostly declarative and makes debugging easier.
    panel_a_stats, panel_a_max = collect_group_stage_stats(df_mode, 'mas', mas_order, x_order, seed_offset=0)
    panel_b_stats, _ = collect_group_stage_stats(df_mode, 'dataset', panel_b_datasets, x_order, seed_offset=5000)
    df_c = df_mode[df_mode['dataset'].isin(qa_datasets)].copy()
    panel_c_models = [llm for llm in LLM_PRIORITY if llm in df_c['llm'].dropna().unique().tolist()]
    panel_c_stats, _ = collect_group_stage_stats(df_c, 'llm', panel_c_models, x_order, seed_offset=10000)
    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    panel_d_models = [llm for llm in LLM_PRIORITY if llm in df_d['llm'].dropna().unique().tolist()]
    panel_d_stats, _ = collect_group_stage_stats(df_d, 'llm', panel_d_models, x_order, seed_offset=15000)
    panel_c_color_values = sns.color_palette("Set1", n_colors=max(len(panel_c_models), 1))
    panel_c_color_map = dict(zip(panel_c_models, panel_c_color_values))
    panel_d_color_values = sns.color_palette("Dark2", n_colors=max(len(panel_d_models), 1))
    panel_d_color_map = dict(zip(panel_d_models, panel_d_color_values))

    fig = plt.figure(figsize=(24.8, 16.8))
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, wspace=0.16, hspace=0.28, height_ratios=[1.0, 1.0])

    # Panel a
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_axis_off()

    panel_a_cmap = mcolors.LinearSegmentedColormap.from_list("panel_a_gray_purple", PALETTE_1)
    _, panel_a_top, panel_a_tick = compute_adaptive_rate_axis(panel_a_max)
    panel_a_norm = mcolors.Normalize(vmin=0, vmax=panel_a_top)
    mini_x_step = 1.03
    mini_box_width = 0.82
    mini_box_height = 0.82
    mini_x_positions = np.arange(len(x_order)) * mini_x_step
    # Each MAS gets its own inset axis so the top-left quadrant can be filled
    # as a 2 x 3 grid, rather than wasting space on one large sparse matrix.
    panel_a_group_positions = [
        (0.00, 0.69),
        (0.46, 0.69),
        (0.00, 0.385),
        (0.46, 0.385),
        (0.00, 0.08),
        (0.46, 0.08),
    ]
    panel_a_group_width = 0.45
    panel_a_group_height = 0.20

    # Keep only the panel letter inside the figure; see the appendix prompt
    # note above for the manuscript-facing full panel names.
    ax_a.text(
        0.0,
        1.05,
        "a",
        transform=ax_a.transAxes,
        ha='left',
        va='bottom',
        fontweight='bold',
        fontsize=24,
        clip_on=False,
    )

    na_patch = mpatches.Patch(
        facecolor='#F8FAFC',
        edgecolor='#94A3B8',
        linestyle='--',
        hatch='////',
        label='Step Not Present',
    )
    ax_a.legend(
        handles=[na_patch],
        loc='upper left',
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor='white',
        edgecolor='#CBD5E1',
        borderpad=0.45,
        fontsize=14,
    )

    for (left, bottom), mas_name in zip(panel_a_group_positions, mas_order):
        ax_mini = ax_a.inset_axes(
            [left, bottom, panel_a_group_width, panel_a_group_height],
            transform=ax_a.transAxes,
        )
        ax_mini.set_title(mas_name, loc='left', fontsize=16, fontweight='normal', pad=4)

        for x_val, stage in zip(mini_x_positions, x_order):
            cell_stats = panel_a_stats.get((mas_name, stage))
            if cell_stats is None:
                # Some MASes structurally do not expose later stages. These are
                # rendered explicitly as N/A instead of being left blank.
                box = mpatches.Rectangle(
                    (x_val - mini_box_width / 2, -mini_box_height / 2),
                    mini_box_width,
                    mini_box_height,
                    facecolor='#F8FAFC',
                    edgecolor='#94A3B8',
                    linewidth=1.5,
                    linestyle='--',
                    hatch='////',
                    zorder=1,
                )
                ax_mini.add_patch(box)
                ax_mini.text(
                    x_val,
                    0,
                    "N/A",
                    ha='center',
                    va='center',
                    color='#94A3B8',
                    fontsize=10.2,
                    fontweight='normal',
                    zorder=2,
                )
                continue

            failure_rate = cell_stats['failure_rate']
            failed_count = cell_stats['failed_count']
            total_count = cell_stats['total_count']
            ci_lower, ci_upper = cell_stats['ci_95']
            face_color = panel_a_cmap(panel_a_norm(failure_rate))
            box = mpatches.Rectangle(
                (x_val - mini_box_width / 2, -mini_box_height / 2),
                mini_box_width,
                mini_box_height,
                facecolor=face_color,
                edgecolor='white',
                linewidth=2.0,
                zorder=1,
            )
            ax_mini.add_patch(box)

            luminance = 0.299 * face_color[0] + 0.587 * face_color[1] + 0.114 * face_color[2]
            text_color = 'white' if luminance < 0.62 else '#111827'
            ax_mini.text(
                x_val,
                0.16,
                f"{failure_rate:.1f}%",
                ha='center',
                va='center',
                color=text_color,
                fontsize=11.0,
                fontweight='normal',
                zorder=2,
            )
            ax_mini.text(
                x_val,
                0.0,
                f"{failed_count}/{total_count}",
                ha='center',
                va='center',
                color=text_color,
                fontsize=9.4,
                fontweight='normal',
                zorder=2,
            )
            ax_mini.text(
                x_val,
                -0.16,
                f"CI {ci_lower:.1f}-{ci_upper:.1f}",
                ha='center',
                va='center',
                color=text_color,
                fontsize=8.2,
                fontweight='normal',
                zorder=2,
            )

        ax_mini.set_xlim(mini_x_positions[0] - 0.55, mini_x_positions[-1] + 0.55)
        ax_mini.set_ylim(-0.56, 0.56)
        ax_mini.set_aspect('equal', adjustable='box')
        ax_mini.set_xticks(mini_x_positions)
        ax_mini.set_xticklabels(
            display_x_order,
            rotation=0,
            ha='center',
            fontweight='normal',
            fontsize=11.5,
        )
        ax_mini.set_yticks([])
        ax_mini.tick_params(axis='x', length=0, pad=8)
        for spine_name in ['top', 'right', 'left', 'bottom']:
            ax_mini.spines[spine_name].set_visible(False)

    cax_a = ax_a.inset_axes([0.945, 0.17, 0.024, 0.64], transform=ax_a.transAxes)
    sm_a = plt.cm.ScalarMappable(cmap=panel_a_cmap, norm=panel_a_norm)
    sm_a.set_array([])
    cbar_a = plt.colorbar(sm_a, cax=cax_a)
    cbar_a.set_label(
        'Failure Rate (%)',
        rotation=270,
        labelpad=12,
        fontsize=15,
        fontweight='normal',
    )
    cbar_ticks = np.arange(0, panel_a_top + 0.001, panel_a_tick)
    cbar_a.set_ticks(cbar_ticks)
    cbar_a.ax.yaxis.set_label_position('left')
    cbar_a.ax.yaxis.set_ticks_position('right')
    cbar_a.ax.tick_params(labelsize=13)
    cbar_a.outline.set_visible(False)

    # Panel b
    # The subplot calls below are the safest place to swap panel-letter labels
    # or redirect a panel to a different helper without touching the data prep.
    ax_b = fig.add_subplot(gs[0, 1])
    plot_dataset_divergence_panel(ax_b, panel_b_stats)

    # Panels c and d
    ax_c = fig.add_subplot(gs[1, 0])
    plot_kinematic_panel(
        ax_c,
        panel_c_stats,
        panel_c_models,
        panel_c_color_map,
        "c",
        display_names=model_display_names,
    )

    ax_d = fig.add_subplot(gs[1, 1])
    plot_kinematic_panel(
        ax_d,
        panel_d_stats,
        panel_d_models,
        panel_d_color_map,
        "d",
        display_names=model_display_names,
    )

    plt.subplots_adjust(left=0.065, right=0.92, top=0.96, bottom=0.08)
    fig.canvas.draw()

    # A small amount of manual post-layout alignment is used here because the
    # publication figure mixes regular subplots, inset axes, and floating
    # shared colorbars. When reusing this renderer for another failure mode,
    # treat the constants in this block as layout knobs rather than data logic:
    # `target_right_left` moves the right column, while the width-scale knobs
    # below compress selected panels without changing their left alignment.
    ax_a_pos = ax_a.get_position()
    ax_b_pos = ax_b.get_position()
    ax_c_pos = ax_c.get_position()
    ax_d_pos = ax_d.get_position()

    target_a_left = ax_c_pos.x0
    target_right_left = 0.495
    # This local width scale only affects the bespoke Phase I panel-c layout.
    # Reuse it when the legend / shared colorbar balance needs more whitespace
    # without changing the underlying data or x-axis labels. The current value
    # follows the clarified request: 85% of the previous 0.80 setting, i.e.
    # 68% of the original GridSpec width.
    panel_c_width_scale = 0.68
    # The `1.2.1` revision keeps the same left anchor for the right column but
    # shortens the panel-b and panel-d x-axis span to 80% of the previous
    # width, matching the requested tighter layout.
    panel_b_width_scale = 0.80 if code == "1.2.1" else 1.0
    panel_d_width_scale = 0.80 if code == "1.2.1" else 1.0

    ax_a.set_position([target_a_left, ax_a_pos.y0, ax_a_pos.width, ax_a_pos.height])
    ax_b.set_position([target_right_left, ax_b_pos.y0, ax_b_pos.width * panel_b_width_scale, ax_b_pos.height])
    ax_c.set_position([ax_c_pos.x0, ax_c_pos.y0, ax_c_pos.width * panel_c_width_scale, ax_c_pos.height])
    ax_d.set_position([target_right_left, ax_d_pos.y0, ax_d_pos.width * panel_d_width_scale, ax_d_pos.height])
    fig.canvas.draw()

    ax_a_pos = ax_a.get_position()
    ax_c_pos = ax_c.get_position()
    ax_d_pos = ax_d.get_position()
    column_gap = max(ax_d_pos.x0 - ax_c_pos.x1, 0.02)
    shared_cbar_width = min(0.012, column_gap * 0.30)
    shared_cbar_pad = max(0.0035, min(0.006, (column_gap - shared_cbar_width) * 0.22))
    shared_cbar_left = ax_c_pos.x1 + max(0.001, shared_cbar_pad - 0.0045)
    # Place the panel-a colorbar and the shared c/d colorbar on the same
    # middle column so the page reads as a clean two-column composition.
    cbar_a_height = ax_a_pos.height * 0.64
    cbar_a_bottom = ax_a_pos.y0 + ax_a_pos.height * 0.17
    cbar_a.ax.set_position([shared_cbar_left, cbar_a_bottom, shared_cbar_width, cbar_a_height])
    fig.canvas.draw()

    kinematic_axes = [axis for axis in (ax_c, ax_d) if axis is not None]
    kinematic_top = max(axis.get_position().y1 for axis in kinematic_axes)
    kinematic_height = max(axis.get_position().height for axis in kinematic_axes)
    cbar_height = kinematic_height * 0.82
    cbar_bottom = kinematic_top - cbar_height
    cax_delta = fig.add_axes([shared_cbar_left, cbar_bottom, shared_cbar_width, cbar_height])
    sm_delta = plt.cm.ScalarMappable(cmap=delta_cmap, norm=delta_norm)
    sm_delta.set_array([])
    cbar_delta = plt.colorbar(sm_delta, cax=cax_delta)
    cbar_delta.set_label(
        'Arrow Color = Step-to-Step Change (Delta %)',
        rotation=270,
        labelpad=14,
        fontweight='normal',
        fontsize=16,
    )
    cbar_delta.set_ticks([-10, -5, 0, 5, 10])
    cbar_delta.ax.yaxis.set_label_position('left')
    cbar_delta.ax.yaxis.set_ticks_position('left')
    cbar_delta.ax.tick_params(labelsize=14)
    cbar_delta.outline.set_visible(False)

    main_pdf_path, main_png_path = save_figure_variants(fig, output_dir, f"failure_mode_{code}")
    plt.close(fig)
    print(f"Saved enhanced main figure: {main_pdf_path}")
    print(f"Saved enhanced main figure PNG: {main_png_path}")


def plot_failure_mode_212_enhanced(code, df_mode, output_dir):
    """
    Generate the appendix-style main-text figure for dense multi-stage modes.

    This variant is currently used for Failure Modes `2.1.2`, `2.2.1`,
    `2.2.2`, `3.1.1`, `3.1.2`, and `3.1.3`.
    It keeps Panel a in the dense matrix layout that already fits the five- or
    six-stage Phase II pipelines well, but applies the same paper-facing
    cleanup used by the Phase I figures:
    - Panel titles are reduced to panel letters only.
    - Exact numbers under Panels b--d are removed from the figure and shifted to
      appendix tables in the manuscript.
    - Panels c/d add translucent 95% CI ribbons so the trajectory uncertainty
      remains visible after the numeric tables are removed.
    """
    config = AUDIT_CONFIG[code]
    mode_name = config["name"]

    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found.")
        return

    df_mode = apply_failure_mode_step_filter(code, df_mode)
    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found after step filtering.")
        return

    unique_round_stages = df_mode[['round_num', 'step', 'round_stage']].drop_duplicates()
    unique_round_stages['step_idx'] = unique_round_stages['step'].map({step: idx for idx, step in enumerate(STAGE_ORDER)})
    unique_round_stages = unique_round_stages.sort_values(['round_num', 'step_idx'])
    x_order = unique_round_stages['round_stage'].tolist()

    if not x_order:
        return

    df_mode = df_mode.copy()
    df_mode['round_stage'] = pd.Categorical(df_mode['round_stage'], categories=x_order, ordered=True)

    qa_datasets = ["MedQA", "PubMedQA", "MedXpertQA"]
    vqa_datasets = ["PathVQA", "VQA-RAD", "SLAKE"]
    datasets_present = df_mode['dataset'].dropna().unique().tolist()
    qa_present = [dataset for dataset in qa_datasets if dataset in datasets_present]
    vqa_present = [dataset for dataset in vqa_datasets if dataset in datasets_present]
    panel_b_datasets = qa_present + vqa_present

    model_display_names = {
        "DeepSeek-V3.2-Thinking": "DeepSeek-V3.2",
        "GPT-5.2": "GPT-5.2",
        "Gemini-3-Flash-Preview": "Gemini-3-Flash",
        "Qwen3-8B": "Qwen-3",
        "GLM-4.6V": "GLM-4.6V",
        "Qwen3-VL-8B-Thinking": "Qwen-3VL",
    }

    display_x_order = [stage_label.replace("-", "\n", 1).replace("Svnthesis", "Synthesis") for stage_label in x_order]
    x_numeric = np.arange(len(x_order))
    delta_cmap = plt.cm.coolwarm
    delta_norm = mcolors.CenteredNorm(vcenter=0, halfrange=20)

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
            'font.weight': 'normal',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none',
            'text.usetex': False,
            'axes.linewidth': 1.4,
            'axes.labelsize': 16,
            'axes.labelweight': 'normal',
            'axes.titlesize': 24,
            'axes.titleweight': 'normal',
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 13,
            'legend.title_fontsize': 13,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )

    def build_series_arrays(panel_stats, series_name):
        y_vals = []
        ci_lower = []
        ci_upper = []
        for stage in x_order:
            cell_stats = panel_stats.get((series_name, stage))
            if cell_stats is None:
                y_vals.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
                continue
            lower, upper = cell_stats['ci_95']
            y_vals.append(cell_stats['failure_rate'])
            ci_lower.append(lower)
            ci_upper.append(upper)
        return np.asarray(y_vals, dtype=float), np.asarray(ci_lower, dtype=float), np.asarray(ci_upper, dtype=float)

    def plot_dataset_divergence_panel(ax, panel_stats):
        qa_bottom = np.zeros(len(x_order))
        for dataset in qa_present:
            y_vals = np.array(
                [panel_stats.get((dataset, stage), {}).get('failure_rate', 0.0) for stage in x_order],
                dtype=float,
            )
            ax.fill_between(
                x_numeric,
                qa_bottom,
                qa_bottom + y_vals,
                color=ENHANCED_DATASET_COLOR_MAP[dataset],
                alpha=0.92,
                edgecolor='white',
                linewidth=1.5,
                zorder=2,
            )
            qa_bottom += y_vals

        vqa_top = np.zeros(len(x_order))
        for dataset in vqa_present:
            y_vals = np.array(
                [panel_stats.get((dataset, stage), {}).get('failure_rate', 0.0) for stage in x_order],
                dtype=float,
            )
            ax.fill_between(
                x_numeric,
                vqa_top,
                vqa_top - y_vals,
                color=ENHANCED_DATASET_COLOR_MAP[dataset],
                alpha=0.92,
                edgecolor='white',
                linewidth=1.5,
                zorder=2,
            )
            vqa_top -= y_vals

        for x_val in x_numeric:
            ax.axvline(x_val, color='#D8DEE6', linestyle='--', linewidth=0.8, alpha=0.42, zorder=1)

        ax.axhline(0, color='#334155', linewidth=1.2, zorder=5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y_val, pos: f"{abs(y_val):.0f}%"))
        ax.set_xlim(x_numeric[0] - 0.25, x_numeric[-1] + 0.25)
        ax.set_ylabel("Accumulated Failure Representation", fontweight='normal', fontsize=16)
        ax.set_title("b", fontweight='bold', fontsize=24, loc='left', pad=14, y=1.01)
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(
            display_x_order,
            rotation=0,
            ha='center',
            fontweight='normal',
            fontsize=12.5,
        )
        ax.grid(axis='y', linestyle=':', alpha=0.25)

        for spine_name in ['left', 'bottom']:
            ax.spines[spine_name].set_linewidth(1.0)
            ax.spines[spine_name].set_color('#334155')

        ax.tick_params(axis='x', which='major', width=0.95, length=4.2, pad=10, direction='out', color='#475569')
        ax.tick_params(axis='y', which='major', width=0.95, length=4.2, direction='out', color='#475569')
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(axis='y', which='minor', width=0.75, length=2.4, direction='out', color='#64748B')

        max_extent = max(
            float(qa_bottom.max()) if qa_bottom.size else 0.0,
            float(np.abs(vqa_top.min())) if vqa_top.size else 0.0,
        )
        if max_extent > 0:
            ax.set_ylim(-max_extent * 1.08, max_extent * 1.08)
        else:
            ax.set_ylim(-5, 5)

        handles = [
            mpatches.Patch(facecolor=ENHANCED_DATASET_COLOR_MAP[dataset], edgecolor='none', label=dataset)
            for dataset in panel_b_datasets
        ]
        if handles:
            ax.legend(
                handles=handles,
                loc='upper left',
                bbox_to_anchor=(0.0, 1.015),
                frameon=True,
                fancybox=False,
                framealpha=0.96,
                facecolor='white',
                edgecolor='#CBD5E1',
                ncol=2,
                borderpad=0.55,
                labelspacing=0.55,
                handletextpad=0.65,
                columnspacing=1.6,
                handlelength=1.6,
            )

    def plot_kinematic_panel(ax, panel_stats, series_order, color_map, title, display_names=None):
        for x_val in x_numeric:
            ax.axvline(x_val, color='#E5E7EB', linestyle='--', linewidth=0.8, alpha=0.48, zorder=0)

        handles = []
        labels = []
        prepared_series = []

        for idx, series_name in enumerate(series_order):
            y_vals, ci_lower, ci_upper = build_series_arrays(panel_stats, series_name)
            valid_mask = ~np.isnan(y_vals)
            if not valid_mask.any():
                continue

            color = color_map[series_name]
            marker = SERIES_MARKERS[idx % len(SERIES_MARKERS)]
            display_label = display_names.get(series_name, series_name) if display_names else series_name

            prepared_series.append(
                {
                    'display_label': display_label,
                    'color': color,
                    'marker': marker,
                    'y_vals': y_vals,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'valid_mask': valid_mask,
                }
            )

        max_panel_upper = max(
            (
                float(np.nanmax(series['ci_upper'][series['valid_mask']]))
                for series in prepared_series
                if np.count_nonzero(series['valid_mask']) > 0
            ),
            default=0.0,
        )
        _, y_max, tick_step = compute_adaptive_rate_axis(max_panel_upper)

        robust_lower_candidates = []
        for series in prepared_series:
            if not np.count_nonzero(series['valid_mask']):
                continue

            y_valid = series['y_vals'][series['valid_mask']]
            ci_lower_valid = series['ci_lower'][series['valid_mask']]
            # Wide CIs from tiny denominators should remain visible, but they
            # should not force the entire panel back to a 0%-anchored axis when
            # all observed trajectories already live in the upper range.
            robust_lower_candidates.append(np.maximum(ci_lower_valid, y_valid - 2 * tick_step))

        y_min = 0.0
        if robust_lower_candidates:
            robust_lower = float(np.nanmin(np.concatenate(robust_lower_candidates)))
            visible_span = max_panel_upper - robust_lower
            if robust_lower >= 25.0 and visible_span <= 80.0:
                lower_pad = max(4.0, min(tick_step * 1.5, visible_span * 0.12))
                y_min = tick_step * np.floor(max(0.0, robust_lower - lower_pad) / tick_step)
                if y_min < tick_step:
                    y_min = 0.0

        for series in prepared_series:
            if np.count_nonzero(series['valid_mask']) >= 2:
                ax.fill_between(
                    x_numeric[series['valid_mask']],
                    series['ci_lower'][series['valid_mask']],
                    series['ci_upper'][series['valid_mask']],
                    color=series['color'],
                    alpha=0.10,
                    linewidth=0,
                    zorder=1,
                )

        for series in prepared_series:
            scatter = ax.scatter(
                x_numeric[series['valid_mask']],
                series['y_vals'][series['valid_mask']],
                color=series['color'],
                s=285,
                zorder=6,
                edgecolors='white',
                lw=1.8,
                marker=series['marker'],
                label=series['display_label'],
            )
            handles.append(scatter)
            labels.append(series['display_label'])

            for stage_idx in range(len(x_order) - 1):
                if np.isnan(series['y_vals'][stage_idx]) or np.isnan(series['y_vals'][stage_idx + 1]):
                    continue

                delta = series['y_vals'][stage_idx + 1] - series['y_vals'][stage_idx]
                segment_color = delta_cmap(delta_norm(delta))
                segment_lw = 2.2 + abs(delta) / 5.5

                ax.annotate(
                    "",
                    xy=(x_numeric[stage_idx + 1], series['y_vals'][stage_idx + 1]),
                    xytext=(x_numeric[stage_idx], series['y_vals'][stage_idx]),
                    arrowprops={
                        'arrowstyle': "->",
                        'color': segment_color,
                        'lw': segment_lw,
                        'shrinkA': 9,
                        'shrinkB': 9,
                        'mutation_scale': 18,
                    },
                    zorder=5,
                )

        ax.set_title(title, fontweight='bold', fontsize=24, loc='left', pad=14, y=1.01)
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(
            display_x_order,
            rotation=0,
            ha='center',
            fontweight='normal',
            fontsize=12.5,
        )
        ax.set_xlim(x_numeric[0] - 0.2, x_numeric[-1] + 0.2)
        ax.set_ylabel("Absolute Failure Rate (%)", fontweight='normal', fontsize=16)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 0.001, tick_step))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis='y', linestyle=':', alpha=0.35)

        for spine_name in ['left', 'bottom']:
            ax.spines[spine_name].set_linewidth(1.0)
            ax.spines[spine_name].set_color('#334155')

        ax.tick_params(axis='x', which='major', width=0.95, length=4.2, pad=10, direction='out', color='#475569')
        ax.tick_params(axis='y', which='major', width=0.95, length=4.2, direction='out', color='#475569')
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(axis='y', which='minor', width=0.75, length=2.4, direction='out', color='#64748B')

        if handles:
            legend_upper_left_codes = {"2.2.2", "3.1.1", "3.1.2", "3.1.3"}
            legend_loc = 'upper left' if code in legend_upper_left_codes else 'lower left'
            legend_anchor = (0.0, 0.98) if code in legend_upper_left_codes else (0.0, 0.02)
            ax.legend(
                handles,
                labels,
                loc=legend_loc,
                bbox_to_anchor=legend_anchor,
                frameon=True,
                fancybox=False,
                framealpha=0.96,
                facecolor='white',
                edgecolor='#CBD5E1',
                ncol=2,
                borderpad=0.70,
                labelspacing=0.90,
                handletextpad=0.65,
                handlelength=1.8,
                columnspacing=1.6,
                markerscale=0.78,
            )

    mas_priority = ["ColaCare", "HealthcareAgent", "MAC", "MDAgents", "MedAgents", "ReConcile"]
    mas_present = df_mode['mas'].dropna().unique().tolist()
    mas_order = [mas for mas in mas_priority if mas in mas_present]
    mas_order.extend(sorted(mas for mas in mas_present if mas not in mas_order))

    panel_a_stats, panel_a_max = collect_group_stage_stats(df_mode, 'mas', mas_order, x_order, seed_offset=0)
    panel_b_stats, _ = collect_group_stage_stats(df_mode, 'dataset', panel_b_datasets, x_order, seed_offset=5000)

    df_c = df_mode[df_mode['dataset'].isin(qa_datasets)].copy()
    panel_c_models = [llm for llm in LLM_PRIORITY if llm in df_c['llm'].dropna().unique().tolist()]
    panel_c_stats, _ = collect_group_stage_stats(df_c, 'llm', panel_c_models, x_order, seed_offset=10000)

    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    panel_d_models = [llm for llm in LLM_PRIORITY if llm in df_d['llm'].dropna().unique().tolist()]
    panel_d_stats, _ = collect_group_stage_stats(df_d, 'llm', panel_d_models, x_order, seed_offset=15000)

    panel_c_color_values = sns.color_palette("Set1", n_colors=max(len(panel_c_models), 1))
    panel_c_color_map = dict(zip(panel_c_models, panel_c_color_values))
    panel_d_color_values = sns.color_palette("Dark2", n_colors=max(len(panel_d_models), 1))
    panel_d_color_map = dict(zip(panel_d_models, panel_d_color_values))

    fig = plt.figure(figsize=(24.6, 17.2))
    fig.patch.set_facecolor('white')
    gs = GridSpec(2, 2, figure=fig, wspace=0.18, hspace=0.28, height_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a_cmap = mcolors.LinearSegmentedColormap.from_list("panel_a_gray_purple", PALETTE_1)
    _, panel_a_top, panel_a_tick = compute_adaptive_rate_axis(panel_a_max)
    panel_a_norm = mcolors.Normalize(vmin=0, vmax=panel_a_top)

    x_positions = np.arange(len(x_order)) * 1.14
    y_positions = np.arange(len(mas_order))[::-1] * 1.12
    y_map = dict(zip(mas_order, y_positions))
    box_width = 0.94
    box_height = 0.94

    ax_a.set_title("a", fontweight='bold', fontsize=24, loc='left', pad=14, y=1.01)

    for y_val in y_positions:
        ax_a.hlines(y_val, xmin=x_positions[0], xmax=x_positions[-1], color='#D3D9E1', linewidth=2.0, zorder=1)

    for mas in mas_order:
        y_val = y_map[mas]
        for x_val, stage in zip(x_positions, x_order):
            cell_stats = panel_a_stats.get((mas, stage))

            if cell_stats is None:
                box = mpatches.Rectangle(
                    (x_val - box_width / 2, y_val - box_height / 2),
                    box_width,
                    box_height,
                    facecolor='#F8FAFC',
                    edgecolor='#94A3B8',
                    linewidth=1.5,
                    linestyle='--',
                    hatch='////',
                    zorder=2,
                )
                ax_a.add_patch(box)
                ax_a.text(
                    x_val,
                    y_val,
                    "N/A",
                    ha='center',
                    va='center',
                    color='#94A3B8',
                    fontsize=10.2,
                    fontweight='normal',
                    zorder=3,
                )
                continue

            failure_rate = cell_stats['failure_rate']
            failed_count = cell_stats['failed_count']
            total_count = cell_stats['total_count']
            ci_lower, ci_upper = cell_stats['ci_95']
            face_color = panel_a_cmap(panel_a_norm(failure_rate))

            box = mpatches.Rectangle(
                (x_val - box_width / 2, y_val - box_height / 2),
                box_width,
                box_height,
                facecolor=face_color,
                edgecolor='white',
                linewidth=2.0,
                zorder=2,
            )
            ax_a.add_patch(box)

            luminance = 0.299 * face_color[0] + 0.587 * face_color[1] + 0.114 * face_color[2]
            text_color = 'white' if luminance < 0.62 else '#111827'
            ax_a.text(x_val, y_val + 0.18, f"{failure_rate:.1f}%", ha='center', va='center', color=text_color, fontsize=11.0, zorder=3)
            ax_a.text(x_val, y_val + 0.01, f"{failed_count}/{total_count}", ha='center', va='center', color=text_color, fontsize=9.5, zorder=3)
            ax_a.text(x_val, y_val - 0.18, f"CI {ci_lower:.1f}-{ci_upper:.1f}", ha='center', va='center', color=text_color, fontsize=8.2, zorder=3)

    ax_a.set_xlim(x_positions[0] - 0.60, x_positions[-1] + 0.60)
    ax_a.set_ylim(y_positions[-1] - 0.62, y_positions[0] + 0.62)
    ax_a.set_aspect('equal', adjustable='box')
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(display_x_order, rotation=0, ha='center', fontweight='normal', fontsize=12.2)
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels(mas_order, fontweight='normal', fontsize=13.2)
    ax_a.tick_params(axis='x', length=0, pad=10)
    ax_a.tick_params(axis='y', length=0)
    for spine_name in ['top', 'right', 'left', 'bottom']:
        ax_a.spines[spine_name].set_visible(False)

    na_patch = mpatches.Patch(
        facecolor='#F8FAFC',
        edgecolor='#94A3B8',
        linestyle='--',
        hatch='////',
        label='Step Not Present',
    )
    ax_a.legend(
        handles=[na_patch],
        loc='upper left',
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor='white',
        edgecolor='#CBD5E1',
        borderpad=0.45,
        fontsize=13,
    )

    sm_a = plt.cm.ScalarMappable(cmap=panel_a_cmap, norm=panel_a_norm)
    sm_a.set_array([])
    cbar_a = plt.colorbar(sm_a, ax=ax_a, pad=0.015, shrink=0.78, aspect=18)
    cbar_a.set_label('Failure Rate (%)', rotation=270, labelpad=16, fontsize=14, fontweight='normal')
    cbar_a.set_ticks(np.arange(0, panel_a_top + 0.001, panel_a_tick))
    cbar_a.ax.tick_params(labelsize=12.5)
    cbar_a.outline.set_visible(False)

    ax_b = fig.add_subplot(gs[0, 1])
    plot_dataset_divergence_panel(ax_b, panel_b_stats)

    ax_c = fig.add_subplot(gs[1, 0])
    plot_kinematic_panel(ax_c, panel_c_stats, panel_c_models, panel_c_color_map, "c", display_names=model_display_names)

    ax_d = fig.add_subplot(gs[1, 1])
    plot_kinematic_panel(ax_d, panel_d_stats, panel_d_models, panel_d_color_map, "d", display_names=model_display_names)

    plt.subplots_adjust(left=0.065, right=0.925, top=0.96, bottom=0.08)
    fig.canvas.draw()

    # Keep the Panel b typography unchanged. For the dense Phase III synthesis
    # modes, the user requested a narrower but taller plotting region relative
    # to the current panel-b geometry; other modes keep the previous 90% x 90%
    # reduction.
    ax_b_pos = ax_b.get_position()
    panel_b_width_scale = 0.81 if code in {"3.1.1", "3.1.2", "3.1.3"} else 0.90
    panel_b_height_scale = 0.99 if code in {"3.1.1", "3.1.2", "3.1.3"} else 0.90
    ax_b_new_width = ax_b_pos.width * panel_b_width_scale
    ax_b_new_height = ax_b_pos.height * panel_b_height_scale
    ax_b.set_position([ax_b_pos.x0, ax_b_pos.y1 - ax_b_new_height, ax_b_new_width, ax_b_new_height])

    # Keep the typography unchanged for Panels c/d, but reduce the plotting
    # region itself to 85% so the axes occupy less visual weight.
    for axis in (ax_c, ax_d):
        axis_pos = axis.get_position()
        new_width = axis_pos.width * 0.85
        new_height = axis_pos.height * 0.85
        axis.set_position([axis_pos.x0, axis_pos.y1 - new_height, new_width, new_height])

    fig.canvas.draw()

    ax_d_pos = ax_d.get_position()
    cax_delta = fig.add_axes([ax_d_pos.x1 + 0.012, ax_d_pos.y0 + ax_d_pos.height * 0.10, 0.012, ax_d_pos.height * 0.78])
    sm_delta = plt.cm.ScalarMappable(cmap=delta_cmap, norm=delta_norm)
    sm_delta.set_array([])
    cbar_delta = plt.colorbar(sm_delta, cax=cax_delta)
    cbar_delta.set_label(
        'Arrow Color = Step-to-Step Change (Delta %)',
        rotation=270,
        labelpad=14,
        fontweight='normal',
        fontsize=15,
    )
    cbar_delta.ax.tick_params(labelsize=12.5)
    cbar_delta.outline.set_visible(False)

    main_pdf_path, main_png_path = save_figure_variants(fig, output_dir, f"failure_mode_{code}")
    plt.close(fig)
    print(f"Saved enhanced main figure: {main_pdf_path}")
    print(f"Saved enhanced main figure PNG: {main_png_path}")


def plot_failure_mode_comprehensive(code, df_mode, output_dir):
    """
    Generate the final comprehensive figure.
    - Panel A uses the V3 MAS cognitive pipeline style.
    - Panel B uses the V4 dataset modality divergence style.
    - Panels C and D use the V4 kinematic degradation style.

    Compared with `plot_failure_mode_111_enhanced`, this path favors reuse and
    consistency over bespoke page-level tuning.

    This should remain the default path for most failure modes. Reuse it when
    the figure can keep the shared visual grammar and only needs changes in
    labels, ordering, colors, or axis ranges. Promote a mode to its own
    renderer only after the shared layout becomes a real constraint.
    """
    config = AUDIT_CONFIG[code]
    mode_name = config["name"]

    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found.")
        return

    df_mode = apply_failure_mode_step_filter(code, df_mode)
    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found after step filtering.")
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
    panel_title_fontsize = 20
    panel_title_pad = 16
    panel_title_y = 1.03
    panel_title_linespacing = 1.12
    panel_b_xtick_pad = 18

    plt.rcParams.update(
        {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica'],
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none',
            'text.usetex': False,
            'axes.linewidth': 1.2,
            'axes.labelsize': common_axis_label_fontsize,
            'axes.titlesize': panel_title_fontsize,
            'xtick.labelsize': common_tick_fontsize,
            'ytick.labelsize': common_tick_fontsize,
            'legend.fontsize': common_tick_fontsize,
            'legend.title_fontsize': common_tick_fontsize,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )

    fig = plt.figure(figsize=(24, 18.4))
    fig.patch.set_facecolor('white')

    gs = GridSpec(2, 2, figure=fig, wspace=0.24, hspace=0.22, height_ratios=[1.10, 1.20])

    qa_datasets = ["MedQA", "PubMedQA", "MedXpertQA"]
    vqa_datasets = ["PathVQA", "VQA-RAD", "SLAKE"]

    mas_priority = ["ColaCare", "HealthcareAgent", "MAC", "MDAgents", "MedAgents", "ReConcile"]
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
        label='Step Not Present',
    )
    ax_a.legend(
        handles=[na_patch],
        loc='lower left',
        bbox_to_anchor=(0.0, 0.985),
        borderaxespad=0,
        frameon=False,
            fontsize=16,
    )
    ax_a.set_title(
        "A. Failure rates across multi-agent architectures and collaboration stages.",
        fontweight='bold',
        fontsize=panel_title_fontsize,
        loc='left',
        pad=panel_title_pad,
        y=panel_title_y,
        linespacing=panel_title_linespacing,
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
        "B. Cumulative failure rates stratified by text and visual medical datasets.",
        fontweight='bold',
        fontsize=panel_title_fontsize,
        loc='left',
        pad=panel_title_pad,
        y=panel_title_y,
        linespacing=panel_title_linespacing,
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
            "DeepSeek-V3.2-Thinking": "DeepSeek-V3.2",
            "GPT-5.2": "GPT-5.2",
            "Gemini-3-Flash-Preview": "Gemini-3-Flash",
            "Qwen3-8B": "Qwen-3",
            "GLM-4.6V": "GLM-4.6V",
            "Qwen3-VL-8B-Thinking": "Qwen-3VL",
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

        ax.set_title(
            title,
            fontweight='bold',
            fontsize=panel_title_fontsize,
            loc='left',
            pad=panel_title_pad,
            y=panel_title_y,
            linespacing=panel_title_linespacing,
        )
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
    ax_c, _ = plot_kinematic_panel(
        gs[1, 0],
        df_c,
        "C. Absolute failure rates and stage-to-stage changes\nfor large language models in text QA tasks.",
        "Set1",
        seed_offset=10000,
    )

    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    ax_d, _ = plot_kinematic_panel(
        gs[1, 1],
        df_d,
        "D. Absolute failure rates and stage-to-stage changes\nfor vision-language models in medical VQA tasks.",
        "Dark2",
        seed_offset=15000,
    )

    plt.subplots_adjust(left=0.065, right=0.905, top=0.965, bottom=0.055)
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
        'Stage-to-Stage Change (Delta %)',
        rotation=270,
        labelpad=20,
        fontweight='bold',
        fontsize=common_colorbar_label_fontsize,
    )
    cbar_delta.ax.tick_params(labelsize=common_colorbar_tick_fontsize)
    cbar_delta.outline.set_visible(False)

    filename = f"failure_mode_{code}.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Saved Nature-quality comprehensive figure: {save_path}")


def main():
    """
    Entry point used for paper figure reproduction.

    Examples:
        `uv run python scripts/audit_drawing.py 1.1.1`
        `uv run python scripts/audit_drawing.py 1.2.1 2.2.2`
        `uv run python scripts/audit_drawing.py`
    """
    input_base_dir = project_root / "logs" / "audit_results" / "20260302"
    output_dir = project_root / "logs" / "audit_results" / "figures_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {output_dir}")

    df_global = process_audit_data_to_df(input_base_dir)
    requested_codes = sys.argv[1:] if len(sys.argv) > 1 else list(AUDIT_CONFIG.keys())

    invalid_codes = [code for code in requested_codes if code not in AUDIT_CONFIG]
    if invalid_codes:
        raise ValueError(f"Unknown failure mode code(s): {', '.join(invalid_codes)}")

    print("\nGenerating Plots...")
    for code in requested_codes:
        df_mode = df_global[df_global['failure_mode'] == code]
        # Route special paper-tuned modes explicitly here. For a future bespoke
        # renderer, add another branch like `elif code == "x.x.x": ...`.
        if code in {"1.1.1", "1.2.1"}:
            plot_failure_mode_111_enhanced(code, df_mode, output_dir)
        elif code in {"2.1.2", "2.2.1", "2.2.2", "3.1.1", "3.1.2", "3.1.3"}:
            plot_failure_mode_212_enhanced(code, df_mode, output_dir)
        else:
            plot_failure_mode_comprehensive(code, df_mode, output_dir)

    print("\nVisualization Audit Completed.")


if __name__ == "__main__":
    main()
