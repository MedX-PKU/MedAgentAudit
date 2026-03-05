'''
./scripts/audit_drawing.py
This script visualizes the failure rates for each failure mode across different rounds and stages, 
based on the audit results from multi-agent systems. 
It answers 3 core questions regarding the evolution of failure modes:
1. MAS frameworks evolution (Heatmap)
2. Medical scenarios/datasets evolution (Line plot with 95% CI)
3. Base LLMs evolution for Medical QA & VQA (Line plot with 95% CI)
'''
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import matplotlib.patheffects as pe
import scipy.interpolate as interp
import matplotlib.colors as mcolors

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))

AUDIT_CONFIG = {
    "1.1.1": {"status_key": "factual_hallucination_status", "name": "Factual Hallucination", "valid_rounds":[1, 2, 3]},
    "1.2.1": {"status_key": "modality_neglect_status", "name": "Modality Neglect", "valid_rounds":[1, 2, 3]},
    "2.1.1": {"status_key": "role_task_alignment", "name": "Role Assignment Mismatch", "valid_rounds": [1]},
    "2.1.2": {"status_key": "knowledge_activation_status", "name": "Knowledge Activation Fail", "valid_rounds": [1, 2, 3]},
    "2.2.1": {"status_key": "interaction_redundancy", "name": "Interaction Redundancy", "valid_rounds": [1, 2, 3]},
    "2.2.2": {"status_key": "conflict_resolution_status", "name": "Unresolved Conflicts", "valid_rounds":[1, 2, 3]},
    "3.1.1": {"status_key": "suppression_status", "name": "Minority Suppression", "valid_rounds": [1, 2, 3]},
    "3.1.2": {"status_key": "authority_bias_status", "name": "Authority Bias", "valid_rounds": [1, 2, 3]},
    "3.1.3": {"status_key": "neglect_of_conflict_status", "name": "Neglect of Contradictions", "valid_rounds": [1, 2, 3]},
    "3.2.1": {"status_key": "inter_round_consistency_status", "name": "Inter-round Inconsistency", "valid_rounds": [2, 3]}
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
    "3.2.1": "3_2_1_self_contradiction_when_decision"
}

STAGE_ORDER = ["role_assignment", "analysis", "synthesis", "review", "decision"]

STAGE_DISPLAY = {
    "role_assignment": "Assign",
    "analysis": "Analysis",
    "synthesis": "Synthesis",
    "review": "Review",
    "decision": "Decision"
}

def process_audit_data_to_df(audit_results_path: Path):
    """
    Parse JSONL logs and construct a structured Pandas DataFrame.
    """
    records =[]
    jsonl_files = list(audit_results_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} files. Processing into DataFrame...")

    MAS_MAP = {
        "colacare": "ColaCare",
        "healthcareagent": "HealthcareAgent",
        "mac": "MAC",
        "mdagents": "MDAgents",
        "medagent": "MedAgent",
        "reconcile": "ReConcile"
    }
    
    LLM_MAP = {
        "deepseek-reasoner": "DeepSeek-V3.2-Thinking",
        "gpt-5.2": "GPT-5.2",
        "gemini-3-flash-preview": "Gemini-3-Flash-Preview",
        "qwen3-8b": "Qwen3-8B",
        "glm-4.6v": "GLM-4.6V",
        "qwen3-vl-8b-thinking": "Qwen3-VL-8B-Thinking"
    }

    for jsonl_file in tqdm(jsonl_files, desc="Scanning Logs"):
        if "errors" in jsonl_file.name:
            continue
            
        stem = jsonl_file.stem
        parts = stem.split('_')
        if len(parts) < 3:
            continue
            
        raw_mas = parts[0]
        mas = MAS_MAP.get(raw_mas.lower(), raw_mas)
        
        dataset = parts[1]
        if dataset.lower() == "medxpertqa-text":
            dataset = "MedXpertQA"
            
        raw_llm = "_".join(parts[2:])
        llm = LLM_MAP.get(raw_llm.lower(), raw_llm)
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): 
                        continue 
                    case_data = json.loads(line)
                    
                    audit_data = None
                    if "case_history" in case_data and "audit" in case_data["case_history"]:
                        audit_data = case_data["case_history"]["audit"]
                    
                    if not audit_data: 
                        continue

                    rounds = audit_data.get("rounds",[])
                    
                    for r_data in rounds:
                        round_num = r_data.get("round")
                        if not round_num: 
                            continue

                        for code, config in AUDIT_CONFIG.items():
                            if round_num not in config["valid_rounds"]:
                                continue

                            log_key = LOG_KEY_MAP[code]
                            entries = r_data.get(log_key,[])
                            
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
                                
                                records.append({
                                    "failure_mode": code,
                                    "mas": mas,
                                    "dataset": dataset,
                                    "llm": llm,
                                    "round_num": round_num,
                                    "step": step,
                                    "round_stage": round_stage,
                                    "failed": is_failed
                                })
        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")
            continue
            
    df = pd.DataFrame(records)
    return df

def plot_failure_mode_comprehensive_v1(code, df_mode, output_dir):
    """
    Generate a highly polished 4-panel multi-plot figure for Nature Medicine standards.
    Panel A: MAS Evolution (Heatmap)
    Panel B: Dataset Evolution (Ridgeline/Joyplot)
    Panel C: Base Models QA (Radar Chart)
    Panel D: Vision Models VQA (Radar Chart)
    """
    config = AUDIT_CONFIG[code]
    mode_name = config["name"]
    
    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found.")
        return

    # Dynamically determine valid X axis order
    unique_round_stages = df_mode[['round_num', 'step', 'round_stage']].drop_duplicates()
    unique_round_stages['step_idx'] = unique_round_stages['step'].map({s: i for i, s in enumerate(STAGE_ORDER)})
    unique_round_stages = unique_round_stages.sort_values(['round_num', 'step_idx'])
    x_order = unique_round_stages['round_stage'].tolist()
    
    if not x_order:
        return

    df_mode = df_mode.copy()
    df_mode['round_stage'] = pd.Categorical(df_mode['round_stage'], categories=x_order, ordered=True)

    # Nature style typography and aesthetics
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 14
    })

    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor('white')
    plt.suptitle(f"Failure Mode {code}: {mode_name}\nEvolution across Collaboration Stages", 
                 fontsize=28, fontweight='bold', y=0.98)
    
    # Use GridSpec for advanced layout control (allowing Polar plots for C & D)
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.35)

    # =========================================================
    # Panel A: Temporal Heatmap (System Blueprint)
    # =========================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    df_a = df_mode.groupby(['mas', 'round_stage'], observed=False)['failed'].agg(['mean', 'count']).reset_index()
    df_a['failure_rate'] = df_a['mean'] * 100
    pivot_a = df_a.pivot(index='mas', columns='round_stage', values='failure_rate')
    
    mas_order = ["ColaCare", "HealthcareAgent", "MAC", "MDAgents", "MedAgent", "ReConcile"]
    mas_order = [m for m in mas_order if m in df_mode['mas'].unique()]
    pivot_a = pivot_a.reindex(index=mas_order, columns=x_order)
    
    # Nature style continuous colormap
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    
    sns.heatmap(pivot_a, cmap=cmap, annot=True, fmt=".1f", vmin=0, vmax=100,
                cbar_kws={'label': 'Failure Rate (%)', 'shrink': 0.8}, ax=ax_a, 
                linewidths=2, linecolor='white', annot_kws={"size": 11, "weight": "bold"})
    
    # Handle structurally N/A missing elements
    for i in range(pivot_a.shape[0]):
        for j in range(pivot_a.shape[1]):
            if np.isnan(pivot_a.iloc[i, j]):
                ax_a.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor='#f0f0f0', 
                                             hatch='////', edgecolor='white', lw=2))
                
    na_patch = mpatches.Patch(facecolor='#f0f0f0', hatch='////', edgecolor='white', label='Structurally N/A')
    ax_a.legend(handles=[na_patch], loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    
    ax_a.set_title("A. MAS Framework Evolution", fontweight='bold', loc='left', pad=15)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.tick_params(axis='x', rotation=30)
    ax_a.tick_params(axis='y', rotation=0)

    # =========================================================
    # Panel B: Dataset Evolution (Ridgeline Plot / Joyplot)
    # =========================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    df_b = df_mode.groupby(['dataset', 'round_stage'], observed=False)['failed'].mean().reset_index()
    df_b['failure_rate'] = df_b['failed'] * 100
    
    datasets = ["MedQA", "PubMedQA", "MedXpertQA", "PathVQA", "VQA-RAD", "SLAKE"]
    datasets = [d for d in datasets if d in df_b['dataset'].unique()]
    
    # Custom palette: Blues for QA, Oranges for VQA
    color_map = {
        "MedQA": "#1f77b4", "PubMedQA": "#4292c6", "MedXpertQA": "#9ecae1",
        "PathVQA": "#d94801", "VQA-RAD": "#fd8d3c", "SLAKE": "#fdd0a2"
    }

    # Parameters for Ridgeline
    overlap = 1.5
    x_numeric = np.arange(len(x_order))
    
    for i, dataset in enumerate(reversed(datasets)): # Reverse to plot bottom-up
        ds_data = df_b[df_b['dataset'] == dataset].set_index('round_stage').reindex(x_order)['failure_rate'].fillna(0).values
        
        # Base offset for this dataset
        base_y = i * (100 / overlap) 
        
        # Normalize the heights slightly to fit the ridgeline aesthetic
        y = base_y + (ds_data * 0.8) 
        
        color = color_map.get(dataset, "#333333")
        
        # Fill area and draw line
        ax_b.fill_between(x_numeric, base_y, y, color=color, alpha=0.8, zorder=len(datasets)-i)
        ax_b.plot(x_numeric, y, color='white', linewidth=2, zorder=len(datasets)-i)
        
        # Add Dataset Label
        ax_b.text(-0.2, base_y + 10, dataset, fontweight='bold', color=color, ha='right', va='center', fontsize=13)

    ax_b.set_title("B. Medical Scenario Degradation (Ridgeline)", fontweight='bold', loc='left', pad=15)
    ax_b.set_xticks(x_numeric)
    ax_b.set_xticklabels(x_order, rotation=30, ha='right')
    ax_b.set_yticks([]) # Hide Y axis ticks, ridgeline labels replace them
    ax_b.spines['left'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.spines['top'].set_visible(False)
    ax_b.grid(axis='x', linestyle='--', alpha=0.5)

    # =========================================================
    # Helper Function for Radar Charts (Panel C & D)
    # =========================================================
    def plot_radar(ax, df_subset, title, palette):
        if df_subset.empty:
            ax.text(0.5, 0.5, "No Data Available", ha='center', va='center', fontsize=16)
            ax.axis('off')
            return

        df_agg = df_subset.groupby(['llm', 'round_stage'], observed=False)['failed'].mean().reset_index()
        df_agg['failure_rate'] = df_agg['failed'] * 100
        
        models = df_agg['llm'].unique()
        angles = np.linspace(0, 2 * np.pi, len(x_order), endpoint=False).tolist()
        angles += angles[:1] # Close the loop
        
        ax.set_theta_offset(np.pi / 2) # Start at top
        ax.set_theta_direction(-1) # Clockwise
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(x_order, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], color="grey", size=10)
        ax.grid(color='#E8E8E8', linestyle='-', linewidth=1.5)
        ax.spines['polar'].set_visible(False) # Clean Nature look

        colors = sns.color_palette(palette, n_colors=len(models))
        
        for idx, model in enumerate(models):
            values = df_agg[df_agg['llm'] == model].set_index('round_stage').reindex(x_order)['failure_rate'].fillna(0).tolist()
            values += values[:1] # Close the loop
            
            ax.plot(angles, values, color=colors[idx], linewidth=2.5, linestyle='solid', label=model)
            ax.fill(angles, values, color=colors[idx], alpha=0.15)
            # Add markers
            ax.scatter(angles, values, color=colors[idx], s=50, zorder=10)

        ax.set_title(title, fontweight='bold', loc='left', pad=30, fontsize=18)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)

    # =========================================================
    # Panel C: Base Model Evolution (Medical QA) - Radar Chart
    # =========================================================
    ax_c = fig.add_subplot(gs[1, 0], polar=True)
    qa_datasets = ["MedQA", "PubMedQA", "MedXpertQA"]
    df_c = df_mode[df_mode['dataset'].isin(qa_datasets)].copy()
    plot_radar(ax_c, df_c, "C. Cognitive Cycle: Base Models (Text QA)", "Set1")

    # =========================================================
    # Panel D: Vision Model Evolution (Medical VQA) - Radar Chart
    # =========================================================
    ax_d = fig.add_subplot(gs[1, 1], polar=True)
    vqa_datasets = ["PathVQA", "VQA-RAD", "SLAKE"]
    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    plot_radar(ax_d, df_d, "D. Cognitive Cycle: Vision Models (VQA)", "Dark2")

    # Final layout adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f"Failure_Mode_{code}_Nature_Evolution.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Saved Nature-quality comprehensive figure: {save_path}")

def plot_failure_mode_comprehensive_v2(code, df_mode, output_dir):
    """
    Generate a V2 highly polished 4-panel figure for Nature Medicine standards.
    Features: Bubble Heatmap, Small Multiples Area Plots, and Temporal Dumbbell Plots.
    """
    config = AUDIT_CONFIG[code]
    mode_name = config["name"]
    
    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found.")
        return

    # Dynamically determine valid X axis order
    unique_round_stages = df_mode[['round_num', 'step', 'round_stage']].drop_duplicates()
    unique_round_stages['step_idx'] = unique_round_stages['step'].map({s: i for i, s in enumerate(STAGE_ORDER)})
    unique_round_stages = unique_round_stages.sort_values(['round_num', 'step_idx'])
    x_order = unique_round_stages['round_stage'].tolist()
    
    if not x_order:
        return

    df_mode = df_mode.copy()
    df_mode['round_stage'] = pd.Categorical(df_mode['round_stage'], categories=x_order, ordered=True)

    # Nature style typography and aesthetics (Minimalist, clean, high data-ink ratio)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.linewidth': 1.0,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor('white')
    plt.suptitle(f"Failure Mode {code}: {mode_name}\nEvolution across Collaboration Stages", 
                 fontsize=24, fontweight='bold', y=0.98, ha='center')
    
    # Advanced GridSpec layout
    gs = GridSpec(3, 2, figure=fig, wspace=0.25, hspace=0.45)

    # =========================================================
    # Panel A: MAS Framework Evolution (Bubble Heatmap / Dot Plot)
    # =========================================================
    ax_a = fig.add_subplot(gs[0, :]) # Spans both columns
    
    df_a = df_mode.groupby(['mas', 'round_stage'], observed=False)['failed'].mean().reset_index()
    df_a['failure_rate'] = df_a['failed'] * 100
    
    mas_order = ["ColaCare", "HealthcareAgent", "MAC", "MDAgents", "MedAgent", "ReConcile"]
    mas_order = [m for m in mas_order if m in df_mode['mas'].unique()]
    
    # Create coordinate mapping
    x_map = {stage: i for i, stage in enumerate(x_order)}
    y_map = {mas: i for i, mas in enumerate(reversed(mas_order))}
    
    df_a['x'] = df_a['round_stage'].map(x_map)
    df_a['y'] = df_a['mas'].map(y_map)
    
    # Draw structurally N/A background grid
    for x_val in x_map.values():
        for y_val in y_map.values():
            ax_a.add_patch(mpatches.Rectangle((x_val-0.4, y_val-0.4), 0.8, 0.8, 
                                            fill=False, hatch='//////', color='#e0e0e0', lw=0, zorder=0))

    # Scatter plot for Bubble Heatmap
    scatter = ax_a.scatter(
        df_a['x'], df_a['y'], 
        s=df_a['failure_rate'] * 15 + 50, # Scale size
        c=df_a['failure_rate'], 
        cmap='magma_r', # Nature favors perceptually uniform colormaps like magma or viridis
        vmin=0, vmax=100, 
        edgecolors='black', linewidths=0.5, alpha=0.9, zorder=2
    )

    ax_a.set_xticks(list(x_map.values()))
    ax_a.set_xticklabels(list(x_map.keys()), rotation=0, fontweight='bold')
    ax_a.set_yticks(list(y_map.values()))
    ax_a.set_yticklabels(list(y_map.keys()), fontweight='bold')
    
    # Colorbar and Legends
    cbar = plt.colorbar(scatter, ax=ax_a, pad=0.02, shrink=0.7, aspect=15)
    cbar.set_label('Failure Rate (%)', rotation=270, labelpad=20, fontweight='bold')
    
    # Size legend
    kw = dict(prop="sizes", num=4, color="gray", fmt="{x:.0f}%", func=lambda s: (s-50)/15)
    ax_a.legend(*scatter.legend_elements(**kw), title="Rate (Size)", bbox_to_anchor=(1.15, 0.5), 
                loc="center left", frameon=False)
                
    ax_a.set_title("A. MAS Framework Temporal Vulnerability (Bubble Heatmap)", fontweight='bold', loc='left', pad=15)
    ax_a.spines['left'].set_visible(False)
    ax_a.spines['bottom'].set_visible(False)
    ax_a.tick_params(axis='both', length=0) # Hide ticks, keep labels
    ax_a.grid(True, linestyle='--', color='gray', alpha=0.2, zorder=1)

    # =========================================================
    # Panel B: Dataset Evolution (Small Multiples Area Plot)
    # =========================================================
    # Sub-GridSpec for Small Multiples
    gs_b = gs[1, :].subgridspec(2, 3, wspace=0.15, hspace=0.4)
    
    qa_datasets = ["MedQA", "PubMedQA", "MedXpertQA"]
    vqa_datasets = ["PathVQA", "VQA-RAD", "SLAKE"]
    all_datasets = qa_datasets + vqa_datasets
    
    color_map = {
        "MedQA": "#08519c", "PubMedQA": "#3182bd", "MedXpertQA": "#6baed6", # Blues
        "PathVQA": "#a63603", "VQA-RAD": "#e6550d", "SLAKE": "#fd8d3c"       # Oranges
    }

    # First add a dummy axis just to hold the Panel Title
    ax_b_title = fig.add_subplot(gs[1, :])
    ax_b_title.axis('off')
    ax_b_title.set_title("B. Medical Scenario Degradation (Small Multiples with 95% CI)", fontweight='bold', loc='left', pad=25)

    max_rate = df_mode.groupby(['dataset', 'round_stage'], observed=False)['failed'].mean().max() * 100
    y_limit = min(100, max_rate * 1.2) if not pd.isna(max_rate) else 100

    x_numeric = np.arange(len(x_order))
    
    for idx, ds in enumerate(all_datasets):
        if ds not in df_mode['dataset'].unique(): continue
        
        row, col = divmod(idx, 3)
        ax_small = fig.add_subplot(gs_b[row, col])
        
        df_ds = df_mode[df_mode['dataset'] == ds]
        
        # Use Seaborn's lineplot to automatically calculate bootstrapped 95% CI
        df_ds_plot = df_ds.copy()
        df_ds_plot['x_num'] = df_ds_plot['round_stage'].map(x_map)
        df_ds_plot['failure_rate'] = df_ds_plot['failed'] * 100
        
        sns.lineplot(data=df_ds_plot, x='x_num', y='failure_rate', 
                     errorbar=('ci', 95), color=color_map[ds], ax=ax_small, lw=2)
        
        # Fill area under the mean curve for aesthetic impact
        mean_data = df_ds_plot.groupby('x_num')['failure_rate'].mean()
        ax_small.fill_between(mean_data.index, mean_data.values, 0, color=color_map[ds], alpha=0.15)
        
        ax_small.set_title(ds, fontweight='bold', color=color_map[ds], fontsize=12)
        ax_small.set_ylim(0, y_limit)
        ax_small.set_xlim(0, len(x_order)-1)
        ax_small.set_xlabel("")
        ax_small.set_ylabel("")
        
        ax_small.set_xticks(x_numeric)
        if row == 1:
            ax_small.set_xticklabels(x_order, rotation=45, ha='right', fontsize=9)
        else:
            ax_small.set_xticklabels([])
            
        if col != 0:
            ax_small.set_yticklabels([])
        else:
            ax_small.set_ylabel("Failure (%)", fontsize=10)
            
        ax_small.grid(axis='y', linestyle=':', alpha=0.6)

    # =========================================================
    # Helper Function: Temporal Dumbbell Plots (Panel C & D)
    # =========================================================
    def plot_temporal_dumbbell(ax, df_subset, title, palette_name):
        ax.set_title(title, fontweight='bold', loc='left', pad=15)
        
        if df_subset.empty:
            ax.text(0.5, 0.5, "No Data Available", ha='center', va='center')
            ax.axis('off')
            return

        df_agg = df_subset.groupby(['llm', 'round_stage'], observed=False)['failed'].mean().reset_index()
        df_agg['failure_rate'] = df_agg['failed'] * 100
        
        models = sorted(df_agg['llm'].unique())
        colors = sns.color_palette(palette_name, n_colors=len(models))
        color_dict = dict(zip(models, colors))
        
        y_positions = np.arange(len(x_order))
        
        for i, stage in enumerate(x_order):
            stage_data = df_agg[df_agg['round_stage'] == stage]
            if stage_data.empty or stage_data['failure_rate'].isna().all():
                continue
                
            min_val = stage_data['failure_rate'].min()
            max_val = stage_data['failure_rate'].max()
            
            # Draw connecting dumbbell line (Variance indicator)
            ax.plot([min_val, max_val], [i, i], color='#d3d3d3', lw=4, zorder=1)
            
            # Draw individual dots
            for _, row in stage_data.iterrows():
                ax.scatter(row['failure_rate'], i, color=color_dict[row['llm']], 
                           s=120, edgecolors='white', lw=1.5, zorder=2, label=row['llm'] if i==0 else "")
                           
        # Formatting Time flowing downwards
        ax.set_yticks(y_positions)
        ax.set_yticklabels(x_order, fontweight='bold')
        ax.invert_yaxis() # Important: Time flows from top to bottom
        
        ax.set_xlabel("Failure Rate (%)", fontweight='bold')
        ax.set_xlim(-5, 105)
        ax.xaxis.grid(True, linestyle='--', alpha=0.5)
        
        # Unique legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))

    # =========================================================
    # Panel C: Base Model Evolution QA (Temporal Dumbbell)
    # =========================================================
    ax_c = fig.add_subplot(gs[2, 0])
    df_c = df_mode[df_mode['dataset'].isin(qa_datasets)].copy()
    plot_temporal_dumbbell(ax_c, df_c, "C. Model Performance Spread: Text QA", "Set1")

    # =========================================================
    # Panel D: Vision Model Evolution VQA (Temporal Dumbbell)
    # =========================================================
    ax_d = fig.add_subplot(gs[2, 1])
    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    plot_temporal_dumbbell(ax_d, df_d, "D. Model Performance Spread: Vision VQA", "Dark2")

    # Final layout adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f"Failure_Mode_{code}_Nature_Evolution_V2.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Saved V2 Nature-quality comprehensive figure: {save_path}")




def plot_failure_mode_comprehensive_v3(code, df_mode, output_dir):
    """
    Generate a V3 highly polished 4-panel figure for Nature Medicine standards.
    Features: 
    - Panel A: Radial/Circular Heatmap (Circos-style)
    - Panel B: Smoothed Bump Chart (Rank + Bubble Size)
    - Panel C & D: Clinical Step Plots (Kaplan-Meier Style Cumulative Failure)
    """
    config = AUDIT_CONFIG[code]
    mode_name = config["name"]
    
    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found.")
        return

    # Dynamically determine valid X axis order
    unique_round_stages = df_mode[['round_num', 'step', 'round_stage']].drop_duplicates()
    unique_round_stages['step_idx'] = unique_round_stages['step'].map({s: i for i, s in enumerate(STAGE_ORDER)})
    unique_round_stages = unique_round_stages.sort_values(['round_num', 'step_idx'])
    x_order = unique_round_stages['round_stage'].tolist()
    
    if not x_order:
        return

    df_mode = df_mode.copy()
    df_mode['round_stage'] = pd.Categorical(df_mode['round_stage'], categories=x_order, ordered=True)

    # Nature Medicine Aesthetics
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 12,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('white')
    plt.suptitle(f"Failure Mode {code}: {mode_name}\nDynamic Evolution of Cognitive Vulnerabilities", 
                 fontsize=26, fontweight='bold', y=0.98, ha='center')
    
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.35)

    # =========================================================
    # Panel A: Radial Heatmap (Circos-style Cognitive Cycle)
    # =========================================================
    ax_a = fig.add_subplot(gs[0, 0], polar=True)
    
    df_a = df_mode.groupby(['mas', 'round_stage'], observed=False)['failed'].mean().reset_index()
    df_a['failure_rate'] = df_a['failed'] * 100
    pivot_a = df_a.pivot(index='mas', columns='round_stage', values='failure_rate')
    
    mas_order = ["ColaCare", "HealthcareAgent", "MAC", "MDAgents", "MedAgent", "ReConcile"]
    mas_order = [m for m in mas_order if m in df_mode['mas'].unique()]
    pivot_a = pivot_a.reindex(index=mas_order, columns=x_order)
    
    # Polar coordinates setup
    num_stages = len(x_order)
    num_mas = len(mas_order)
    theta = np.linspace(0.0, 2 * np.pi, num_stages + 1)
    r = np.arange(1, num_mas + 2) # Start from 1 to leave a hole in the middle (Donut style)
    
    Theta, R = np.meshgrid(theta, r)
    
    # Custom Colormap (Nature Medicine loves deep Teal to Crimson for risk)
    cmap = sns.color_palette("mako_r", as_cmap=True)
    cmap.set_bad(color='#f0f0f0') # NaN color
    
    # Plotting the pcolormesh
    values = pivot_a.values
    mesh = ax_a.pcolormesh(Theta, R, values, cmap=cmap, vmin=0, vmax=100, edgecolor='white', linewidth=1.5)
    
    # Add hatching for NaNs (Structurally N/A)
    for i in range(num_mas):
        for j in range(num_stages):
            if np.isnan(values[i, j]):
                # Create a patch for missing data
                theta_slice = np.linspace(theta[j], theta[j+1], 50)
                ax_a.fill_between(theta_slice, r[i], r[i+1], facecolor='#f0f0f0', hatch='////', edgecolor='white', lw=1)

    # Styling the Polar plot
    ax_a.set_theta_offset(np.pi / 2)
    ax_a.set_theta_direction(-1) # Clockwise
    
    # X-ticks (Stages)
    ax_a.set_xticks(theta[:-1] + np.pi/num_stages)
    ax_a.set_xticklabels(x_order, fontweight='bold', fontsize=11)
    
    # Y-ticks (MAS labels)
    ax_a.set_yticks(r[:-1] + 0.5)
    ax_a.set_yticklabels(mas_order, fontweight='bold', fontsize=11)
    ax_a.set_rlabel_position(0)
    
    # Remove borders
    ax_a.spines['polar'].set_visible(False)
    ax_a.grid(color='white', lw=1.5)
    
    # Center text
    ax_a.text(0, 0, "Cognitive\nCycle", ha='center', va='center', fontweight='bold', fontsize=12, color='gray')
    
    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax_a, pad=0.1, shrink=0.7, aspect=15)
    cbar.set_label('Failure Rate (%)', rotation=270, labelpad=20, fontweight='bold')
    
    # Legend for N/A
    na_patch = mpatches.Patch(facecolor='#f0f0f0', hatch='////', edgecolor='white', label='Structurally N/A')
    ax_a.legend(handles=[na_patch], loc='lower right', bbox_to_anchor=(1.3, -0.1), frameon=False)
    
    ax_a.set_title("A. Framework Architecture (Radial Cycle)", fontweight='bold', loc='left', pad=25, x=-0.1)


# =========================================================
    # Panel B: Dataset Evolution (Smoothed Bump Chart)
    # =========================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    # 1. Aggregate and calculate failure rates
    df_b = df_mode.groupby(['dataset', 'round_stage'], observed=False)['failed'].mean().reset_index()
    df_b['failure_rate'] = df_b['failed'] * 100
    
    # 2. Calculate Ranks at each stage (Rank 1 is the highest failure rate)
    df_b['rank'] = df_b.groupby('round_stage')['failure_rate'].rank(method='first', ascending=False)
    
    qa_datasets = ["MedQA", "PubMedQA", "MedXpertQA"]
    vqa_datasets = ["PathVQA", "VQA-RAD", "SLAKE"]
    datasets = [d for d in (qa_datasets + vqa_datasets) if d in df_b['dataset'].unique()]
    
    color_map = {
        "MedQA": "#08519c", "PubMedQA": "#3182bd", "MedXpertQA": "#6baed6",
        "PathVQA": "#a63603", "VQA-RAD": "#e6550d", "SLAKE": "#fd8d3c"
    }

    x_numeric = np.arange(len(x_order))
    x_map = {stage: i for i, stage in enumerate(x_order)}
    df_b['x_num'] = df_b['round_stage'].map(x_map)
    
    for ds in datasets:
        # Extract specific dataset and drop ANY rows with NaN in rank or x_num
        ds_data = df_b[df_b['dataset'] == ds].dropna(subset=['x_num', 'rank', 'failure_rate']).copy()
        
        # PchipInterpolator requires strictly increasing x values
        ds_data = ds_data.sort_values('x_num').drop_duplicates(subset=['x_num'])
        
        if ds_data.empty: 
            continue
            
        x_vals = ds_data['x_num'].values
        y_ranks = ds_data['rank'].values
        sizes = ds_data['failure_rate'].values
        
        # Smooth interpolation for Bump Chart curves
        if len(x_vals) > 1:
            x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
            # Use PchipInterpolator to avoid severe overshooting common in cubic splines
            spline = interp.PchipInterpolator(x_vals, y_ranks)
            y_smooth = spline(x_smooth)
            
            # Plot smooth ribbon
            ax_b.plot(x_smooth, y_smooth, color=color_map[ds], linewidth=4, alpha=0.6, zorder=1)
            
        # Scatter actual data points (size mapped to failure rate)
        scatter = ax_b.scatter(x_vals, y_ranks, s=sizes*15 + 50, color=color_map[ds], 
                               edgecolor='white', lw=1.5, zorder=3)
        
        # Add Dataset Label at the beginning of the line
        if len(y_ranks) > 0:
            start_y = y_ranks[0]
            start_x = x_vals[0]
            ax_b.text(start_x - 0.2, start_y, ds, color=color_map[ds], fontweight='bold', ha='right', va='center')

    ax_b.set_xticks(x_numeric)
    ax_b.set_xticklabels(x_order, rotation=30, ha='right', fontweight='bold')
    
    # Invert Y axis so Rank 1 (Most difficult) is at the top
    ax_b.invert_yaxis() 
    
    # Dynamically set Y-ticks based on max rank possible (handling cases with missing datasets)
    max_rank_val = df_b['rank'].max()
    if not np.isnan(max_rank_val) and max_rank_val >= 1:
        ax_b.set_yticks(np.arange(1, int(max_rank_val) + 1))
        
    ax_b.set_ylabel("Difficulty Rank (1 = Highest Failure)", fontweight='bold')
    ax_b.set_title("B. Dataset Difficulty Ranking Evolution (Bump Chart)", fontweight='bold', loc='left', pad=15)
    ax_b.grid(axis='y', linestyle='--', alpha=0.3)
    ax_b.spines['left'].set_visible(False)
    ax_b.spines['bottom'].set_linewidth(1.5)
    
    # Legend for Bubble Size
    kw = dict(prop="sizes", num=4, color="gray", fmt="{x:.0f}%", func=lambda s: (s-50)/15)
    ax_b.legend(*scatter.legend_elements(**kw), title="Failure Rate", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    # =========================================================
    # Helper Function: Clinical Step Plot (Kaplan-Meier style)
    # =========================================================
    def plot_clinical_step(ax, df_subset, title, palette_name):
        if df_subset.empty:
            ax.text(0.5, 0.5, "No Data Available", ha='center', va='center')
            ax.axis('off')
            return

        df_agg = df_subset.groupby(['llm', 'round_stage'], observed=False)['failed'].mean().reset_index()
        df_agg['failure_rate'] = df_agg['failed'] * 100
        
        models = sorted(df_agg['llm'].unique())
        colors = sns.color_palette(palette_name, n_colors=len(models))
        
        x_numeric = np.arange(len(x_order))
        
        for idx, model in enumerate(models):
            model_data = df_agg[df_agg['llm'] == model].set_index('round_stage').reindex(x_order)['failure_rate'].fillna(0).values
            
            # Step plot (Clinical/Survival style)
            ax.step(x_numeric, model_data, where='post', color=colors[idx], linewidth=3, label=model, zorder=3)
            
            # Shaded area under the step plot
            ax.fill_between(x_numeric, model_data, step='post', color=colors[idx], alpha=0.1, zorder=1)
            
            # Highlight the "Events" (jump points)
            ax.scatter(x_numeric, model_data, color=colors[idx], s=60, edgecolor='white', lw=1.5, zorder=4)

        ax.set_title(title, fontweight='bold', loc='left', pad=15)
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(x_order, rotation=30, ha='right', fontweight='bold')
        ax.set_ylabel("Cumulative / Absolute Failure Rate (%)", fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='-', alpha=0.3)
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.98))

    # =========================================================
    # Panel C: LLM Evolution QA (Clinical Step Plot)
    # =========================================================
    ax_c = fig.add_subplot(gs[1, 0])
    df_c = df_mode[df_mode['dataset'].isin(qa_datasets)].copy()
    plot_clinical_step(ax_c, df_c, "C. Base Model Risk Accumulation: Text QA", "Set1")

    # =========================================================
    # Panel D: LLM Evolution VQA (Clinical Step Plot)
    # =========================================================
    ax_d = fig.add_subplot(gs[1, 1])
    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    plot_clinical_step(ax_d, df_d, "D. Vision Model Risk Accumulation: VQA", "Dark2")

    # Final layout adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f"Failure_Mode_{code}_Nature_Evolution_V3.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Saved V3 Nature-quality comprehensive figure: {save_path}")

def main():
    input_base_dir = project_root / "logs" / "audit_results" / "20260302"
    output_dir = project_root / "logs" / "audit_results" / "figures_style_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {output_dir}")

    # Build universal DataFrame with all structured info
    df_global = process_audit_data_to_df(input_base_dir)

    print("\nGenerating Plots...")
    for code in AUDIT_CONFIG.keys():
        df_mode = df_global[df_global['failure_mode'] == code]
        plot_failure_mode_comprehensive_v3(code, df_mode, output_dir)

    print("\nVisualization Audit Completed.")

if __name__ == "__main__":
    main()