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


def main():
    input_base_dir = project_root / "logs" / "audit_results" / "20260302"
    output_dir = project_root / "logs" / "audit_results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {output_dir}")

    # Build universal DataFrame with all structured info
    df_global = process_audit_data_to_df(input_base_dir)

    print("\nGenerating Plots...")
    for code in AUDIT_CONFIG.keys():
        df_mode = df_global[df_global['failure_mode'] == code]
        plot_failure_mode_comprehensive_v1(code, df_mode, output_dir)

    print("\nVisualization Audit Completed.")

if __name__ == "__main__":
    main()