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

def plot_failure_mode_comprehensive(code, df_mode, output_dir):
    """
    Generate a highly polished 4-panel 2x2 multi-plot figure for Nature Medicine standards.
    Panel A: MAS Evolution (Heatmap)
    Panel B: Dataset Evolution (Line Plot w/ CI)
    Panel C: Medical QA Base LLMs (Line Plot w/ CI)
    Panel D: Medical VQA Vision LLMs (Line Plot w/ CI)
    """
    config = AUDIT_CONFIG[code]
    mode_name = config["name"]
    
    if df_mode.empty:
        print(f"Skipping plot for {code} ({mode_name}): No data found.")
        return

    # Dynamically determine valid X axis order present in data
    unique_round_stages = df_mode[['round_num', 'step', 'round_stage']].drop_duplicates()
    unique_round_stages['step_idx'] = unique_round_stages['step'].map({s: i for i, s in enumerate(STAGE_ORDER)})
    unique_round_stages = unique_round_stages.sort_values(['round_num', 'step_idx'])
    x_order = unique_round_stages['round_stage'].tolist()
    
    if not x_order:
        return

    # Convert to Ordered Categorical for reliable plotting
    df_mode = df_mode.copy()
    df_mode['round_stage'] = pd.Categorical(df_mode['round_stage'], categories=x_order, ordered=True)

    fig = plt.figure(figsize=(22, 16))
    plt.suptitle(f"Failure Mode {code}: {mode_name}\nEvolution across Collaboration Stages", fontsize=28, fontweight='bold', y=0.96)
    
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # Panel A: MAS Evolution (Heatmap)
    # ---------------------------------------------------------
    ax_a = plt.subplot(2, 2, 1)
    
    df_a = df_mode.groupby(['mas', 'round_stage'], observed=False)['failed'].agg(['mean', 'count']).reset_index()
    df_a['failure_rate'] = df_a['mean'] * 100
    
    pivot_a = df_a.pivot(index='mas', columns='round_stage', values='failure_rate')
    
    # Ensure stable ordering of Y axis
    mas_order =["ColaCare", "HealthcareAgent", "MAC", "MDAgents", "MedAgent", "ReConcile"]
    mas_order = [m for m in mas_order if m in df_mode['mas'].unique()]
    pivot_a = pivot_a.reindex(index=mas_order, columns=x_order)
    
    cmap = sns.light_palette("darkred", as_cmap=True)
    
    sns.heatmap(pivot_a, cmap=cmap, annot=True, fmt=".1f", vmin=0, vmax=100,
                cbar_kws={'label': 'Failure Rate (%)'}, ax=ax_a, 
                linewidths=1.5, linecolor='white')
    
    # Process structurally N/A missing elements
    for i in range(pivot_a.shape[0]):
        for j in range(pivot_a.shape[1]):
            if np.isnan(pivot_a.iloc[i, j]):
                ax_a.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor='#E0E0E0', hatch='///', edgecolor='white'))
                
    na_patch = mpatches.Patch(facecolor='#E0E0E0', hatch='///', label='Structurally N/A')
    ax_a.legend(handles=[na_patch], loc='upper left', bbox_to_anchor=(1.0, 1.05), title="", frameon=False, fontsize=12)
    
    ax_a.set_title("A. MAS Framework Evolution", fontsize=20, fontweight='bold', loc='left', pad=15)
    ax_a.set_xlabel("Collaboration Stage", fontsize=16)
    ax_a.set_ylabel("MAS Framework", fontsize=16)
    ax_a.tick_params(axis='x', rotation=45, labelsize=12)
    ax_a.tick_params(axis='y', rotation=0, labelsize=14)

    # ---------------------------------------------------------
    # Panel B: Dataset Evolution (Medical Scenarios)
    # ---------------------------------------------------------
    ax_b = plt.subplot(2, 2, 2)
    
    df_b_agg = df_mode.groupby(['dataset', 'mas', 'llm', 'round_stage'], observed=False)['failed'].mean().reset_index()
    df_b_agg['failure_rate'] = df_b_agg['failed'] * 100
    df_b_agg = df_b_agg.dropna(subset=['failure_rate'])
    
    dataset_palette = {
        "MedQA": "#08306B", "PubMedQA": "#2171B5", "MedXpertQA": "#6BAED6",
        "PathVQA": "#8C2D04", "VQA-RAD": "#D94801", "SLAKE": "#FD8D3C"
    }
    hue_order_b =[d for d in dataset_palette.keys() if d in df_b_agg['dataset'].unique()]
    
    if len(x_order) > 1:
        sns.lineplot(data=df_b_agg, x='round_stage', y='failure_rate', hue='dataset', 
                     palette=dataset_palette, marker='o', markersize=8, errorbar=('ci', 95), 
                     err_style='band', ax=ax_b, hue_order=hue_order_b, linewidth=2.5)
    else:
        # Fallback for single-stage mode
        sns.stripplot(data=df_b_agg, x='round_stage', y='failure_rate', hue='dataset', 
                      palette=dataset_palette, dodge=True, ax=ax_b, hue_order=hue_order_b, jitter=True, alpha=0.7)
        sns.pointplot(data=df_b_agg, x='round_stage', y='failure_rate', hue='dataset', 
                      palette=dataset_palette, dodge=True, ax=ax_b, hue_order=hue_order_b, 
                      errorbar=('ci', 95), linestyle='none', markers='D', capsize=0.1)
        handles, labels = ax_b.get_legend_handles_labels()
        ax_b.legend(handles[:len(hue_order_b)], labels[:len(hue_order_b)], title="Dataset", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))
        
    ax_b.set_title("B. Dataset Evolution (Medical Scenarios)", fontsize=20, fontweight='bold', loc='left', pad=15)
    ax_b.set_xlabel("Collaboration Stage", fontsize=16)
    ax_b.set_ylabel("Failure Rate (%)", fontsize=16)
    ax_b.set_ylim(-5, 105)
    ax_b.tick_params(axis='x', rotation=45, labelsize=12)
    ax_b.tick_params(axis='y', labelsize=12)
    if len(x_order) > 1:
        ax_b.legend(title="Dataset", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))

    # ---------------------------------------------------------
    # Panel C: LLM Evolution (Medical QA)
    # ---------------------------------------------------------
    ax_c = plt.subplot(2, 2, 3)
    
    qa_datasets = ["MedQA", "PubMedQA", "MedXpertQA"]
    df_c = df_mode[df_mode['dataset'].isin(qa_datasets)].copy()
    
    if not df_c.empty:
        df_c_agg = df_c.groupby(['llm', 'mas', 'dataset', 'round_stage'], observed=False)['failed'].mean().reset_index()
        df_c_agg['failure_rate'] = df_c_agg['failed'] * 100
        df_c_agg = df_c_agg.dropna(subset=['failure_rate'])
        
        qa_llm_palette = sns.color_palette("Set1", n_colors=len(df_c_agg['llm'].unique()))
        
        if len(x_order) > 1:
            sns.lineplot(data=df_c_agg, x='round_stage', y='failure_rate', hue='llm', 
                         marker='s', markersize=8, errorbar=('ci', 95), err_style='band', ax=ax_c, palette=qa_llm_palette, linewidth=2.5)
            ax_c.legend(title="QA Models", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))
        else:
            sns.stripplot(data=df_c_agg, x='round_stage', y='failure_rate', hue='llm', 
                          palette=qa_llm_palette, dodge=True, ax=ax_c, jitter=True, alpha=0.7)
            sns.pointplot(data=df_c_agg, x='round_stage', y='failure_rate', hue='llm', 
                          palette=qa_llm_palette, dodge=True, ax=ax_c, errorbar=('ci', 95), 
                          linestyle='none', markers='s', capsize=0.1)
            handles, labels = ax_c.get_legend_handles_labels()
            n_hues = len(df_c_agg['llm'].unique())
            ax_c.legend(handles[:n_hues], labels[:n_hues], title="QA Models", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))
            
        ax_c.set_title("C. Base Model Evolution (Medical QA)", fontsize=20, fontweight='bold', loc='left', pad=15)
        ax_c.set_xlabel("Collaboration Stage", fontsize=16)
        ax_c.set_ylabel("Failure Rate (%)", fontsize=16)
        ax_c.set_ylim(-5, 105)
        ax_c.tick_params(axis='x', rotation=45, labelsize=12)
        ax_c.tick_params(axis='y', labelsize=12)
    else:
        ax_c.text(0.5, 0.5, "No QA Data Available", ha='center', va='center', fontsize=20)
        ax_c.axis('off')

    # ---------------------------------------------------------
    # Panel D: LLM Evolution (Medical VQA)
    # ---------------------------------------------------------
    ax_d = plt.subplot(2, 2, 4)
    
    vqa_datasets =["PathVQA", "VQA-RAD", "SLAKE"]
    df_d = df_mode[df_mode['dataset'].isin(vqa_datasets)].copy()
    
    if not df_d.empty:
        df_d_agg = df_d.groupby(['llm', 'mas', 'dataset', 'round_stage'], observed=False)['failed'].mean().reset_index()
        df_d_agg['failure_rate'] = df_d_agg['failed'] * 100
        df_d_agg = df_d_agg.dropna(subset=['failure_rate'])
        
        vqa_llm_palette = sns.color_palette("Set2", n_colors=len(df_d_agg['llm'].unique()))
        
        if len(x_order) > 1:
            sns.lineplot(data=df_d_agg, x='round_stage', y='failure_rate', hue='llm', 
                         marker='^', markersize=8, errorbar=('ci', 95), err_style='band', ax=ax_d, palette=vqa_llm_palette, linewidth=2.5)
            ax_d.legend(title="VQA Models", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))
        else:
            sns.stripplot(data=df_d_agg, x='round_stage', y='failure_rate', hue='llm', 
                          palette=vqa_llm_palette, dodge=True, ax=ax_d, jitter=True, alpha=0.7)
            sns.pointplot(data=df_d_agg, x='round_stage', y='failure_rate', hue='llm', 
                          palette=vqa_llm_palette, dodge=True, ax=ax_d, errorbar=('ci', 95), 
                          linestyle='none', markers='^', capsize=0.1)
            handles, labels = ax_d.get_legend_handles_labels()
            n_hues = len(df_d_agg['llm'].unique())
            ax_d.legend(handles[:n_hues], labels[:n_hues], title="VQA Models", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))
            
        ax_d.set_title("D. Vision Model Evolution (Medical VQA)", fontsize=20, fontweight='bold', loc='left', pad=15)
        ax_d.set_xlabel("Collaboration Stage", fontsize=16)
        ax_d.set_ylabel("Failure Rate (%)", fontsize=16)
        ax_d.set_ylim(-5, 105)
        ax_d.tick_params(axis='x', rotation=45, labelsize=12)
        ax_d.tick_params(axis='y', labelsize=12)
    else:
        ax_d.text(0.5, 0.5, "No VQA Data Available", ha='center', va='center', fontsize=20)
        ax_d.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    filename = f"Failure_Mode_{code}_Evolution.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=500, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive figure: {save_path}")


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
        plot_failure_mode_comprehensive(code, df_mode, output_dir)

    print("\nVisualization Audit Completed.")

if __name__ == "__main__":
    main()