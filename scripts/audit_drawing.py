'''
this script is to visualize the failure rates for each failure mode across different rounds and stages, based on the audit results from multi-agent systems. 
The granularity of visualization is at the level of each failure mode and every round, showing how the failure rates evolve as the discussion progresses through different stages (e.g., role assignment, analysis, synthesis, review, decision-making). The output will be a set of bar charts that illustrate the failure rates for each mode across rounds and stages, 
allowing us to identify patterns and trends in the failures.
'''
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))

AUDIT_CONFIG = {
    "1.1.1": {
        "status_key": "factual_hallucination_status",
        "name": "Factual Hallucination",
        "valid_rounds": [1, 2, 3]
    },
    "1.2.1": {
        "status_key": "modality_neglect_status",
        "name": "Modality Neglect",
        "valid_rounds": [1, 2, 3]
    },
    "2.1.1": {
        "status_key": "role_task_alignment",
        "name": "Role Assignment Mismatch",
        "valid_rounds": [1]
    },
    "2.1.2": {
        "status_key": "knowledge_activation_status",
        "name": "Knowledge Activation Fail",
        "valid_rounds": [1, 2, 3]
    },
    "2.2.1": {
        "status_key": "interaction_redundancy",
        "name": "Interaction Redundancy",
        "valid_rounds": [1, 2, 3] 
    },
    "2.2.2": {
        "status_key": "conflict_resolution_status",
        "name": "Unresolved Conflicts",
        "valid_rounds": [1, 2, 3]
    },
    "3.1.1": {
        "status_key": "suppression_status",
        "name": "Minority Suppression",
        "valid_rounds": [1, 2, 3]
    },
    "3.1.2": {
        "status_key": "authority_bias_status",
        "name": "Authority Bias",
        "valid_rounds": [1, 2, 3]
    },
    "3.1.3": {
        "status_key": "neglect_of_conflict_status",
        "name": "Neglect of Contradictions",
        "valid_rounds": [1, 2, 3]
    },
    "3.2.1": {
        "status_key": "inter_round_consistency_status",
        "name": "Inter-round Inconsistency",
        "valid_rounds": [2, 3]
    }
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
    "role_assignment": "Assign", # simplify for better display
    "analysis": "Analysis",
    "synthesis": "Synthesis",
    "review": "Review",
    "decision": "Decision"
}

# style configuration for plots
PLOT_STYLE = {
    "font_label": 24,       
    "font_title": 28,
    "font_tick": 14,
    "font_bar_text": 12,
    "color_palette": "pastel",
    "grid_alpha": 0.3,
    "grid_style": "--",
    "y_limit_max": 105
}

def apply_axis_style(ax, title, xlabel="Discussion Progression (Round - Stage)", ylabel="Failure Rate (%)"):
    ax.set_title(title, fontsize=PLOT_STYLE["font_title"], pad=20)
    ax.set_xlabel(xlabel, fontsize=PLOT_STYLE["font_label"])
    ax.set_ylabel(ylabel, fontsize=PLOT_STYLE["font_label"])
    ax.tick_params(axis='x', labelrotation=45, labelsize=PLOT_STYLE["font_tick"])
    ax.tick_params(axis='y', labelsize=PLOT_STYLE["font_tick"])
    ax.yaxis.grid(True, linestyle=PLOT_STYLE["grid_style"], alpha=PLOT_STYLE["grid_alpha"])
    ax.set_ylim(0, PLOT_STYLE["y_limit_max"])
    sns.despine()


def process_audit_data(audit_results_path: Path):
    """
    Traverse the files and aggregate the data by (FailureMode → Round → Stage).
    """
    # structure: stats[code][round_num][stage_name] = {'total': 0, 'failed': 0}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total': 0, 'failed': 0})))
    
    jsonl_files = list(audit_results_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} files. Processing...")

    for jsonl_file in tqdm(jsonl_files, desc="Scanning Logs"):
        # remove error log
        if "errors" in jsonl_file.name:
            continue
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): 
                        continue 
                    case_data = json.loads(line)
                    
                    # extract audit data
                    audit_data = None
                    if "case_history" in case_data and "audit" in case_data["case_history"]:
                        audit_data = case_data["case_history"]["audit"]
                    
                    if not audit_data: 
                        continue

                    rounds = audit_data.get("rounds", [])
                    
                    for r_data in rounds:
                        round_num = r_data.get("round")
                        if not round_num: 
                            continue

                        for code, config in AUDIT_CONFIG.items():
                            # check the round restriction for this failure mode
                            if round_num not in config["valid_rounds"]:
                                continue

                            log_key = LOG_KEY_MAP[code]
                            entries = r_data.get(log_key, [])
                            
                            if isinstance(entries, dict): 
                                entries = [entries]
                            if not entries: 
                                continue

                            status_key = config["status_key"]

                            for entry in entries:
                                step = entry.get("step", "unknown").lower()
                                if step not in STAGE_ORDER:
                                    # some old log might not have "step" field, we just skip those entries for stage-specific stats
                                    continue
                                
                                # statistics aggregation
                                stats[code][round_num][step]['total'] += 1
                                
                                result_obj = entry.get("audit_result", {})
                                # if status == 1 or True, we consider it as a failure case for that mode
                                status_val = result_obj.get(status_key)
                                is_failed = str(status_val) == "1" or status_val is True
                                
                                if is_failed:
                                    stats[code][round_num][step]['failed'] += 1

        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")
            continue
            
    return stats

def plot_failure_mode(code, mode_stats, output_dir):
    """
    draw a bar chart for the given failure mode, showing the failure rates across rounds and stages.
    X axis: R1-Analysis, R1-Synthesis ... R3-Decision
    Y axis: failure rate in percentage
    for failure mode 2.1.1, we only have round 1 data, so we need to adjust the width of the bar.
    """
    config = AUDIT_CONFIG[code]
    valid_rounds = sorted(config["valid_rounds"])
    
    x_labels = []
    y_values = []
    frac_labels = []
    bar_colors = []
    
    # define color：R1=Blueish, R2=Orangish, R3=Greenish (use Seaborn muted palette for better print visibility)
    # use the first 3 colors from the "muted" palette for better visibility in print
    round_palette = sns.color_palette("muted", 3)
    color_map = {1: round_palette[0], 2: round_palette[1], 3: round_palette[2]}

    has_data = False

    for r in valid_rounds:
        # accord Analysis -> Synthesis -> Review -> Decision order.
        # some cases has role_assignment stage, but some doesn't, we just put it in the front if exists, and the rest stages are in fixed order.
        stages_to_check = STAGE_ORDER
        
        for stage in stages_to_check:
            data_point = mode_stats[r].get(stage)
            
            if data_point and data_point['total'] > 0:
                has_data = True
                rate = (data_point['failed'] / data_point['total']) * 100
                frac_labels.append(f"{data_point['failed']}/{data_point['total']}")
                
                label = f"R{r}-{STAGE_DISPLAY.get(stage, stage)}"
                x_labels.append(label)
                y_values.append(rate)
                bar_colors.append(color_map.get(r, (0.5, 0.5, 0.5))) # grey

    if not has_data:
        print(f"Skipping plot for {code} ({config['name']}): No data found.")
        return

    # create bar plot
    # For failure mode 2.1.1 we only have one round, which often results in a visually "thin"
    # figure with too much horizontal whitespace if we reuse the general sizing rule.
    if code == "2.1.1":
        fig, ax = plt.subplots(figsize=(5, 7))
        bar_width = 0.3
        ax.set_xlim(-0.3, 0.3)
    else:
        fig, ax = plt.subplots(figsize=(max(10, len(x_labels) * 1.2), 7))
        bar_width = 0.6

    bars = ax.bar(x_labels, y_values, color=bar_colors, edgecolor='black', alpha=0.85, width=bar_width)

    # label value on top of bars
    for bar, val, frac in zip(bars, y_values, frac_labels):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 1, 
            f"{val:.1f}%\n({frac})",
            ha='center', va='bottom', 
            fontsize=PLOT_STYLE["font_bar_text"],
            fontweight='bold'
        )

    # use the style function to apply consistent styling
    title = f"Failure Mode {code}: {config['name']}\n(Global Failure Rate across Rounds)"
    apply_axis_style(ax, title)

    # add legend for rounds
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map[r], edgecolor='black', label=f'Round {r}')
        for r in valid_rounds
    ]
    ax.legend(handles=legend_elements, title="Collaboration Round", fontsize=14, title_fontsize=16)

    plt.tight_layout()
    
    filename = f"Failure_Mode_{code}.pdf"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {save_path}")


def main():
    input_base_dir = project_root / "logs" / "audit_results" / "20260213"
    output_dir = project_root / "logs" / "audit_results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {output_dir}")

    global_stats = process_audit_data(input_base_dir)

    print("\nGenerating Plots...")
    for code in AUDIT_CONFIG.keys():
        plot_failure_mode(code, global_stats[code], output_dir)

    print("\nVisualization Audit Completed.")

if __name__ == "__main__":
    main()
