"""
Script: calculate_human_eval_metrics.py
Purpose: Calculate human evaluation metrics for the MedAgentAudit project.

Tasks:
1. Open-coding Phase: Calculate Cohen's Kappa for 10 failure modes (Binary per category) 
   and Macro-average across two independent human annotators.
2. Audit Phase: Calculate Sensitivity, Specificity, F1-Score, and Human-AI Agreement (Cohen's Kappa)
   between the automated audit system and majority-voted human annotations.
"""

import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from medagentaudit.utils.json_utils import load_json
from medagentaudit.utils.logger import DualLogger


# System Paths
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
human_eval_results_path = project_root / "logs" / "human_eval_results"
open_coding_results_dir = human_eval_results_path / "opencoding_results"
audit_results_dir = human_eval_results_path / "audit_results"
autonomous_audit_results_path = project_root / "logs" / "structured_logs_for_audit_human_evaluation"

terminal_log_file = human_eval_results_path / "human_eval.log"
terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
sys.stdout = DualLogger(terminal_log_file, sys.stdout)
sys.stderr = DualLogger(terminal_log_file, sys.stderr)
# The 10 Failure Modes defined in the Codebook
FAILURE_MODES =[
    "1.1.1", "1.2.1", 
    "2.1.1", "2.1.2", "2.2.1", "2.2.2", 
    "3.1.1", "3.1.2", "3.1.3", "3.2.1"
]

def calculate_open_coding_agreement():
    """
    Protocol Sec 1.3.2.9:
    Calculate Binary Cohen's Kappa per category and the Macro-average 
    between two independent annotators for the open-coding phase.
    """
    print("Starting Open-Coding Agreement Calculation...")
    
    files = list(open_coding_results_dir.glob("*.json"))

    # Load annotations for both reviewers
    data_rev1 = load_json(files[0]).get('annotations', {})
    data_rev2 = load_json(files[1]).get('annotations', {})

    # Create case-to-taxonomy mapping
    # Using composite key: caseId + dataset + mas to ensure absolute uniqueness
    def extract_mapping(data):
        mapping = {}
        for key, val in data.items():
            uid = f"{val['caseId'].lower()}_{val['dataset'].lower()}_{val['mas'].lower()}"
            mapping[uid] = val.get('taxonomy',[])
        return mapping

    map1 = extract_mapping(data_rev1)
    map2 = extract_mapping(data_rev2)

    # Find intersection of annotated cases
    common_cases = set(map1.keys()).intersection(set(map2.keys()))
    print(f"Found {len(common_cases)} common cases annotated by both reviewers.")

    kappas = {}
    
    for mode in FAILURE_MODES:
        y_rev1, y_rev2 = [],[]
        for case in common_cases:
            # "0.0.0" means no failure. If mode in taxonomy, value is 1, else 0.
            y_rev1.append(1 if mode in map1[case] else 0)
            y_rev2.append(1 if mode in map2[case] else 0)
        
        # Calculate Kappa for this specific failure mode
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(y_rev1, y_rev2)
            # Handle edge cases where all annotations are 0 (Kappa returns NaN)
            if np.isnan(kappa):
                kappa = 1.0 if y_rev1 == y_rev2 else 0.0
                
        kappas[mode] = kappa

    # Compute Macro-average
    macro_kappa = np.mean(list(kappas.values()))
    
    # Format into DataFrame for paper
    df_kappa = pd.DataFrame([kappas]).T
    df_kappa.columns = ["Cohen's Kappa"]
    df_kappa.index.name = "Failure Mode"
    
    print(f"Open-Coding Macro-average Kappa: {macro_kappa:.4f}")
    return df_kappa, macro_kappa


def calculate_audit_metrics():
    """
    Protocol Sec 2.3.2:
    Calculate Sensitivity, Specificity, F1-Score, and Cohen's Kappa 
    between Automated Audit System and Human Majority Vote.
    """
    print("Starting Autonomous Audit Metrics Calculation...")

    # 1. Process Human Annotations (Majority Vote)
    human_files = list(audit_results_dir.glob("*.json"))
    human_votes = defaultdict(list) # TODO: this place has collected 3 human's decision, now need to calculate the irr (Fleiss' Kappa)

    for f in human_files:
        data = load_json(f).get('annotations', {})
        for _, val in data.items():
            # Composite key: caseId + dataset + mas + llm + failure_code (taxonomyKey)
            uid = f"{val['caseId'].lower()}_{val['dataset'].lower()}_{val['mas'].lower()}_{val['llm'].lower()}_{val['taxonomyKey']}"
            vote = 1 if val.get('verdict', '').lower() == 'yes' else 0
            human_votes[uid].append(vote)

    # Resolve Majority Vote (Protocol requires 3 annotators, 2/3 majority)
    human_ground_truth = {}
    for uid, votes in human_votes.items():
        majority = 1 if sum(votes) >= len(votes) / 2.0 else 0
        human_ground_truth[uid] = majority
        
    print(f"Aggregated majority votes for {len(human_ground_truth)} unique human audit instances.")

    # 2. Process Autonomous System JSONL Logs
    ai_predictions = {}
    ai_files = list(autonomous_audit_results_path.glob("*.jsonl"))
    
    for f in ai_files:
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip(): 
                    continue
                val = json.loads(line.strip())
                uid = f"{val['qid'].lower()}_{val['dataset'].lower()}_{val['mas'].lower()}_{val['llm'].lower()}_{val['failure_code']}"
                ai_predictions[uid] = int(val['mas_audit_result'])
                
    print(f"Loaded {len(ai_predictions)} predictions from autonomous audit system.")

    # 3. Calculate Metrics per Failure Mode
    metrics_records =[]
    
    for mode in FAILURE_MODES:
        y_true, y_pred = [],[]
        
        # We find all evaluated instances for the current failure mode
        for uid in human_ground_truth.keys():
            if uid.endswith(f"_{mode}"):
                if uid in ai_predictions:
                    y_true.append(human_ground_truth[uid])
                    y_pred.append(ai_predictions[uid])
                else:
                    print(f"UID {uid} missing in AI predictions.")

        if not y_true:
            print(f"No matched pairs found for failure mode {mode}")
            continue

        # Compute Confusion Matrix Elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Metrics Calculation
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(y_true, y_pred)
            if np.isnan(kappa):
                kappa = 1.0 if y_true == y_pred else 0.0

        metrics_records.append({
            "Failure Mode": mode,
            "Sample Size (N)": len(y_true),
            "Sensitivity (Recall)": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "F1-Score": round(f1, 4),
            "Cohen's Kappa": round(kappa, 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        })

    # 4. Generate Pandas DataFrame & Calculate Macro-Averages
    df_metrics = pd.DataFrame(metrics_records).set_index("Failure Mode")
    
    # Calculate Macro Averages for main paper table
    macro_metrics = df_metrics[["Sensitivity (Recall)", "Specificity", "F1-Score", "Cohen's Kappa"]].mean()
    df_metrics.loc["Macro-Average"] = [
        df_metrics["Sample Size (N)"].sum(),
        macro_metrics["Sensitivity (Recall)"],
        macro_metrics["Specificity"],
        macro_metrics["F1-Score"],
        macro_metrics["Cohen's Kappa"],
        df_metrics["TP"].sum(), df_metrics["TN"].sum(), 
        df_metrics["FP"].sum(), df_metrics["FN"].sum()
    ]
    
    return df_metrics

def main():
    print("="*60)
    print(" 📊 MedAgentAudit - Nature Medicine Submission Metrics")
    print("="*60)
    
    # 1. Open-Coding
    open_coding_df, open_coding_macro = calculate_open_coding_agreement()
    if open_coding_df is not None:
        print("\n---[Table 1: Open-Coding Inter-Annotator Agreement] ---")
        print(open_coding_df.to_string())
        print(f"\n>> Overall Macro-Average Kappa: {open_coding_macro:.4f}")
    
    # 2. Audit Metrics
    audit_metrics_df = calculate_audit_metrics()
    if audit_metrics_df is not None:
        print("\n---[Table 2: Automated Audit System vs Human Experts] ---")
        print(audit_metrics_df.to_string())
        
        if open_coding_df is not None:
            open_coding_df.to_csv(human_eval_results_path / "opencoding_agreement.csv")
        audit_metrics_df.to_csv(human_eval_results_path / "audit_metrics.csv")
        print(f"Saved publication tables to {human_eval_results_path}")

if __name__ == "__main__":
    main()