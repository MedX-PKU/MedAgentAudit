"""
Script: calculate_human_eval_metrics.py
Purpose: Calculate human evaluation metrics for the MedAgentAudit project.

Tasks:
1. Open-coding Phase: Calculate Cohen's Kappa for 10 failure modes (Binary per category) 
   and Macro-average across two independent human annotators.
2. Audit Phase: Calculate Sensitivity, Specificity, F1-Score, and Human-AI Agreement (Cohen's Kappa)
   between the automated audit system and majority-voted human annotations.
   Additionally, calculate Fleiss' Kappa for Inter-Rater Reliability (IRR) among 3 human annotators.
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


def compute_fleiss_kappa(votes_list, n_categories=2):
    """
    Calculate Generalized Fleiss' Kappa for multiple raters (Inter-Rater Reliability). 
     handles potential missing raters using the generalized statistical formula.
    
    :param votes_list: list of lists, where each inner list contains binary ratings for a single case.
                       e.g. [[1, 0, 1],[0, 0, 0], [1, 1, 1]]
    :param n_categories: Number of categories (2 for Binary classification: 0/1)
    :return: Fleiss' Kappa score
    """
    # Filter cases strictly with at least 2 raters (Fleiss requires >1 to compute agreement)
    valid_votes = [votes for votes in votes_list if len(votes) >= 2] # vote is a list of votes for a single case, e.g. [1, 0, 1]
    N = len(valid_votes) # Number of subjects/cases with valid ratings
    if N == 0:
        return np.nan
        
    n_ij = np.zeros((N, n_categories)) # shape (N, 2) for binary classification, every classification category has a count of votes for each case
    n_i = np.zeros(N) # (N,) here N = 40 
    
    # 1. Count assignments
    for i, votes in enumerate(valid_votes):
        n_i[i] = len(votes) # len(votes)=3
        for v in votes:
            n_ij[i, int(v)] += 1
            
    # 2. Calculate p_j (Proportion of all assignments to category j)
    total_assignments = np.sum(n_i) # total votes across all cases
    p_j = np.sum(n_ij, axis=0) / total_assignments # np.sum(n_ij, axis=0) means summing across rows for each category, here total_assignment is 120, shape (2,)
    P_e = np.sum(p_j**2) # Expected agreement chance
    
    # 3. Calculate P_i (Extent of agreement for the i-th subject)
    P_i = np.zeros(N)
    for i in range(N):
        P_i[i] = (np.sum(n_ij[i]**2) - n_i[i]) / (n_i[i] * (n_i[i] - 1)) # n_i[i] = 3, so n_i[i]*(n_i[i]-1) = 6, 
        
    P_bar = np.mean(P_i) # Observed overall agreement
    
    # 4. Handle edge case for perfect agreement distribution
    if P_e == 1.0:
        return 1.0 if P_bar == 1.0 else 0.0
        
    # 5. Final Fleiss' Kappa
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def calculate_open_coding_agreement():
    """
    Protocol Sec 1.3.2.9:
    Calculate Binary Cohen's Kappa per category and the Macro-average 
    between two independent annotators for the open-coding phase.
    """
    print("Starting Open-Coding Agreement Calculation...")
    
    files = list(open_coding_results_dir.glob("*.json"))
    if len(files) < 2:
        print("Warning: Insufficient open-coding files found (Needs 2). Skipping Open-coding phase.")
        return None, 0.0

    # Load annotations for both reviewers
    data_rev1 = load_json(files[0]).get('annotations', {})
    data_rev2 = load_json(files[1]).get('annotations', {})

    def extract_mapping(data):
        mapping = {}
        for key, val in data.items():
            uid = f"{val['caseId'].lower()}_{val['dataset'].lower()}_{val['mas'].lower()}"
            mapping[uid] = val.get('taxonomy',[])
        return mapping

    map1 = extract_mapping(data_rev1)
    map2 = extract_mapping(data_rev2)

    common_cases = set(map1.keys()).intersection(set(map2.keys()))
    print(f"Found {len(common_cases)} common cases annotated by both reviewers.")

    kappas = {}
    for mode in FAILURE_MODES:
        y_rev1, y_rev2 = [],[]
        for case in common_cases:
            y_rev1.append(1 if mode in map1[case] else 0)
            y_rev2.append(1 if mode in map2[case] else 0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(y_rev1, y_rev2)
            if np.isnan(kappa):
                kappa = 1.0 if y_rev1 == y_rev2 else 0.0
        kappas[mode] = kappa

    macro_kappa = np.mean(list(kappas.values()))
    
    df_kappa = pd.DataFrame([kappas]).T
    df_kappa.columns =["Cohen's Kappa"]
    df_kappa.index.name = "Failure Mode"
    
    print(f"Open-Coding Macro-average Kappa: {macro_kappa:.4f}")
    return df_kappa, macro_kappa


def calculate_audit_metrics():
    """
    Protocol Sec 2.3.2:
    Calculate Sensitivity, Specificity, F1-Score, and Cohen's Kappa 
    between Automated Audit System and Human Majority Vote.
    Includes Human Inter-Rater Reliability (Fleiss' Kappa) computation.
    """
    print("Starting Autonomous Audit Metrics Calculation...")

    # 1. Process Human Annotations & Calculate Fleiss' Kappa (IRR)
    human_files = list(audit_results_dir.glob("*.json"))
    human_votes = defaultdict(list)

    for f in human_files:
        data = load_json(f).get('annotations', {})
        for _, val in data.items():
            # Composite key for absolute uniqueness
            uid = f"{val['caseId'].lower()}_{val['dataset'].lower()}_{val['mas'].lower()}_{val['llm'].lower()}_{val['taxonomyKey']}"
            vote = 1 if val.get('verdict', '').lower() == 'yes' else 0
            human_votes[uid].append(vote)
            
    r_counts =[len(v) for v in human_votes.values()]
    if r_counts:
        print(f">> Human Audit Stats -> Total Unique Audit Instances: {len(human_votes)}, "
              f"Average raters per case: {np.mean(r_counts):.2f} (Min: {np.min(r_counts)}, Max: {np.max(r_counts)})")

    # Calculate Fleiss' Kappa for human evaluators per failure mode
    print("Calculating Fleiss' Kappa (IRR) for 3 Human Annotators...")
    human_irr_dict = {}
    for mode in FAILURE_MODES:
        mode_votes =[votes for uid, votes in human_votes.items() if uid.endswith(f"_{mode}")]
        human_irr_dict[mode] = compute_fleiss_kappa(mode_votes)

    # 2. Resolve Majority Vote
    human_ground_truth = {}
    for uid, votes in human_votes.items():
        majority = 1 if sum(votes) >= len(votes) / 2.0 else 0
        human_ground_truth[uid] = majority
        
    print(f"Aggregated majority votes for {len(human_ground_truth)} unique instances.")

    # 3. Process Autonomous System JSONL Logs
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
                
    print(f"Loaded {len(ai_predictions)} predictions from Autonomous Audit System.")

    # 4. Calculate Metrics per Failure Mode
    metrics_records =[]
    
    for mode in FAILURE_MODES:
        y_true, y_pred = [],[]
        
        # Match evaluations
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
            "Human IRR (Fleiss' Kappa)": round(human_irr_dict.get(mode, np.nan), 4),
            "Sensitivity (Recall)": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "F1-Score": round(f1, 4),
            "AI-Human Agreement (Cohen's Kappa)": round(kappa, 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        })

    # 5. Generate Pandas DataFrame & Calculate Macro-Averages
    df_metrics = pd.DataFrame(metrics_records).set_index("Failure Mode")
    
    # Calculate Macro Averages for main paper table
    macro_metrics = df_metrics[[
        "Human IRR (Fleiss' Kappa)", "Sensitivity (Recall)", 
        "Specificity", "F1-Score", "AI-Human Agreement (Cohen's Kappa)"
    ]].mean()
    
    df_metrics.loc["Macro-Average"] =[
        df_metrics["Sample Size (N)"].sum(),
        macro_metrics["Human IRR (Fleiss' Kappa)"],
        macro_metrics["Sensitivity (Recall)"],
        macro_metrics["Specificity"],
        macro_metrics["F1-Score"],
        macro_metrics["AI-Human Agreement (Cohen's Kappa)"],
        df_metrics["TP"].sum(), df_metrics["TN"].sum(), 
        df_metrics["FP"].sum(), df_metrics["FN"].sum()
    ]
    
    return df_metrics

def main():
    print("="*75)
    print(" 📊 MedAgentAudit - Evaluation Metrics Calculation")
    print("="*75)
    
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
        # Enhance Table formatting for console output
        print(audit_metrics_df.to_string(na_rep="NaN"))
        
        # Write to files
        if open_coding_df is not None:
            open_coding_df.to_csv(human_eval_results_path / "opencoding_agreement.csv")
        audit_metrics_df.to_csv(human_eval_results_path / "audit_metrics.csv")
        print(f"\nSaved publication tables to {human_eval_results_path}")

if __name__ == "__main__":
    main()
