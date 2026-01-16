'''
./scripts/audit_stas.py
this script is to compute the statistics for audited results from multi-agent systems
'''
from openai import OpenAI
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import time
import argparse
from tqdm import tqdm
import sys
import csv
from pathlib import Path

# Ensure project root is in path
current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
project_root = current_file_path.parents[1]
utils_root = project_root / "medagentaudit" / "utils"
sys.path.extend([str(project_root), str(utils_root)])
from logger import DualLogger

def statistics_of_audit_role_assignment(audit_results_path: Path, metrics_folder_path: Path) -> float:
    '''
    failure mode 2.1.1 
    this function is to compute the rate of false role assignment in the audited case history.
    we import the audit results from the given path, and we output the rate of false role assignment.
    and export specific failure rates for each framework-dataset combination.
    '''
    if not audit_results_path.exists():
        print(f"The path {audit_results_path} does not exist.")
        raise FileNotFoundError(f"The path {audit_results_path} does not exist.")
    
    json_files = list(audit_results_path.rglob("*.json"))
    print(f"Found {len(json_files)} audit result files in {audit_results_path}.")

    # Defining the scope
    mas_list = ['colacare', 'healthcareagent', 'mac', 'mdagents', 'medagent', 'reconcile']
    datasets_list = ['MedQA', 'MedXpertQA-text', 'PubMedQA', 'PathVQA', 'Slake', 'VQA-RAD']

    # Data structures for statistics
    total_role_assignments = 0
    false_role_assignments = 0
    
    # Dictionary to store stats: Key=(mas, dataset), Value={'total': 0, 'false': 0}
    stats_by_group: Dict[Tuple[str, str], Dict[str, int]] = {}

    for json_file in tqdm(json_files, desc="Processing audit files"): # Added tqdm for better progress visualization
        
        # 1. Identify MAS and Dataset from file path
        file_path_str = str(json_file)
        print(f"Processing file: {json_file}")
        current_mas = "Unknown"
        current_dataset = "Unknown"

        for m in mas_list:
            if m in file_path_str.split('/'):
                current_mas = m
                print(f"Current MAS identified: {current_mas}")
                break
        
        for d in datasets_list:
            if d in file_path_str.split('/'):
                current_dataset = d
                print(f"Current Dataset identified: {current_dataset}")
                break
        
        group_key = (current_mas, current_dataset)
        if group_key not in stats_by_group:
            stats_by_group[group_key] = {'total': 0, 'false': 0}

        # 2. Process JSON Content
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                audit_results_data = json.load(f)
                
            case_history = audit_results_data.get("case_history", [])
            
            if isinstance(case_history, dict):
                audit_data = case_history.get("audit", None)
            else:
                # Fallback if case_history is a list (common in some frameworks), usually audit is not inside a list history directly
                audit_data = None 

            if not audit_data:
                print(f"No audit data found in file: {json_file}")
                continue
            
            # 3. Count Failures
            rounds = audit_data.get("rounds", [])
            if not rounds: continue

            for audit_round_data in rounds:
                role_assignments = audit_round_data.get("2_1_1_role_assignment", [])
                if role_assignments:
                    for assignment in role_assignments:
                        # Global stats
                        total_role_assignments += 1
                        
                        # Group stats
                        stats_by_group[group_key]['total'] += 1

                        audit_res = assignment.get("audit_result", {})
                        if audit_res.get("role_task_alignment") == "1":
                            # Global stats
                            false_role_assignments += 1
                            # Group stats
                            stats_by_group[group_key]['false'] += 1
                            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # --- Output Global Statistics ---
    if total_role_assignments == 0:
        print("No role assignments found in the audit results.")
        return 0.0
    
    global_rate = false_role_assignments / total_role_assignments
    print(f"\n=== Global Statistics ===")
    print(f"Total Assignments: {total_role_assignments}")
    print(f"False Assignments: {false_role_assignments}")
    print(f"False Role Assignment Rate (Global): {global_rate:.2%}\n")

    # --- Compute and Export Group Statistics ---
    csv_output_path = metrics_folder_path / "role_assignment_failure_rates.csv"
    print(f"Exporting group statistics to: {csv_output_path}")

    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Framework', 'Dataset', 'Total_Assignments', 'False_Assignments', 'Failure_Rate']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Sort keys for cleaner output (by Framework then Dataset)
        sorted_keys = sorted(stats_by_group.keys())

        for mas, dataset in sorted_keys:
            data = stats_by_group[(mas, dataset)]
            total = data['total']
            false_count = data['false']
            rate = (false_count / total) if total > 0 else 0.0
            
            writer.writerow({
                'Framework': mas,
                'Dataset': dataset,
                'Total_Assignments': total,
                'False_Assignments': false_count,
                'Failure_Rate': f"{rate:.4f}" # Retain precision for CSV
            })
            
            # Optional: Print to console as well
            if total > 0:
                print(f"[{mas} - {dataset}]: Rate = {rate:.2%} ({false_count}/{total})")

    return global_rate


if __name__ == "__main__":
    # Ensure this matches your directory structure
    timestamp = "20260112"
    metrics_folder_path = project_root / "logs" / "metrics" / timestamp
    terminal_log_dir = metrics_folder_path / "terminal_log"
    terminal_log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_file = terminal_log_dir / f"{timestamp}_metrics.log"
    
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    audit_result_path = project_root / "logs" / "audit_results" / timestamp
    
    statistics_of_audit_role_assignment(audit_result_path, metrics_folder_path)