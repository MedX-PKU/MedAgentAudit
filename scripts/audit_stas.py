'''
./scripts/audit_stas.py
this script is to compute the statistics for audited results from multi-agent systems
we need to give statistics 3 dimensions: mas, dataset, llm
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
import pandas as pd

# Ensure project root is in path
current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
project_root = current_file_path.parents[1]
utils_root = project_root / "medagentaudit" / "utils"
sys.path.extend([str(project_root), str(utils_root)])
from logger import DualLogger

AUDIT_CONFIG = {
    "2.1.1": {
        "log_key": "2_1_1_role_assignment", 
        "status_key": "role_task_alignment",
        "sheet_name": "2_1_1_role_assignment_rates",
        "headers": ['framework', 'dataset', 'llm', 'total_assignments_audited', 'role_mismatch_count', 'failure_rate']
    },
    "2.1.2": {
        "log_key": "2_1_2_domain_specific_knowledge_activation",
        "status_key": "knowledge_activation_status",
        "sheet_name": "2_1_2_knowledge_activation_rates",
        "headers": ['framework', 'dataset', 'llm', 'total_activations_audited', 'generic_response_count', 'failure_rate']
    },
    "2.2.1": {
        "log_key": "2_2_1_repetition_of_initial_views", 
        "status_key": "interaction_redundancy",
        "sheet_name": "2_2_1_interaction_redundancy_rates",
        "headers": ['framework', 'dataset', 'llm', 'total_interactions_audited', 'redundant_count', 'failure_rate']
    },
    "2.2.2": {
        "log_key": "2_2_2_unresolved_conflicts",
        "status_key": "conflict_resolution_status",
        "sheet_name": "2_2_2_conflict_resolution_rates",
        "headers": ['framework', 'dataset', 'llm', 'total_conflicts_audited', 'unresolved_conflict_count', 'failure_rate']
    },
    "3.1.1": {
        "log_key": "3_1_1_suppression_of_minority_views", 
        "status_key": "suppression_status",
        "sheet_name": "3_1_1_minority_suppression_rates",
        "headers": ['framework', 'dataset', 'llm', 'total_decisions_audited', 'minority_suppression_count', 'failure_rate']
    },
    "3.1.2": {
        "log_key": "3_1_2_authority_bias", 
        "status_key": "authority_bias_status",
        "sheet_name": "3_1_2_authority_bias_rates",
        "headers": ['framework', 'dataset', 'llm', 'total_reasoning_audited', 'authority_bias_count', 'failure_rate']
    },
    "3.1.3": {
        "log_key": "3_1_3_neglect_of_contradictions", 
        "status_key": "neglect_of_conflict_status",
        "sheet_name": "3_1_3_neglect_of_conflict_rates",
        "headers": ['framework', 'dataset', 'llm', 'total_consensus_audited', 'false_consensus_count', 'failure_rate']
    },
    "3.2.1": {
        "log_key": "3_2_1_self_contradiction_when_decision", 
        "status_key": "inter_round_consistency_status",
        "sheet_name": "3_2_1_inter_round_consistency_rates",
        "headers": ['framework', 'dataset', 'llm', 'total_consistency_audited', 'inconsistent_flip_count', 'failure_rate']
    }
}

def stas_of_audit_results(audit_results_path: Path, metrics_folder_path: Path) -> float:
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
    llm_models = [
        "gpt-5.2",
        "deepseek-reasoner",
        "glm-4.6v",
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "qwen3-vl-8b-thinking",
        "qwen3-8b"
    ]
    
    # Dictionary to store stats: Key=(mas, dataset), Value={'total': 0, 'false': 0}
    stats = {code: {} for code in AUDIT_CONFIG.keys()}

    for json_file in tqdm(json_files, desc="Processing audit files"): # Added tqdm for better progress visualization
        
        # 1. Identify MAS and Dataset from file path
        file_path_str = str(json_file)
        print(f"Processing file: {json_file}")
        current_mas = "Unknown"
        current_dataset = "Unknown"
        current_llm = "Unknown"

        current_mas = next((m for m in mas_list if m in file_path_str.split('/')), "Unknown") ; print(f"Identified MAS: {current_mas}")
        current_dataset = next((d for d in datasets_list if d in file_path_str.split('/')), "Unknown") ; print(f"Identified Dataset: {current_dataset}")
        current_llm = next((l for l in llm_models if l in file_path_str.split('/')), "Unknown") ; print(f"Identified LLM: {current_llm}")
        
        group_key = (current_mas, current_dataset, current_llm)

        for code in AUDIT_CONFIG:
            if group_key not in stats[code]:
                stats[code][group_key] = {'total': 0, 'failed': 0}

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

            for audit_round in rounds:
                for code, config in AUDIT_CONFIG.items():
                    log_key = config["log_key"]
                    status_key = config["status_key"]
                    
                    entries = audit_round.get(log_key, [])
                    
                    if entries:
                        for entry in entries:
                            stats[code][group_key]['total'] += 1
                            # Check failure status ("1" is failure)
                            result_obj = entry.get("audit_result", {})
                            if result_obj.get(status_key) == "1":
                                stats[code][group_key]['failed'] += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    print("\n=== Exporting Statistics ===")
    
    # --- Compute and Export Group Statistics ---
    excel_output_path = metrics_folder_path / "failure_rates_metrics_stas.xlsx"
    print(f"Exporting group statistics to: {excel_output_path}")

    for code, config in AUDIT_CONFIG.items():
        sheet_name = config["sheet_name"]
        headers = config["headers"]
        
        rows = []
        for (mas, dataset, llm), counts in stats[code].items():
            total = counts['total']
            failed = counts['failed']
            failure_rate = (failed / total * 100) if total > 0 else 0.0
            
            row = [
                mas,
                dataset,
                llm,
                total,
                failed,
                f"{failure_rate:.2f}%"
            ]
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=headers)
        
        with pd.ExcelWriter(excel_output_path, mode='a' if excel_output_path.exists() else 'w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Exported statistics for {code} to sheet: {sheet_name}")
        


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
    
    stas_of_audit_results(audit_result_path, metrics_folder_path)