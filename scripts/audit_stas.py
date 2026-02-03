'''
./scripts/audit_stas.py
this script is to compute the statistics for audited results from multi-agent systems
we need to give statistics 3 dimensions: mas, dataset, llm
'''
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

# Configuration for Audit Failure Modes
AUDIT_CONFIG = {
    "1.1.1": {
        "log_key": "1_1_1_factual_hallucination",
        "status_key": "factual_hallucination_status",
        "name": "Factual Hallucination"
    },
    "1.2.1": {
        "log_key": "1_2_1_neglect_or_misinterpretation_of_modality_info", 
        "status_key": "modality_neglect_status",
        "name": "Modality Neglect"
    },
    "2.1.1": {
        "log_key": "2_1_1_role_assignment", 
        "status_key": "role_task_alignment",
        "name": "Role Assignment Mismatch"
    },
    "2.1.2": {
        "log_key": "2_1_2_domain_specific_knowledge_activation",
        "status_key": "knowledge_activation_status",
        "name": "Knowledge Activation Fail"
    },
    "2.2.1": {
        "log_key": "2_2_1_repetition_of_initial_views", 
        "status_key": "interaction_redundancy",
        "name": "Interaction Redundancy"
    },
    "2.2.2": {
        "log_key": "2_2_2_unresolved_conflicts",
        "status_key": "conflict_resolution_status",
        "name": "Unresolved Conflicts"
    },
    "3.1.1": {
        "log_key": "3_1_1_suppression_of_minority_views", 
        "status_key": "suppression_status",
        "name": "Minority Suppression"
    },
    "3.1.2": {
        "log_key": "3_1_2_authority_bias", 
        "status_key": "authority_bias_status",
        "name": "Authority Bias"
    },
    "3.1.3": {
        "log_key": "3_1_3_neglect_of_contradictions", 
        "status_key": "neglect_of_conflict_status",
        "name": "Neglect of Contradictions"
    },
    "3.2.1": {
        "log_key": "3_2_1_self_contradiction_when_decision", 
        "status_key": "inter_round_consistency_status",
        "name": "Inter-round Inconsistency"
    }
}

def extract_metadata_from_path(file_path: str) -> Tuple[str, str, str]:
    """
    Extracts MAS, Dataset, and LLM from the file path.
    """
    mas_list = ['colacare', 'healthcareagent', 'mac', 'mdagents', 'medagent', 'reconcile']
    datasets_list = ['MedQA', 'MedXpertQA-text', 'PubMedQA', 'PathVQA', 'SLAKE', 'VQA-RAD']
    datasets_sorted = sorted(datasets_list, key=len, reverse=True)

    # Based on your description, specific list of LLMs involved
    llm_models = [
        "gpt-5.2",
        "deepseek-reasoner",
        "glm-4.6v",
        "gemini-3-flash-preview",
        "qwen3-vl-8b-thinking",
        "qwen3-8b"
    ]
    
    lower_path = file_path.lower()
    parts = file_path.split('/') # Assume standard separator or handled by pathlib conversion before

    # Identify MAS (Framework)
    current_mas = "Unknown"
    for m in mas_list:
        if m in lower_path:
            # Try to match the exact casing from the list if possible, or formatted
            current_mas = m 
            break
            
    current_dataset = "Unknown"
    for d in datasets_sorted: # this is to let the PubMedQA in the front, so as to avoid being misindentified as MedQA
        if d in file_path: # Case sensitive search might be better for Datasets
            current_dataset = d
            break
            
    # Identify LLM
    current_llm = "Unknown"
    for l in llm_models:
        if l in file_path:
            current_llm = l
            break
            
    return current_mas, current_dataset, current_llm

def process_audit_files(audit_results_path: Path) -> Dict[str, Any]:
    """
    Walks through JSONL files and aggregates statistics.
    Structure: results[mas][(dataset, llm)][audit_code] = {'total': 0, 'failed': 0, 'failure rate'}
    """
    if not audit_results_path.exists():
        raise FileNotFoundError(f"The path {audit_results_path} does not exist.")
    
    jsonl_files = list(audit_results_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} audit result files (jsonl) in {audit_results_path}.")

    # Aggregation structure
    # Key: MAS Name
    # Value: Dict -> Key: (Dataset, LLM) -> Value: Dict -> Key: Audit Code -> Value: {total, failed}
    aggregated_stats = {}

    for jsonl_file in tqdm(jsonl_files, desc="Processing audit files"):
        file_path_str = str(jsonl_file)
        
        # 1. Identify Metadata
        mas, dataset, llm = extract_metadata_from_path(file_path_str)
        
        # Normalize MAS name for dictionary keys (e.g. capitalize for filename later if needed, but keeping raw for now)
        if mas not in aggregated_stats:
            aggregated_stats[mas] = {}
        
        group_key = (dataset, llm)
        if group_key not in aggregated_stats[mas]:
            aggregated_stats[mas][group_key] = {
                code: {'total': 0, 'failed': 0} for code in AUDIT_CONFIG.keys()
            }

        # 2. Process JSONL Content (Line by Line)
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        case_data = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to decode JSON at line {line_num} in {jsonl_file.name}")
                        continue

                    # Extract audit data
                    audit_data = None
                    
                    # Strategy 1: Top level audit
                    if "audit" in case_data and isinstance(case_data["audit"], dict):
                        audit_data = case_data["audit"]
                    
                    # Strategy 2: Inside case_history (if dict)
                    elif "case_history" in case_data:
                        ch = case_data["case_history"]
                        if isinstance(ch, dict):
                            audit_data = ch.get("audit")
                    
                    if not audit_data:
                        # If no audit data found for this case, skip
                        continue
                    
                    rounds = audit_data.get("rounds", [])
                    if not rounds:
                        continue

                    # 3. Count Failures per Round
                    for audit_round in rounds:
                        for code, config in AUDIT_CONFIG.items():
                            log_key = config["log_key"]
                            status_key = config["status_key"]
                            
                            entries = audit_round.get(log_key, [])
                            
                            # Some logs store single dict, some store list of dicts. Handle both.
                            if isinstance(entries, dict):
                                entries = [entries]
                            
                            if entries and isinstance(entries, list):
                                for entry in entries:
                                    aggregated_stats[mas][group_key][code]['total'] += 1
                                    
                                    result_obj = entry.get("audit_result", {})
                                    # Check failure status ("1" is failure)
                                    if str(result_obj.get(status_key)) == "1":
                                        aggregated_stats[mas][group_key][code]['failed'] += 1

        except Exception as e:
            print(f"Error reading file {jsonl_file}: {e}")
            continue
            
    return aggregated_stats

def export_statistics(aggregated_stats: Dict[str, Any], metrics_folder_path: Path):
    """
    Exports statistics to CSV files, one per MAS Framework.
    """
    if not metrics_folder_path.exists():
        metrics_folder_path.mkdir(parents=True, exist_ok=True)

    print("\n=== Exporting Statistics to CSV ===")

    for mas, dataset_llm_data in aggregated_stats.items():
        if mas == "Unknown":
            continue # Skip unidentified files
            
        # Prepare Rows for CSV
        csv_rows = []
        
        for (dataset, llm), codes_data in dataset_llm_data.items():
            for code, stats in codes_data.items():
                total = stats['total']
                failed = stats['failed']
                rate = (failed / total * 100) if total > 0 else 0.0
                
                row = {
                    "Framework": mas,
                    "Dataset": dataset,
                    "LLM": llm,
                    "Failure_Mode_ID": code,
                    "Failure_Mode_Name": AUDIT_CONFIG[code]["name"],
                    "Total_Audited_Count": total,
                    "Failure_Count": failed,
                    "Failure_Rate(%)": f"{rate:.2f}"
                }
                csv_rows.append(row)
        
        if not csv_rows:
            print(f"No data populated for Framework: {mas}")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(csv_rows)
        
        # Define filename: ColaCare_stas.csv
        # Capitalize first letter of MAS for filename aesthetics if needed, or keep raw
        filename = f"{mas}_stas.csv"
        output_file = metrics_folder_path / filename
        
        # Sort for better readability
        # Sort by Dataset, then LLM, then Failure Mode ID
        df = df.sort_values(by=["Dataset", "LLM", "Failure_Mode_ID"])
        
        df.to_csv(output_file, index=False)
        print(f"Exported [{mas}] statistics to: {output_file}")


def stas_of_audit_results(audit_results_path: Path, metrics_folder_path: Path):
    """
    Main function to drive the statistical analysis.
    """
    # 1. Process all files and aggregate data in memory
    stats = process_audit_files(audit_results_path)
    
    # 2. Export separated CSVs per MAS
    export_statistics(stats, metrics_folder_path)


if __name__ == "__main__":
    timestamp = "20260202" 
    
    # Define paths
    audit_result_path = project_root / "logs" / "audit_results" / timestamp
    metrics_folder_path = project_root / "logs" / "metrics" 
    
    # Setup Logging
    terminal_log_dir = metrics_folder_path / timestamp / "terminal_log"
    terminal_log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_file = terminal_log_dir / f"{timestamp}_audit_stats.log"
    
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    
    # Redirect stdout/stderr
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    print(f"Starting Audit Statistics Calculation for timestamp: {timestamp}")
    
    try:
        stas_of_audit_results(audit_result_path, metrics_folder_path)
        print("Statistics calculation completed successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()