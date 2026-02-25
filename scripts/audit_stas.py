'''
./scripts/audit_stas.py
this script is to compute the statistics for audited results from multi-agent systems,the granularity of statistics is at the level of each failure mode and every round.
'''
# TODO : aggregate the audit reports at the granularity of rounds: for example the round1 failure rate, round2 failure rate, etc so as to we can visualize it!
import json
from typing import Tuple
from tqdm import tqdm
import sys
from pathlib import Path
import pandas as pd
from medagentaudit.utils.logger import DualLogger
from collections import defaultdict

# Ensure project root is in path
current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
project_root = current_file_path.parents[1]

# Configuration for Audit Failure Modes
AUDIT_CONFIG = {
    "1.1.1": {
        "log_key": "1_1_1_factual_hallucination",
        "status_key": "factual_hallucination_status",
        "name": "Factual Hallucinations during Input Interpretation"
    },
    "1.2.1": {
        "log_key": "1_2_1_neglect_or_misinterpretation_of_modality_info", 
        "status_key": "modality_neglect_status",
        "name": "Neglect or Misinterpretation of Modality Information during Input Interpretation"
    },
    "2.1.1": {
        "log_key": "2_1_1_role_assignment", 
        "status_key": "role_task_alignment",
        "name": "Mismatch Between Assigned Roles and Clinical Tasks during Collaborative discussion"
    },
    "2.1.2": {
        "log_key": "2_1_2_domain_specific_knowledge_activation",
        "status_key": "knowledge_activation_status",
        "name": "Failure to Activate Specialist Knowledge During Role Execution during Collaborative discussion"
    },
    "2.2.1": {
        "log_key": "2_2_1_repetition_of_initial_views", 
        "status_key": "interaction_redundancy",
        "name": "Repetition of Initial Views during Collaborative discussion"
    },
    "2.2.2": {
        "log_key": "2_2_2_unresolved_conflicts",
        "status_key": "conflict_resolution_status",
        "name": "Unresolved Conflicts during Collaborative discussion"
    },
    "3.1.1": {
        "log_key": "3_1_1_suppression_of_minority_views", 
        "status_key": "suppression_status",
        "name": "Suppression of Correct Minority Views by Incorrect Consensus during Decision-making"
    },
    "3.1.2": {
        "log_key": "3_1_2_authority_bias", 
        "status_key": "authority_bias_status",
        "name": "Reasoning Distorted by Authority Bias during Decision-making"
    },
    "3.1.3": {
        "log_key": "3_1_3_neglect_of_contradictions", 
        "status_key": "neglect_of_conflict_status",
        "name": "Neglect of Contradictions in Reasoning Process during Decision-making"
    },
    "3.2.1": {
        "log_key": "3_2_1_self_contradiction_when_decision", 
        "status_key": "inter_round_consistency_status",
        "name": "Self-Contradiction in Viewpoints Across Rounds during Decision-making"
    }
}

def extract_metadata_from_path(file_path: str) -> Tuple[str, str, str]:
    """
    Extracts MAS, Dataset, and LLM from the file path.
    """
    # Based on your description, specific list of LLMs involved
    lower_path = file_path.lower()
    parts = lower_path.split('_') # Assume standard separator or handled by pathlib conversion before
    current_mas = parts[0]
    current_dataset = parts[1]
    current_llm = parts[2]
            
    return current_mas, current_dataset, current_llm

def process_audit_files(audit_results_path: Path) -> defaultdict:
    """
    Walks through JSONL files and aggregates statistics.
    """
    if not audit_results_path.exists():
        raise FileNotFoundError(f"The path {audit_results_path} does not exist.")
    
    jsonl_files = list(audit_results_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} audit result files (jsonl) in {audit_results_path}.")

    aggregated_stats = defaultdict(lambda: {'total': 0, 'failed': 0})

    for jsonl_file in tqdm(jsonl_files, desc="Processing audit files"):
        file_name = jsonl_file.stem
        
        # 1. Identify Metadata
        mas, dataset, llm = extract_metadata_from_path(file_name)
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                json_data = json.loads(line)
                case_history = json_data["case_history"]
                audit_data = case_history.get("audit")
                
                if not audit_data:
                    continue
                
                rounds = audit_data.get("rounds", [])
                if not rounds:
                    continue

                # 3. Count Failures per Round
                for audit_round_data in rounds:
                    for code, config in AUDIT_CONFIG.items():
                        log_key = config["log_key"]
                        status_key = config["status_key"]
                        
                        entries = audit_round_data.get(log_key, [])
                        
                        # Some logs store single dict, some store list of dicts. Handle both.
                        if isinstance(entries, dict):
                            entries = [entries]
                        
                        if entries and isinstance(entries, list):
                            for entry in entries:
                                aggregated_stats[(code,mas,dataset)]['total'] += 1
                                result_obj = entry.get("audit_result", {})
                                # Check failure status ("1" is failure)
                                if str(result_obj.get(status_key)) == "1":
                                    aggregated_stats[(code,mas,dataset)]['failed'] += 1

    return aggregated_stats

def export_statistics(aggregated_stats: list, metrics_folder_path: Path):
    """
    Exports statistics to CSV files
    """
    if not metrics_folder_path.exists():
        metrics_folder_path.mkdir(parents=True, exist_ok=True)

    print("\n=== Exporting Statistics to CSV ===")
    csv_rows = []
    for (code, mas, dataset), stats in aggregated_stats:
        # Prepare Rows for CSV
        total = stats['total']
        failed = stats['failed']
        failure_rate = (failed / total * 100) if total > 0 else 0.0
        row = {
            "Failure_Mode_ID": code,
            "Framework": mas,
            "Dataset": dataset,
            "Failure_Mode_Name": AUDIT_CONFIG[code]["name"],
            "Total_Audited_Count": total,
            "Failure_Count": failed,
            "Failure_Rate(%)": f"{failure_rate:.2f}"
        }
        csv_rows.append(row)
        
    if not csv_rows:
        print(f"No data populated for Framework: {mas}")
    # Convert to DataFrame
    df = pd.DataFrame(csv_rows)
    
    # Define filename: ColaCare_stas.csv
    # Capitalize first letter of MAS for filename aesthetics if needed, or keep raw
    filename = "audit_results_stas.csv"
    output_file = metrics_folder_path / filename
            
    df.to_csv(output_file, index=False)
    print(f"Exported [{mas}] statistics to: {output_file}")


def stas_of_audit_results(audit_results_path: Path, metrics_folder_path: Path):
    """
    Main function to drive the statistical analysis.
    """
    # 1. Process all files and aggregate data in memory
    stats = process_audit_files(audit_results_path)
    sorted_stats = sorted(stats.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))

    # 2. Export separated CSVs per MAS
    export_statistics(sorted_stats, metrics_folder_path)


if __name__ == "__main__":
    # Define paths
    audit_result_path = project_root / "logs" / "audit_results" / "20260225"
    metrics_folder_path = project_root / "logs" / "metrics"
    
    # Setup Logging
    terminal_log_dir = metrics_folder_path / "terminal_log"
    terminal_log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_file = terminal_log_dir / "audit_stats.log"
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    # Redirect stdout/stderr
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    try:
        stas_of_audit_results(audit_result_path, metrics_folder_path)
        print("Statistics calculation completed successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()