'''
./scripts/extract_logs_for_audit_human_eval.py
This script is designed to extract and preprocess logs from the MAS collaboration results (audit part) for audit human evaluation.
and we need to change the figure's path for online human evaluation.
'''
import random
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))
from medagentaudit.utils.json_utils import save_jsonl, load_jsonl
from medagentaudit.utils.logger import DualLogger

# Define paths
MAS_COLLABORATION_AUDIT_DIR = project_root / "logs" / "audit_results" / "20260202"
EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR = project_root / "logs" / "extracted_logs_for_audit_human_evaluation"
EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR.mkdir(parents=True, exist_ok=True)
def main():
    # read all the jsonl files in the mas collaboration results, 
    input_dir = MAS_COLLABORATION_AUDIT_DIR
    all_json_files = list(input_dir.glob("*.jsonl"))
    print(f"Found {len(all_json_files)} JSONL files in {input_dir}")

    terminal_log_file = EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / f"extract_log_files_for_audit_human_eval_terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    failure_mode_status_key_mapping = {
        "1.1.1": "factual_hallucination_status",
        "1.2.1": "modality_neglect_status",
        "2.1.1": "role_task_alignment",
        "2.1.2": "knowledge_activation_status",
        "2.2.1": "interaction_redundancy",
        "2.2.2": "conflict_resolution_status",
        "3.1.1": "suppression_status",
        "3.1.2": "authority_bias_status",
        "3.1.3": "neglect_of_conflict_status",
        "3.2.1": "inter_round_consistency_status"
    }

    for jsonl_file in all_json_files:
        print(f"Processing file: {jsonl_file}")
        data = load_jsonl(jsonl_file)
        print(f"  - Total records: {len(data)}")

        audit_for_human_evaluation_positive_sample_size = 20
        audit_for_human_evaluation_negative_sample_size = 20
        # randomly shuffle the data and select 20 json records for open coding and 10 other records for open coding human evaluation
        if total_needed > len(data):
            raise ValueError(f"the total data num requested ({total_needed})exceeds the total amount of avaliable data: ({len(data)}) in file: {jsonl_file}")
        random.seed(42)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        open_coding_data = shuffled_data[:open_coding_size]
        open_coding_human_eval_data = shuffled_data[open_coding_size:open_coding_size + open_coding_for_human_evaluation_size]
        # save the two subsets to separate jsonl files
        output_open_coding_file = EXTRACTED_FOR_OPENCODING_LOG_DIR / f"{jsonl_file.stem}_open_coding.jsonl"
        # write json records one record one line to the jsonl file
        for json_record in open_coding_data:
            save_jsonl(json_record, output_open_coding_file)
        print(f"  - Open coding subset saved to: {output_open_coding_file}")
        output_open_coding_human_eval_file = EXTRACTED_FOR_OPENCODING_HUMAN_eval_LOG_DIR / f"{jsonl_file.stem}_open_coding_human_eval.jsonl"
        # write json records one record one line to the jsonl file
        for json_record in open_coding_human_eval_data:
            save_jsonl(json_record, output_open_coding_human_eval_file)
        print(f"  - Extracted data saved to: {output_open_coding_human_eval_file}")
        
if __name__ == "__main__":
    main()