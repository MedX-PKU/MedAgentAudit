'''
./scripts/extract_logs_for_audit_human_eval.py
This script is designed to extract and preprocess logs from the MAS collaboration results (audit part) for audit human evaluation.
and we need to change the figure's path for online human evaluation.
'''
import random
random.seed(42)
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))
from medagentaudit.utils.json_utils import save_jsonl, load_jsonl
from medagentaudit.utils.logger import DualLogger

# Define paths
MAS_COLLABORATION_AUDIT_DIR = project_root / "logs" / "audit_results" / "20260212"
EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR = project_root / "logs" / "extracted_logs_for_audit_human_evaluation"
EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR.mkdir(parents=True, exist_ok=True)
def main():
    # read all the jsonl files in the mas collaboration results, 
    input_dir = MAS_COLLABORATION_AUDIT_DIR
    all_jsonl_files = list(input_dir.glob("*.jsonl"))
    print(f"Found {len(all_jsonl_files)} JSONL files in {input_dir}")

    terminal_log_file = EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / f"extract_log_files_for_audit_human_eval_terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    failure_mode_status_key_mapping = {
        "factual_hallucination_status": "1.1.1",
        "modality_neglect_status": "1.2.1",
        "role_task_alignment": "2.1.1",
        "knowledge_activation_status": "2.1.2",
        "interaction_redundancy": "2.2.1",
        "conflict_resolution_status": "2.2.2",
        "suppression_status": "3.1.1",
        "authority_bias_status": "3.1.2",
        "neglect_of_conflict_status": "3.1.3",
        "inter_round_consistency_status": "3.2.1"
    }
    failure_code_positive_sample_file_mapping = {
        "1.1.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "1.1.1_positive_sample.jsonl",
        "1.2.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "1.2.1_positive_sample.jsonl",
        "2.1.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "2.1.1_positive_sample.jsonl",
        "2.1.2": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "2.1.2_positive_sample.jsonl",
        "2.2.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "2.2.1_positive_sample.jsonl",
        "2.2.2": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "2.2.2_positive_sample.jsonl",
        "3.1.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "3.1.1_positive_sample.jsonl",
        "3.1.2": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "3.1.2_positive_sample.jsonl",
        "3.1.3": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "3.1.3_positive_sample.jsonl",
        "3.2.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "3.2.1_positive_sample.jsonl"
    }
    failure_code_negative_sample_file_mapping = {
        "1.1.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "1.1.1_negative_sample.jsonl",
        "1.2.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "1.2.1_negative_sample.jsonl",
        "2.1.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "2.1.1_negative_sample.jsonl",
        "2.1.2": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "2.1.2_negative_sample.jsonl",
        "2.2.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "2.2.1_negative_sample.jsonl",
        "2.2.2": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "2.2.2_negative_sample.jsonl",
        "3.1.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "3.1.1_negative_sample.jsonl",
        "3.1.2": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "3.1.2_negative_sample.jsonl",
        "3.1.3": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "3.1.3_negative_sample.jsonl",
        "3.2.1": EXTRACTED_FOR_AUDIT_HUMAN_eval_LOG_DIR / "3.2.1_negative_sample.jsonl"
    }
    # we create a temp pool for each failure mode, if the status key exists in the json record, we save it to the corresponding pool according to its value (positive or negative)
    # for as to following random sampling.
    failure_mode_json_file_positive_sample_pool = {
        "1.1.1": [],
        "1.2.1": [],
        "2.1.1": [],
        "2.1.2": [],
        "2.2.1": [],
        "2.2.2": [],
        "3.1.1": [],
        "3.1.2": [],
        "3.1.3": [],
        "3.2.1": []
    }
    failure_mode_json_file_negative_sample_pool = {
        "1.1.1": [],
        "1.2.1": [],
        "2.1.1": [],
        "2.1.2": [],
        "2.2.1": [],
        "2.2.2": [],
        "3.1.1": [],
        "3.1.2": [],
        "3.1.3": [],
        "3.2.1": []
    }
    failure_mode_log_key_mapping = {
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
    for jsonl_file in all_jsonl_files:
        print(f"Processing file: {jsonl_file}")
        jsonl_file_name = jsonl_file.stem
        mas = jsonl_file_name.split("_")[0]
        dataset = jsonl_file_name.split("_")[1]
        llm = jsonl_file_name.split("_")[2]
        data = load_jsonl(jsonl_file)
        print(f"  - Total records: {len(data)}")
        # for every failure mode, we extract the corresponding status key and check if it exists in the json record, if exists, 
        # we save it to a new jsonl file which is named by the failure mode, but we add the dataset, llm and mas info in the json record for later analysis.
        for json_record in data:
            json_record["dataset"] = dataset
            json_record["llm"] = llm
            json_record["mas"] = mas
            for failure_status, failure_code in failure_mode_status_key_mapping.items():
                # we check if the status key exists in the json record
                audit = json_record["case_history"].get("audit", None)
                if not audit:
                    continue
                rounds = audit["rounds"]
                if_failure_status_exists = False
                for round in rounds:
                    failure_log_key = failure_mode_log_key_mapping[failure_code]
                    if failure_log_key in round.keys() and round[failure_log_key] and round[failure_log_key][0]["audit_result"] and failure_status in round[failure_log_key][0]["audit_result"].keys():
                        if_failure_status_exists = True
                        failure_status_value = round[failure_log_key][0]["audit_result"][failure_status]
                        break
                if if_failure_status_exists: 
                    # if the key exists, we check the first value of it, if the value is 1, label it as the positive ,else label it negative. 
                    # at the same time ,we need to add the dataset, llm and mas info in the json record for later analysis.
                    if failure_status_value == "1":
                        failure_mode_json_file_positive_sample_pool[failure_code].append(json_record)
                    else:
                        failure_mode_json_file_negative_sample_pool[failure_code].append(json_record)
    # after processing all the json records in the jsonl file, we do random sampling for each failure mode, we randomly select 20 positive samples and 20 negative samples for each failure mode for audit human evaluation, 
    # if the total number of samples for a failure mode is less than 40， we select as more as we can for that failure mode.
    audit_for_human_eval_positive_sample_size = 20
    audit_for_human_eval_negative_sample_size = 20
    for failure_code in failure_mode_status_key_mapping.values():
        positive_samples = failure_mode_json_file_positive_sample_pool[failure_code]
        negative_samples = failure_mode_json_file_negative_sample_pool[failure_code]
        positive_needed = audit_for_human_eval_positive_sample_size
        negative_needed = audit_for_human_eval_negative_sample_size
        print(f"Failure code: {failure_code} - Total available samples: {len(positive_samples) + len(negative_samples)} (Positive: {len(positive_samples)}, Negative: {len(negative_samples)})")

        if len(positive_samples) + len(negative_samples) == 0:
            print(f"  - No samples available for failure code {failure_code}. Skipping.")
            continue
        # we get the positive and negative samples respectively,we process positive first.
        if len(positive_samples) < positive_needed:
            print(f"  [!] Warning: Only {len(positive_samples)} positive samples available for failure code {failure_code} (less than {positive_needed} required). Taking all available positive samples.")
            selected_positive_samples = positive_samples
        else:
            selected_positive_samples = random.sample(positive_samples, positive_needed)
        # then we process negative samples
        if len(negative_samples) < negative_needed:
            print(f"  [!] Warning: Only {len(negative_samples)} negative samples available for failure code {failure_code} (less than {negative_needed} required). Taking all available negative samples.")
            selected_negative_samples = negative_samples
        else:
            selected_negative_samples = random.sample(negative_samples, negative_needed)
        # we save the selected samples to the corresponding jsonl file according to their failure code and their label (positive or negative)
        positive_sample_output_file = failure_code_positive_sample_file_mapping[failure_code]
        for json_record in selected_positive_samples:
            # we change the image path for online human evaluation
            image = json_record.get("image_path", "")
            if image and Path(image).exists():
                target = "datasets"
                new_image_path = target + image.split(target)[1]
                json_record["image_path"] = new_image_path
            save_jsonl(json_record, positive_sample_output_file)
        print(f"  - Positive samples for failure code {failure_code} saved to: {positive_sample_output_file}")
        negative_sample_output_file = failure_code_negative_sample_file_mapping[failure_code]
        for json_record in selected_negative_samples:
            # we change the image path for online human evaluation
            image = json_record.get("image_path", "")
            if image and Path(image).exists():
                target = "datasets"
                new_image_path = target + image.split(target)[1]
                json_record["image_path"] = new_image_path
            save_jsonl(json_record, negative_sample_output_file)
        print(f"  - Negative samples for failure code {failure_code} saved to: {negative_sample_output_file}")
        
if __name__ == "__main__":
    main()
