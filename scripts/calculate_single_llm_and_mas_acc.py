'''
this script is in Phase 1. It calculates the accuracy of Single llm and MAS, the granularity is at the dataset-LLM level.
'''
from pathlib import Path
from collections import defaultdict
from medagentaudit.utils.json_utils import load_jsonl
from medagentaudit.utils.logger import DualLogger
import sys
import csv
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
single_llm_dir = project_root / "logs" / "single_llm" / "20260202"
mas_dir = project_root / "logs" / "mas_collaboration_results_audit"
output_dir = project_root / "logs" / "accuracy_stats"
output_single_llm_file = output_dir / "single_llm_accuracy_stats.csv"
output_mas_file = output_dir / "mas_accuracy_stats.csv" 
def get_metadata(file_path_str, object = str):
    """retrieve MAS, Dataset, LLM from the file path string"""
    datasets_list = ['MedQA', 'MedXpertQA-text', 'PubMedQA', 'PathVQA', 'SLAKE', 'VQA-RAD']
    # avoid MedQA matching MedXpertQA-text by sorting in descending order of length
    datasets_sorted = sorted(datasets_list, key=len, reverse=True)
    
    llm_keywords = ["gpt", "deepseek", "glm", "gemini", "qwen"]
    mas_list = ['colacare', 'healthcareagent', 'mac', 'mdagents', 'medagent', 'reconcile']

    path_lower = file_path_str.lower()
            
    # 2. identify Dataset
    found_dataset = "Unknown"
    for d in datasets_sorted:
        if d.lower() in path_lower:
            found_dataset = d
            break
            
    # 3. identify LLM
    found_llm = "Unknown"
    parts = Path(file_path_str).parts
    for part in parts:
        part_l = part.lower()
        if any(k in part_l for k in llm_keywords):
            found_llm = part
            break
    if object == "mas":
        # 1. identify MAS
        found_mas = "Unknown"
        for m in mas_list:
            if m in path_lower:
                found_mas = m
                break
    
        return found_dataset, found_llm, found_mas
    else:
        return found_dataset, found_llm
    
terminal_log_file = output_dir / "accuracy_calculation.log"
terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
sys.stdout = DualLogger(terminal_log_file, sys.stdout)
sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条

object_1 = "single_llm"
object_2 = "mas"
# calculate the accuracy for single llm at the granularity of dataset-LLM
all_jsonl_files_single_llm = list(single_llm_dir.glob("*.jsonl"))
all_jsonl_files_mas = list(mas_dir.glob("*.jsonl"))
stats_single_llm = defaultdict(int)
stats_mas = defaultdict(int)
for jsonl_file in all_jsonl_files_single_llm:
    print(f"Processing file: {jsonl_file}")
    dataset, llm = get_metadata(str(jsonl_file), object_1)
    print(f"Dataset: {dataset}, LLM: {llm}")
    # calculate acc
    case_num = 0
    correct_num = 0
    data = load_jsonl(jsonl_file)
    for json_record in data:
        case_num += 1
        if json_record["ground_truth"].lower() == json_record["predicted_answer"].lower():
            correct_num += 1
    acc = correct_num / case_num if case_num > 0 else 0
    print(f"Accuracy: {acc:.4f} ({correct_num}/{case_num})")
    stats_single_llm[(dataset, llm, case_num, correct_num)] = acc
print("all single llm files processed. Now processing MAS files...")
for jsonl_file in all_jsonl_files_mas:
    print(f"Processing file: {jsonl_file}")
    dataset, llm, mas = get_metadata(str(jsonl_file), object_2)
    print(f"Dataset: {dataset}, LLM: {llm}, MAS: {mas}")
    # calculate acc
    case_num = 0
    correct_num = 0
    data = load_jsonl(jsonl_file)
    for json_record in data:
        case_num += 1
        if json_record["ground_truth"].lower() == json_record["predicted_answer"].lower():
            correct_num += 1
    acc = correct_num / case_num if case_num > 0 else 0
    print(f"Accuracy: {acc:.4f} ({correct_num}/{case_num})")
    stats_mas[(dataset, llm, mas, case_num, correct_num)] = acc
print("all MAS files processed. Now writing results to CSV...")

# output csv for single llm
with open(output_single_llm_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "LLM", "Case_Num", "Correct_Num", "Accuracy"])
    for (dataset, llm, case_num, correct_num), acc in stats_single_llm.items():
        writer.writerow([dataset, llm, case_num, correct_num, f"{acc:.4f}"])
print(f"Single LLM accuracy stats written to {output_single_llm_file}")

# output csv for MAS
with open(output_mas_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "LLM", "MAS", "Case_Num", "Correct_Num", "Accuracy"])
    for (dataset, llm, mas, case_num, correct_num), acc in stats_mas.items():
        writer.writerow([dataset, llm, mas, case_num, correct_num, f"{acc:.4f}"])
print(f"MAS accuracy stats written to {output_mas_file}")