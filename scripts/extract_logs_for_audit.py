'''
./scripts/extract_logs_for_audit.py
This script is designed to extract and preprocess logs from the MAS collaboration results for audit human evaluation.
'''
import random
import argparse
from typing import List, Dict, Any, Iterable, Tuple
import pandas as pd
from pathlib import Path
import sys
import re
import json
from medagentaudit.utils.json_utils import save_jsonl, load_jsonl
from medagentaudit.utils.logger import DualLogger
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]

# Define paths
MAS_COLLABORATION_DIR = project_root / "logs" / "mas_collaboration_results" / "20260202"
EXTRACTED_FOR_OPENCODING_LOG_DIR = project_root / "logs" / "extracted_logs_for_open_coding"
EXTRACTED_FOR_OPENCODING_HUMAN_EVL_LOG_DIR = project_root / "logs" / "extracted_logs_for_open_coding_human_evaluation"
EXTRACTED_FOR_OPENCODING_LOG_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_FOR_OPENCODING_HUMAN_EVL_LOG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # read all the jsonl files in the mas collaboration results, 
    input_dir = MAS_COLLABORATION_DIR
    all_json_files = list(input_dir.glob("*.jsonl"))
    print(f"Found {len(all_json_files)} JSONL files in {input_dir}")

    terminal_log_file = EXTRACTED_FOR_OPENCODING_LOG_DIR / f"extract_log_files_for_opencoding_terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    for jsonl_file in all_json_files:
        print(f"Processing file: {jsonl_file}")
        data = load_jsonl(jsonl_file)
        print(f"  - Total records: {len(data)}")

        open_coding_size = 20 ; open_coding_for_human_evaluation_size = 10
        # randomly shuffle the data and select 20 json records for open coding and 10 other records for open coding human evaluation
        total_needed = open_coding_size + open_coding_for_human_evaluation_size
        if total_needed > len(data):
            raise ValueError(f"the total data num requested ({total_needed})exceeds the total amount of avaliable data: ({len(data)}) in file: {jsonl_file}")
        random.seed(42)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        open_coding_data = shuffled_data[:open_coding_size]
        open_coding_human_evl_data = shuffled_data[open_coding_size:open_coding_size + open_coding_for_human_evaluation_size]
        # save the two subsets to separate jsonl files
        output_open_coding_file = EXTRACTED_FOR_OPENCODING_LOG_DIR / f"{jsonl_file.stem}_open_coding.jsonl"
        # write json records one record one line to the jsonl file
        for json_record in open_coding_data:
            save_jsonl(json_record, output_open_coding_file)
        print(f"  - Open coding subset saved to: {output_open_coding_file}")
        output_open_coding_human_evl_file = EXTRACTED_FOR_OPENCODING_HUMAN_EVL_LOG_DIR / f"{jsonl_file.stem}_open_coding_human_evl.jsonl"
        # write json records one record one line to the jsonl file
        for json_record in open_coding_human_evl_data:
            save_jsonl(json_record, output_open_coding_human_evl_file)
        print(f"  - Extracted data saved to: {output_open_coding_human_evl_file}")
        
if __name__ == "__main__":
    main()