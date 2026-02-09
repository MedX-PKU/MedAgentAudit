import os
import random
import argparse
from typing import List, Dict, Any, Iterable, Tuple
import pandas as pd
from pathlib import Path
import sys
import re
import json
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
utils_root = project_root / "medagentaudit" / "utils"
sys.path.append(str(utils_root))
from json_utils import save_json, load_json, load_jsonl

# Define paths
RAW_DATA_DIR = project_root / "datasets" / "raw"
PROCESSED_DATA_DIR = project_root / "datasets" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def convert_to_mc_format(closed_questions: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """Convert CLOSED type questions to medqa_mc_test format"""
    mc_format_data = []
    
    for i, item in enumerate(closed_questions):
        # Build image path
        img_name = item.get('img_name', '')
        Slake_image_path = RAW_DATA_DIR / "SLAKE" / "images"
        image_path = Slake_image_path / img_name
        
        # Build qid
        qid = f"slake_{split_name}_{item.get('qid', i)}"
        
        # Get correct answer
        correct_answer = item.get('answer', '')
        
        # Generate options for multiple choice questions (simplified processing, may need more complex logic)
        # Common answers are Yes/No, or other options
        if correct_answer.lower() in ['yes', 'no']:
            options = {
                "A": "Yes",
                "B": "No"
            }
            answer = "A" if correct_answer.lower() == "yes" else "B"
        else:
            # For other types of answers, create simple options
            options = {
                "A": correct_answer,
                "B": "Other"
            }
            answer = "A"
        
        mc_item = {
            "qid": qid,
            "question": item.get('question', ''),
            "image_path": str(image_path),
            "options": options,
            "answer": answer
        }
        
        mc_format_data.append(mc_item)
    
    return mc_format_data

def process_split(split_name: str, input_file: str):
    """Process single dataset split and return mc lists"""
    print(f"\nProcessing {split_name} dataset...")
    
    # Load original data
    data = load_json(input_file)
    if not data:
        return []
    
    # Separate question types
    closed_questions = separate_questions_by_type(data)
    
    # Convert to new format
    data = convert_to_mc_format(closed_questions, split_name)
    
    print(f"{split_name} processing completed!")
    print(f"  - CLOSED questions: {len(data)} records")
    return data

def separate_questions_by_type(data: List[Dict[str, Any]]) -> tuple:
    """Separate questions by answer_type field, only keep English data"""
    closed_questions = []
    
    for item in data:
        # Only keep English data
        if item.get('q_lang') != 'en':
            continue
        if item.get('answer_type') == 'CLOSED':
            closed_questions.append(item)
    
    print(f"English CLOSED type questions: {len(closed_questions)} records")
    
    return closed_questions

def parse_jsonl(file_path: str) -> Iterable[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error parsing {file_path} line {line_num}: {e}")

def clean_question_text(raw_question: str) -> str:
    """Remove embedded options section from question text.

    Heuristics:
    - Cut from markers like "Answer Choices:", "Choices:", "Options:" (case-insensitive)
    - Also handle missing colon like "Answer Choices (A) ..."
    """
    if not raw_question:
        return ""
    # match the starting index of the answer choices section
    marker_with_colon = re.search(r"(?is)(?:^|\n)\s*(answer\s*choices?|choices?|options?)\s*:\s*", raw_question)
    if marker_with_colon:
        return raw_question[:marker_with_colon.start()].rstrip()

    # match the answer choices which starts with answer choices ()
    marker_without_colon = re.search(r"(?is)(?:^|\n)\s*(answer\s*choices?|choices?|options?)\s*\(", raw_question)
    if marker_without_colon:
        return raw_question[:marker_without_colon.start()].rstrip()

    return raw_question.rstrip()

def convert_items_to_medqa(
    items: Iterable[Dict[str, Any]], # a generator of dict
    start_index: int = 0,
    prefix: str = "medxpertqa_text",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    mc_records: List[Dict[str, Any]] = []
    idx = start_index

    for data in items:
        question_raw = data.get('question', '')
        question = clean_question_text(question_raw)

        options_obj = data.get('options') or data.get('choices') or {}
        # Normalize options to dict[str, str]
        options: Dict[str, str] = {}
        if isinstance(options_obj, dict):
            for k, v in options_obj.items():
                key = str(k).strip()
                options[key] = str(v)
        elif isinstance(options_obj, list):
            # If a list is provided, map to A,B,C,...
            for i, v in enumerate(options_obj):
                options[chr(ord('A') + i)] = str(v)

        label = str(data.get('label', '')).strip()

        if options and label:
            mc_records.append(
                {
                    "qid": f"{prefix}_{idx}",
                    "question": question,
                    "options": options,
                    "answer": label,
                }
            )

        idx += 1

    return mc_records, idx

def random_select_samples(data: List[Dict[str, Any]], sample_size: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Randomly select a subset of samples from the dataset.

    Returns:
        A randomly selected subset of the input data
    """
    if sample_size >= len(data):
        return data

    random.seed(seed)
    return random.sample(data, sample_size)


def process_medqa(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, audit_size: int = None, open_coding_size: int = None):
    """
    Process the MedQA dataset from raw format to standardized format.

    Args:
        raw_dir: Directory containing raw dataset
        output_dir: Directory to save processed dataset
        audit_size: Number of audit samples to select (None for all samples)
        open_coding_size: Number of open coding samples to select (None for all samples)
    """
    # Define paths
    medqa_path = raw_dir / "MedQA" / "questions" / "US" / "test.jsonl"
    audit_output_path = output_dir / "MedQA" / "audit" / "medqa_medqa_audit.json"
    open_coding_output_path = output_dir / "MedQA" / "open_coding" / "medqa_medqa_open_coding.json"

    # Create output directory if it doesn't exist
    Path(audit_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(open_coding_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load JSONL data
    medqa_data = load_jsonl(medqa_path)

    all_processed_data = []

    for i, item in enumerate(medqa_data):
        curated_data = {
            "qid": f"medqa_{str(i + 1).zfill(3)}",
            "question": item["question"],
            "options": item["options"],
            "answer": item["answer_idx"]
        }
        all_processed_data.append(curated_data)
    print(f"Total MedQA questions processed: {len(all_processed_data)}")
    # randomly shuffle the entire dataset
    total_needed = audit_size + open_coding_size
    if total_needed > len(all_processed_data):
        raise ValueError(f"the total data num requested ({total_needed})exceeds the total amount of avaliable data: ({len(all_processed_data)})")
    random.seed(42)
    # copy raw table to avoid messing up the original order
    shuffled_data = all_processed_data.copy()
    random.shuffle(shuffled_data)
    # slice the shuffled data into two non-overlapping subsets
    audit_processed_data = shuffled_data[:audit_size]
    open_coding_processed_data = shuffled_data[audit_size:audit_size + open_coding_size]

    # Save processed data
    save_json(audit_processed_data, audit_output_path)
    save_json(open_coding_processed_data, open_coding_output_path)
    print(f"MedQA audit dataset processed and saved to: {audit_output_path}")
    print(f"MedQA open coding dataset processed and saved to: {open_coding_output_path}")

def process_pubmedqa(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, audit_size: int = None, open_coding_size: int = None):
    """
    Process the PubMedQA dataset from raw format to standardized format.
    in order to randomly extracting two non-overlapping subsets from the same data source, we firstly perform a globaly random shuffle on the
    entire dataset, and then slice it according to the required quantities.

    Args:
        raw_dir: Directory containing raw dataset
        output_dir: Directory to save processed dataset
        audit_size: Number of audit samples to select (None for all samples)
        open_coding_size: Number of open coding samples to select (None for all samples)
    """
    # Define paths
    ori_pqal_path = raw_dir / "PubMedQA" / "ori_pqal.json"
    test_ground_truth_path = raw_dir / "PubMedQA" / "test_ground_truth.json"
    audit_output_path_base = output_dir / "PubMedQA" / "audit"
    open_coding_output_path_base = output_dir / "PubMedQA" / "open_coding"
    audit_output_path = audit_output_path_base / "medqa_pubmedqa_audit.json"
    open_coding_output_path = open_coding_output_path_base / "medqa_pubmedqa_open_coding.json"

    # Create output directory if it doesn't exist
    audit_output_path_base.mkdir(parents=True, exist_ok=True)
    open_coding_output_path_base.mkdir(parents=True, exist_ok=True)

    # Load datasets
    data = load_json(ori_pqal_path)
    labels = load_json(test_ground_truth_path)

    # Define standard options
    options = {"A": "Yes", "B": "No", "C": "Maybe"}
    options_map = {"Yes": "A", "No": "B", "Maybe": "C"}

    all_processed_data = []

    for qid, item_data in data.items():
        # only qid in labels are test set
        if qid not in labels:
            continue
        context = " ".join(item_data["CONTEXTS"])  # Concatenate contexts into a single string
        question = item_data["QUESTION"]
        answer = item_data["final_decision"].capitalize()

        # Multiple-choice version: Concatenate context into the question
        mc_question = (
            f"{question}\n\n"
            f"Context: {context}"
        )

        mc_data = {
            "qid": f"pubmedqa_{qid}",
            "question": mc_question,
            "options": options,
            "answer": options_map[answer]  # Map ground truth to option key
        }

        all_processed_data.append(mc_data)
    print (f"Total PubMedQA MC questions processed: {len(all_processed_data)}")
    # randomly shuffle the entire dataset
    total_needed = audit_size + open_coding_size
    if total_needed > len(all_processed_data):
        raise ValueError(f"the total data num requested ({total_needed})exceeds the total amount of avaliable data: ({len(all_processed_data)})")
    
    # fixed random seed for reproducibility
    random.seed(42)
    # copy raw table to avoid messing up the original order
    shuffled_data = all_processed_data.copy()
    random.shuffle(shuffled_data)
    # slice the shuffled data into two non-overlapping subsets
    audit_data = shuffled_data[:audit_size]
    open_coding_data = shuffled_data[audit_size:audit_size + open_coding_size]

    # Save processed data
    save_json(audit_data, audit_output_path)
    save_json(open_coding_data, open_coding_output_path)
    print(f"PubMedQA open coding dataset processed and saved to: {open_coding_output_path}")
    print(f"PubMedQA audit dataset processed and saved to: {audit_output_path}")


def process_pathvqa(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, audit_size: int = None, open_coding_size: int = None):
    """
    Process the PathVQA dataset from raw format to standardized format.
    This function expects the dataset to be in JSON format (converted from .pkl),
    to avoid using the pickle module.

    Args:
        raw_dir: Directory containing raw dataset
        output_dir: Directory to save processed dataset
        audit_size: Number of audit samples to select (None for all samples)
        open_coding_size: Number of open coding samples to select (None for all samples)
    """
    path_vqa_path = raw_dir / "PathVQA" / "qas" / "test" / "test_qa.pkl"
    path_vqa_images = raw_dir / "PathVQA" / "images" / "test"
    audit_output_path = output_dir / "PathVQA" / "audit" / "medqa_pathvqa_audit.json"
    open_coding_output_path = output_dir / "PathVQA" / "open_coding" / "medqa_pathvqa_open_coding.json"

    # Create output directory if it doesn't exist
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    open_coding_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    data = pd.read_pickle(path_vqa_path)

    processed_data = []

    # Define standard options for yes/no questions
    options = {"A": "Yes", "B": "No"}
    options_map = {"yes": "A", "no": "B"}

    for i, item in enumerate(data):
        answer = str(item["answer"]).lower().strip()

        # Only process questions with yes/no answers
        if answer not in ["yes", "no"]:
            continue

        question = item["question"]
        image_path = os.path.join(path_vqa_images, item["image"]) + ".jpg"

        curated_data = {
            "qid": f"pathvqa_{str(i + 1).zfill(6)}",
            "question": question,
            "image_path": image_path,
            "options": options,
            "answer": options_map[answer]  # Map answer to option key
        }

        processed_data.append(curated_data)
    print(f"Total PathVQA yes/no questions processed: {len(processed_data)}")
    # randomly shuffle the entire dataset
    total_needed = audit_size + open_coding_size
    if total_needed > len(processed_data):  
        raise ValueError(f"the total data num requested ({total_needed})exceeds the total amount of avaliable data: ({len(processed_data)})")
    random.seed(42)
    # copy raw table to avoid messing up the original order
    shuffled_data = processed_data.copy()
    random.shuffle(shuffled_data)
    # slice the shuffled data into two non-overlapping subsets
    audit_data = shuffled_data[:audit_size]
    open_coding_data = shuffled_data[audit_size:audit_size + open_coding_size]

    # Save processed data
    save_json(audit_data, audit_output_path)
    save_json(open_coding_data, open_coding_output_path)
    print(f"PathVQA audit dataset processed and saved to: {audit_output_path}")
    print(f"PathVQA open coding dataset processed and saved to: {open_coding_output_path}")

def process_vqa_rad(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, audit_size: int = None, open_coding_size: int = None):
    """
    Process the VQA-RAD dataset from raw format to standardized format.

    Args:
        raw_dir: Directory containing raw dataset
        output_dir: Directory to save processed dataset
        audit_size: Number of audit samples to select (None for all samples)
        open_coding_size: Number of open coding samples to select (None for all samples)
    """
    # Define paths
    vqa_rad_path = raw_dir / "VQA-RAD" / "test.json"
    vqa_rad_images = raw_dir / "VQA-RAD" / "images" / "test"
    audit_output_path_base = output_dir / "VQA-RAD" / "audit"
    open_coding_output_path_base = output_dir / "VQA-RAD" / "open_coding"
    audit_output_path = audit_output_path_base / "medqa_vqa-rad_audit.json"
    open_coding_output_path = open_coding_output_path_base / "medqa_vqa-rad_open_coding.json"

    # Create output directory if it doesn't exist
    audit_output_path_base.mkdir(parents=True, exist_ok=True)
    open_coding_output_path_base.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = load_json(vqa_rad_path)

    processed_data = []

    # Define standard options for yes/no questions
    options = {"A": "Yes", "B": "No"}
    options_map = {"yes": "A", "no": "B"}

    for i, item in enumerate(data):
        question = item["question"]
        image_path = vqa_rad_images / item["image_path"]
        answer = item["answer"]

        answer_lower = answer.lower().strip()
        if answer_lower in ["yes", "no"]:
            mc_data = {
                "qid": f"vqa_rad_{str(i).zfill(6)}",
                "question": question,
                "image_path": str(image_path),
                "options": options,
                "answer": options_map[answer_lower]
            }
            processed_data.append(mc_data)
    print(f"Total VQA-RAD yes/no questions processed: {len(processed_data)}")

    # randomly shuffle the entire dataset
    total_needed = audit_size + open_coding_size
    if total_needed > len(processed_data):
        raise ValueError(f"the total data num requested ({total_needed})exceeds the total amount of avaliable data: ({len(processed_data)})")
    random.seed(42)
    # copy raw table to avoid messing up the original order
    shuffled_data = processed_data.copy()
    random.shuffle(shuffled_data)
    # slice the shuffled data into two non-overlapping subsets
    audit_data = shuffled_data[:audit_size]
    open_coding_data = shuffled_data[audit_size:audit_size + open_coding_size]
    # Save processed data
    save_json(audit_data, audit_output_path)
    save_json(open_coding_data, open_coding_output_path)
    print(f"VQA-RAD dataset (open coding) processed and saved to: {open_coding_output_path}")
    print(f"VQA-RAD dataset (audit) processed and saved to: {audit_output_path}")

def process_medxpert_qa(raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, audit_size: int = None, open_coding_size: int = None):
    '''
    this function is to preprocess MedXpertQA dataset from raw to standardized format
    '''
    audit_output_dir = output_dir / "MedXpertQA-text" / "audit"
    open_coding_output_dir = output_dir / "MedXpertQA-text" / "open_coding"
    input_dir = raw_dir / "MedXpertQA-text"# raw dir
    audit_output_dir.mkdir(parents=True, exist_ok=True)
    open_coding_output_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / "test.jsonl"
    all_mc: List[Dict[str, Any]] = []
    idx = 0
    if not input_file.exists():
        print(f"Missing file, skip: {input_file}")
        return
    print(f"Converting {input_file}...")
    mc, idx = convert_items_to_medqa(parse_jsonl(input_file), start_index=idx, prefix="medxpertqa_text")
    all_mc.extend(mc)
    print(f"Total MedXpertQA-text MC questions processed: {len(all_mc)}")
    # shuffle the entire dataset
    total_needed = audit_size + open_coding_size
    if total_needed > len(all_mc):
        raise ValueError(f"the total data num requested ({total_needed})exceeds the total amount of avaliable data: ({len(all_mc)})")
    random.seed(42)
    shuffled_data = all_mc.copy()
    random.shuffle(shuffled_data)
    # slice the shuffled data into two non-overlapping subsets
    audit_data = shuffled_data[:audit_size]
    open_coding_data = shuffled_data[audit_size:audit_size + open_coding_size]
    # save audit data
    audit_out = audit_output_dir / "medqa_medxpertqa-text_audit.json"
    save_json(audit_data, audit_out)
    # save open coding data
    open_coding_out = open_coding_output_dir / "medqa_medxpertqa-text_open_coding.json"
    save_json(open_coding_data, open_coding_out)

def process_slake (raw_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR, audit_size: int = None, open_coding_size: int = None):
    '''
    this function is to process the raw slake dataset, and transform it to the dataset we use to opencoding and multi-agent audit!
    '''
    input_dir = raw_dir / "SLAKE"
    
    audit_output_dir = output_dir / "SLAKE" / "audit"
    open_coding_output_dir = output_dir / "SLAKE" / "open_coding"
    audit_output_dir.mkdir(parents=True, exist_ok=True)
    open_coding_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define files to process
    splits = [
        ("test", "test.json"),
    ]
    
    # Process each split
    all_data: List[Dict[str, Any]] = []

    for split_name, input_file in splits:
        input_path = input_dir / input_file
        if input_path.exists():
            data = process_split(split_name, input_path)
            all_data.extend(data)
        else:
            print(f"File does not exist: {input_path}")
    print(f"\nTotal SLAKE questions processed: {len(all_data)}")
    # randomly shuffle the entire dataset
    total_needed = audit_size + open_coding_size
    if total_needed > len(all_data):
        raise ValueError(f"the total data num requested ({total_needed})exceeds the total amount of avaliable data: ({len(all_data)})")
    random.seed(42)
    # copy raw table to avoid messing up the original order
    shuffled_data = all_data.copy()
    random.shuffle(shuffled_data)
    # slice the shuffled data into two non-overlapping subsets
    audit_data = shuffled_data[:audit_size]
    open_coding_data = shuffled_data[audit_size:audit_size + open_coding_size]
    # Save unified JSONL files
    audit_output_path = audit_output_dir / "medqa_slake_audit.json"
    open_coding_output_path = open_coding_output_dir / "medqa_slake_open_coding.json"
    save_json (audit_data, audit_output_path)
    save_json (open_coding_data, open_coding_output_path)
    print(f"SLAKE audit dataset processed and saved to: {audit_output_path}")
    print(f"SLAKE open coding dataset processed and saved to: {open_coding_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process medical datasets into a standardized format")
    parser.add_argument("--medqa", action="store_true", help="Process MedQA dataset")
    parser.add_argument("--pubmedqa", action="store_true", help="Process PubMedQA dataset")
    parser.add_argument("--medxpertqa", action="store_true", help="Process MedXpertQA dataset")
    parser.add_argument("--pathvqa", action="store_true", help="Process PathVQA dataset")
    parser.add_argument("--vqa-rad", action="store_true", help="Process VQA-RAD dataset")
    parser.add_argument("--slake", action="store_true", help="Process SLAKE dataset")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--raw-dir", type=str, default=RAW_DATA_DIR, help="Directory containing raw datasets")
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DATA_DIR, help="Directory to save processed datasets")
    parser.add_argument("--audit-size", type=int, default=100, help="Number of samples to randomly select for audit (None for all samples)")
    parser.add_argument("--open-coding-size", type=int, default=100, help="Number of samples to randomly select for open coding (None for all samples)")

    args = parser.parse_args()

    # If no dataset is specified, show help
    if not (args.medqa or args.pubmedqa or args.pathvqa or args.vqa_rad or args.slake or args.medxpertqa or args.all):
        parser.print_help()
        return

    # Process requested datasets
    if args.all or args.medqa:
        process_medqa(args.raw_dir, args.output_dir, args.audit_size, args.open_coding_size)

    if args.all or args.pubmedqa:
        process_pubmedqa(args.raw_dir, args.output_dir, args.audit_size, args.open_coding_size)

    if args.all or args.medxpertqa:
        process_medxpert_qa(args.raw_dir, args.output_dir, args.audit_size, args.open_coding_size)

    if args.all or args.pathvqa:
        process_pathvqa(args.raw_dir, args.output_dir, args.audit_size, args.open_coding_size)

    if args.all or args.vqa_rad:
        process_vqa_rad(args.raw_dir, args.output_dir, args.audit_size, args.open_coding_size)

    if args.all or args.slake:
        process_slake(args.raw_dir, args.output_dir, args.audit_size, args.open_coding_size)

    print("All requested datasets processed successfully!")


if __name__ == "__main__":
    main()