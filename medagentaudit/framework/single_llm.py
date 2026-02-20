"""
./medagentaudit/framework/single_llm.py
"""
import json
from typing import Dict, Any, Optional, List
import time
import argparse
from tqdm import tqdm
import sys
from pathlib import Path
from medagentaudit.utils.logger import DualLogger
from medagentaudit.utils.encode_image import encode_image
from medagentaudit.utils.json_utils import load_json, save_jsonl
from medagentaudit.core.base_agent import BaseAgent
from medagentaudit.common.agent_type import AgentType

# Ensure project root is in path
current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))

class SingleModelInference(BaseAgent):
    """
    Unified class for running inference with a single LLM or VLLM model
    using various prompting techniques.
    """

    def __init__(self, model_key: str, sample_size: int, config_path: str):
        super().__init__(agent_id="single_llm", agent_type=AgentType.SINGLELLM, config_path=config_path, model_key=model_key)
        """
        Initialize the inference handler.

        Args:
            model_key: Key identifying the model in LLM_MODELS_SETTINGS
            sample_size: Number of samples for self-consistency methods
        """
        self.sample_size = sample_size

    def _prepare_user_message(self,
                            prompt: str,
                            image_path: Optional[str] = None) -> str | List:
        """
        Prepare user message with optional image content.

        Args:
            prompt: Text prompt
            image_path: Optional path to image

        Returns:
            User message as string or list for multimodal content
        """
        if image_path:
            try:
                base64_image = encode_image(image_path)
                return [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")
                # Fall back to text-only if image encoding fails
                return prompt
        else:
            return prompt

    def zero_shot_prompt(self,
                        question: str,
                        options: Optional[Dict[str, str]] = None) -> str:
        """
        Create a zero-shot prompt for either multiple-choice or free-form questions.

        Args:
            question: Question text
            options: Optional multiple choice options

        Returns:
            Formatted prompt string
        """
        if options:
            # Multiple choice
            options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
            prompt = (
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Please respond with the letter of the correct option (A, B, C, etc.) only."
            )
        else:
            # Free form
            prompt = (
                f"Question: {question}\n\n"
                f"Please provide a concise and accurate answer."
            )

        return prompt


    def process_item(self,
                    item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single item using the specified prompting technique.

        Args:
            item: Input data dictionary with question, options, etc.
            prompt_type: Type of prompting to use

        Returns:
            Result dictionary with predicted answer and metadata
        """
        start_time = time.time()

        # Extract item fields
        qid = item.get("qid", "unknown")
        question = item.get("question", "")
        options = item.get("options")
        image_path = item.get("image_path")
        ground_truth = item.get("answer", "")

        print(f"Processing {qid}")

        # Determine if it's a multiple-choice or free-form question
        is_mc = options is not None

        # Set system message based on task type
        if image_path:
            system_message = {"role": "system", "content": "You are a medical vision expert analyzing medical images and answering questions about them."}
        else:
            system_message = {"role": "system", "content": "You are a medical expert answering medical questions with precise and accurate information."}

        prompt = self.zero_shot_prompt(question, options)

        # Prepare user message (text-only or multimodal)
        user_content = self._prepare_user_message(prompt, image_path)
        user_message = {"role": "user", "content": user_content}
        # Call LLM to get responses
        response, reasoning_content, system_message, user_message = self.call_llm(
            system_message=system_message,
            user_message=user_message,
        )

        # For zero-shot and few-shot, use the first response
        predicted_answer = response.strip()
        reasoning = "Direct answer, no explicit reasoning"
        individual_responses = [{"answer": predicted_answer, "full_response": response}]

        # Clean up the predicted answer (extract just the option letter for MC)
        if is_mc and len(predicted_answer) > 1:
            # Look for option letters in the answer
            for option in options.keys():
                if option in predicted_answer or option.lower() in predicted_answer.lower():
                    predicted_answer = option
                    break

        # Calculate processing time
        processing_time = time.time() - start_time

        # Prepare the result structure with improved details
        result = {
            "qid": qid,
            "timestamp": int(time.time()),
            "question": question,
            "options": options,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "reasoning_content": reasoning_content,
            "case_history": {
                "reasoning": reasoning,
                "model": self.model_key,
                "raw_responses": response,
                "individual_responses": individual_responses,
                "processing_time": processing_time
            }
        }

        return result


def main():
    parser = argparse.ArgumentParser(description="Run single model inference on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (MedQA, PubMedQA, PathVQA, VQA-RAD)")
    parser.add_argument("--model_key", type=str, default="qwen-max-latest",
                       help="Model key from LLM_MODELS_SETTINGS")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples for self-consistency methods")
    parser.add_argument("--time_stamp", type=str, required=True, help="Timestamp for logging purposes")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")

    args = parser.parse_args()

    # Dataset and QA type
    dataset_name = args.dataset
    model_key = args.model_key
    num_samples = args.num_samples
    timestamp = args.time_stamp
    config_path = args.config_path
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_key}")
    print(f"Sample Size: {num_samples}")

    main_llm = args.model_key

    terminal_log_file = project_root / "logs" / "single_llm" / timestamp/ f"{dataset_name}_{main_llm}_terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条

    # Set up data path
    data_path = project_root / "datasets" / "processed" / dataset_name / "audit" / f"medqa_{dataset_name.lower()}_audit.json"

    output_file = project_root / "logs" / "single_llm" / timestamp / f"{dataset_name}_{main_llm}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Data path: {data_path}")

    # Initialize the model
    model = SingleModelInference(model_key=model_key, sample_size=num_samples, config_path=config_path)

    # Load the data
    data = load_json(data_path)
    print(f"Loaded {len(data)} items from {data_path}")

    # Track stats
    processed_count = 0
    skipped_count = 0
    error_count = 0
    correct_count = 0

    existing_qids = set()
    if output_file.exists():
        print(f"Output file {output_file} already exists. Appending new results.")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                try:
                    record = json.loads(line)
                    if "qid" in record:
                        existing_qids.add(record["qid"])
                except json.JSONDecodeError:
                    print("Warning: Found a corrupted line in jsonl file, skipping.")
        print(f"Found {len(existing_qids)} already processed cases. They will be skipped.")

    # Process each item
    for item in tqdm(data[:num_samples], desc=f"Processing {dataset_name}"):
        qid = item["qid"]

        if qid in existing_qids:
            print(f"Skipping {qid} - already processed")
            continue

        try:
            # Process the item
            result = model.process_item(
                item=item
            )

            # Save the result
            save_jsonl(result, output_file)

            # Update stats
            processed_count += 1
            if result["predicted_answer"] == result["ground_truth"]:
                correct_count += 1

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            error_count += 1

    # Print summary
    print("\n" + "="*50)
    print(f"Processing Summary for {dataset_name}:")
    print(f"Total items: {len(data)}")
    print(f"Processed: {processed_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Errors: {error_count}")

    if processed_count > 0:
        accuracy = (correct_count / processed_count) * 100
        print(f"Accuracy of processed items: {accuracy:.2f}%")

    print("="*50)


if __name__ == "__main__":
    main()