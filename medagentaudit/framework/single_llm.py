"""
./medagentaudit/framework/single_llm.py
"""

from openai import OpenAI
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import time
import argparse
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure project root is in path
current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
utils_root = current_file_path.parents[1] / "utils"
auditor_root = current_file_path.parents[1] / "auditor"
common_root = current_file_path.parents[1] / "common"
core_root = current_file_path.parents[1] / "core"
project_root = current_file_path.parents[2]
sys.path.extend([str(utils_root), str(project_root), str(auditor_root), str(common_root)])
from logger import DualLogger
from encode_image import encode_image
from json_utils import load_json, save_json, preprocess_response_string
from base_agent import BaseAgent
from agent_type import AgentType
from parse_structured_output import parse_structured_output

class SingleModelInference(BaseAgent):
    """
    Unified class for running inference with a single LLM or VLLM model
    using various prompting techniques.
    """

    def __init__(self, model_key: str, sample_size: int):
        super().__init__(agent_id="single_llm", agent_type=AgentType.SINGLELLM, config_path="config.toml", model_key=model_key)
        """
        Initialize the inference handler.

        Args:
            model_key: Key identifying the model in LLM_MODELS_SETTINGS
            sample_size: Number of samples for self-consistency methods
        """
        self.sample_size = sample_size

    def _call_llm(self,
                 system_message: str,
                 user_message: str | List,
                 n_samples: int = 1,
                 max_retries: int = 3,
                 response_format: Dict | None = None) -> List[str]:
        """
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message (text or multimodal content)
            response_format: Optional format specification for response
            n_samples: Number of samples to generate
            max_retries: Maximum number of retry attempts

        Returns:
            List of LLM response texts
        """
        retries = 0
        all_responses = []

        # For each sample we need
        remaining_samples = n_samples

        while remaining_samples > 0 and retries < max_retries:
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]

                # Some models might not properly support n > 1, so we make multiple calls if needed
                current_n = min(remaining_samples, 1)  # Request just 1 at a time to be safe

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format,
                    n=current_n,
                    stream=False
                )

                responses = [choice.message.content for choice in completion.choices]
                all_responses.extend(responses)
                remaining_samples -= len(responses)

                # Reset retry counter on successful API call
                retries = 0

            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    if all_responses:  # If we have some responses, use those rather than failing
                        print(f"Warning: Only obtained {len(all_responses)}/{n_samples} samples after max retries")
                        break
                    else:
                        raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(1)  # Brief pause before retrying

        return all_responses

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
                    item: Dict[str, Any],
                    prompt_type: str,
                    dataset: str) -> Dict[str, Any]:
        """
        Process a single item using the specified prompting technique.

        Args:
            item: Input data dictionary with question, options, etc.
            prompt_type: Type of prompting to use
            dataset: Dataset name

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

        print(f"Processing {qid} with {prompt_type} prompting")

        # Determine if it's a multiple-choice or free-form question
        is_mc = options is not None

        # Set system message based on task type
        if image_path:
            system_message = "You are a medical vision expert analyzing medical images and answering questions about them."
        else:
            system_message = "You are a medical expert answering medical questions with precise and accurate information."

        prompt = self.zero_shot_prompt(question, options)
        response_format = None
        n_samples = 1

        # Prepare user message (text-only or multimodal)
        user_message = self._prepare_user_message(prompt, image_path)

        # Call LLM to get responses
        responses = self._call_llm(
            system_message=system_message,
            user_message=user_message,
            response_format=response_format,
            n_samples=n_samples
        )

        voting_details = None
        # Process responses based on prompt type

        # For zero-shot and few-shot, use the first response
        predicted_answer = responses[0].strip()
        reasoning = "Direct answer, no explicit reasoning"
        individual_responses = [{"answer": predicted_answer, "full_response": responses[0]}]

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
            "case_history": {
                "reasoning": reasoning,
                "prompt_type": prompt_type,
                "model": self.model_key,
                "raw_responses": responses,
                "individual_responses": individual_responses,
                "voting_details": voting_details,
                "processing_time": processing_time
            }
        }

        return result


def main():
    parser = argparse.ArgumentParser(description="Run single model inference on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (MedQA, PubMedQA, PathVQA, VQA-RAD)")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True,
                       help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--prompt_type", type=str, required=True,
                       choices=["zero_shot", "few_shot", "cot", "self_consistency", "cot_sc"],
                       help="Prompting technique to use")
    parser.add_argument("--model_key", type=str, default="qwen-max-latest",
                       help="Model key from LLM_MODELS_SETTINGS")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples for self-consistency methods")
    parser.add_argument("--time_stamp", type=str, required=True, help="Timestamp for logging purposes")

    args = parser.parse_args()

    # Dataset and QA type
    dataset_name = args.dataset
    qa_type = args.qa_type
    prompt_type = args.prompt_type
    model_key = args.model_key
    num_samples = args.num_samples
    timestamp = args.time_stamp

    print(f"Dataset: {dataset_name}")
    print(f"QA Type: {qa_type}")
    print(f"Prompt Type: {prompt_type}")
    print(f"Model: {model_key}")
    print(f"Sample Size: {num_samples}")

    main_llm = args.model_key

    terminal_log_dir = project_root / "logs" / "single_llm" / timestamp / dataset_name / main_llm / "terminal_log"
    terminal_log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_file = terminal_log_dir / f"{dataset_name}_full_terminal.log"
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条

    # Method name for logging
    method = f"SingleLLM_{prompt_type}"

    # Set up data path
    data_path = project_root / "datasets" / dataset_name / f"medqa_{qa_type}_test.json"

    logs_dir = project_root / "logs" / "single_llm" / timestamp / dataset_name / main_llm
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved to: {logs_dir}")

    print(f"Data path: {data_path}")

    # Initialize the model
    model = SingleModelInference(model_key=model_key, sample_size=num_samples)

    # Load the data
    data = load_json(data_path)
    print(f"Loaded {len(data)} items from {data_path}")

    # Track stats
    processed_count = 0
    skipped_count = 0
    error_count = 0
    correct_count = 0

    # Process each item
    for item in tqdm(data, desc=f"Processing {dataset_name} with {prompt_type}"):
        qid = item["qid"]

        # Skip if already processed
        result_path = logs_dir / f"{qid}-result.json"
        if result_path.exists():
            print(f"Skipping {qid} - already processed")
            skipped_count += 1
            continue

        try:
            # Process the item
            result = model.process_item(
                item=item,
                prompt_type=prompt_type,
                dataset=dataset_name
            )

            # Save the result
            save_json(result, result_path)

            # Update stats
            processed_count += 1
            if result["predicted_answer"] == result["ground_truth"]:
                correct_count += 1

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            error_count += 1

    # Print summary
    print("\n" + "="*50)
    print(f"Processing Summary for {dataset_name} ({qa_type}) with {prompt_type}:")
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