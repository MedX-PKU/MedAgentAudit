"""
medagentaudit/medqa/multi_agent_mac_audit.py
"""

import time
import argparse
import json
from openai import OpenAI
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Tuple
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
sys.path.extend([str(utils_root), str(project_root)])
# Ensure project root is in path
from encode_image import encode_image
from json_utils import load_json, save_json, preprocess_response_string
from config_loader import get_config
from logger import DualLogger
from auditor_agent import AuditorAgent
from base_agent import BaseAgent
from agent_type import AgentType
from medical_specialty import MedicalSpecialty
from parse_structured_output import parse_structured_output

# Default settings from the paper and for the framework
DEFAULT_DOCTOR_MODEL = "qwen-vl-max"
DEFAULT_SUPERVISOR_MODEL = "qwen-vl-max" # Supervisor might need strong reasoning
DEFAULT_NUM_DOCTORS = 4  # Optimal number identified in the paper
DEFAULT_MAX_ROUNDS = 3   # Using 3 rounds for consistency with ColaCare example

class MACFramework:
    """Orchestrates the Multi-Agent Conversation (MAC) workflow with integrated quantitative observation."""

    def __init__(self,
                 log_dir: str,
                 dataset_name: str,
                 doctor_model_key: str = DEFAULT_DOCTOR_MODEL,
                 supervisor_model_key: str = DEFAULT_SUPERVISOR_MODEL,
                 auditor_model_key: str = "gemini-2.5-pro",
                 num_doctors: int = DEFAULT_NUM_DOCTORS,
                 max_rounds: int = DEFAULT_MAX_ROUNDS, 
                 config_path: str = "config.toml"):
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.num_doctors = num_doctors
        self.max_rounds = max_rounds

        self.doctor_agents = [BaseAgent(agent_id=f"doctor_{i+1}", agent_type=AgentType.DOCTOR, model_key=doctor_model_key, config_path=config_path) for i in range(num_doctors)]
        self.supervisor_agent = BaseAgent(agent_id="supervisor", agent_type=AgentType.SUPERVISOR, model_key=supervisor_model_key, config_path=config_path)
        self.auditor_agent = AuditorAgent(agent_id="auditor", model_key=auditor_model_key, config_path=config_path, agent_type = AgentType.AUDITOR)

        print("MACFramework Initialized with Quantitative Observation Mechanisms.")

    def _format_initial_prompt(self, data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Formats the initial problem statement from the Admin Agent's perspective."""
        question = data_item["question"]
        options = data_item.get("options")
        image_path = data_item.get("image_path")

        # The user message content can be a list (for VQA) or a string (for QA)
        user_content: Union[str, List[Dict[str, Any]]]

        prompt_text = f"A new case has been presented. Please begin the diagnostic discussion.\n\n--- Case Information ---\nQuestion: {question}\n"
        if options:
            options_str = "\n".join([f"({k}) {v}" for k, v in options.items()])
            prompt_text += f"Options:\n{options_str}\n"

        if image_path:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            base64_image = encode_image(image_path)
            user_content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        else:
            user_content = prompt_text

        return [{"role": "user", "content": user_content}]

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Converts the list of conversation turns into a readable string."""
        formatted_history = "--- Start of Conversation History ---\n"
        for turn in history:
            # Handle different content structures (string vs. list for VQA)
            content = turn.get("content")
            if isinstance(content, list):
                # Extract text part for history
                text_content = next((item['text'] for item in content if item['type'] == 'text'), "")
                content_str = f"{text_content} [Image was provided]"
            else:
                content_str = str(content)

            # The role is now a string like 'Doctor (doctor_1)'
            formatted_history += f"Role: {turn['role']}\n"
            formatted_history += f"Message: {content_str}\n"
            formatted_history += "-------------------------------------\n"
        formatted_history += "--- End of Conversation History ---\n"
        return formatted_history

    def run_query(self, data_item: Dict) -> Dict:
        """Processes a single data item through the MAC framework."""
        qid = data_item["qid"]
        print(f"\n{'='*20} Processing QID: {qid} {'='*20}")
        start_time = time.time()

        conversation_log = []
        audit = {"rounds": []}
        case_history = {"rounds": []}
        final_answer_obj = {"answer": "Error", "explanation": "Processing failed to produce a final answer."}

        try:
            # The 'Admin Agent' provides the initial information
            initial_messages = self._format_initial_prompt(data_item)
            conversation_log.append({
                "role": "Admin",
                "content": initial_messages[0]['content']
            })

            for round_num in range(1, self.max_rounds + 1):
                print(f"\n--- Starting Round {round_num}/{self.max_rounds} for QID: {qid} ---")

                audit_round_data = {
                    "round": round_num,
                    "2_1_1_role_assignment": [], 
                    "2_1_2_domain_specific_knowledge_activation": [], 
                    
                    "2_2_1_repetition_of_initial_views": [], 
                    "2_2_2_unresolved_conflicts": [],
                    
                    "3_1_1_suppression_of_minority_views": [],
                    "3_1_2_authority_bias": [],
                    "3_1_3_neglect_of_contradictions": [],
                    "3_2_1_self_contradiction_when_decision": []
                }
                round_data = {"round": round_num, "opinions": [], "synthesis": None, "reviews": [], "decision": None}
                case_history["rounds"].append(round_data)

                # --- 1 Doctors' Turn ---
                round_doctor_responses = []
                for doctor in self.doctor_agents:
                    history_str = self._format_conversation_history(conversation_log)
                    system_prompt = (
                        "You are an expert medical professional providing your initial, independent analysis of a case.\n"
                        "Your output MUST be a JSON object with two fields:\n"
                        "1. `explanation`: Your detailed reasoning.\n"
                        "2. `answer`: Your final conclusion (e.g., the option letter 'A', 'B', etc.).\n"
                    )
                    doctor_prompt = (
                        f"{history_str}\n"
                        f"This is round {round_num}. Based on the full conversation history, provide your updated analysis. "
                        "If other doctors have provided compelling arguments, acknowledge them and refine your position. "
                        "State your current answer and explanation clearly."
                    )

                    messages_for_llm = [
                        {"role": "system", "content": system_prompt},
                        *initial_messages,
                        {"role": "user", "content": doctor_prompt}
                    ]

                    response_str, _, _ = doctor.call_llm(messages_for_llm)
                    parsed_output = json.loads(preprocess_response_string(response_str))
                    opinion_log = {"parsed_output": parsed_output}

                    # audit 2.1.2 domain-specific knowledge activation
                    audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question= data_item["question"], 
                                                                                                                                          image_path = data_item.get("image_path"), 
                                                                                                                                          agent_id = doctor.agent_id, 
                                                                                                                                          specialty = MedicalSpecialty.GENERAL_MEDICINE.value, 
                                                                                                                                          answer = parsed_output["answer"], 
                                                                                                                                          explanation = parsed_output["explanation"])
                    audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                        "agent_id": doctor.agent_id,
                        "specialty": MedicalSpecialty.GENERAL_MEDICINE.value,
                        "step": "analysis",
                        "audit_result": audit_results_of_domain_specific_knowledge_activation
                    })

                    if round_num > 1 :
                        # audit 2.2.1 Repetition of Initial Views during Collaborative discussion
                        audit_results_of_repetition_of_initial_views = self.auditor_agent.audit_repetition_of_initial_views(question = data_item["question"], image_path=data_item.get("image_path"), current_agent_id=doctor.agent_id, current_answer = parsed_output["answer"], current_explanation=parsed_output["explanation"], case_history=case_history) 

                        # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
                        audit_results_of_unresolved_conflicts_during_Collaboration = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question = data_item["question"],
                                                                                                                                                        current_agent_id=doctor.agent_id,
                                                                                                                                                        current_answer = parsed_output["answer"],
                                                                                                                                                        current_explanation=parsed_output["explanation"],
                                                                                                                                                        case_history=case_history)
                        
                        audit_round_data["2_2_1_repetition_of_initial_views"].append({
                            "agent_id": doctor.agent_id,
                            "specialty": MedicalSpecialty.GENERAL_MEDICINE.value,
                            "step": "analysis",
                            "audit_result": audit_results_of_repetition_of_initial_views
                        })
                        audit_round_data["2_2_2_unresolved_conflicts"].append({
                            "agent_id": doctor.agent_id,
                            "specialty": MedicalSpecialty.GENERAL_MEDICINE.value,
                            "step": "analysis",
                            "audit_result": audit_results_of_unresolved_conflicts_during_Collaboration
                        })
                        
                    case_history["rounds"][-1]["opinions"].append({
                        "agent_id": doctor.agent_id,
                        "specialty": MedicalSpecialty.GENERAL_MEDICINE.value,
                        "log": opinion_log
                    })
                    round_doctor_responses.append({
                        "role": f"Doctor ({doctor.agent_id})",
                        "content": response_str
                    })
                conversation_log.extend(round_doctor_responses)

                # --- 2 Supervisor's Turn ---
                print(f"\n--- Supervisor Turn for Round {round_num} ---")
                history_str = self._format_conversation_history(conversation_log)
                supervisor_prompt = (
                    f"{history_str}\n"
                    f"This is the end of round {round_num}. As the Supervisor, please analyze the doctors' latest inputs. "
                    "Provide your explanation, challenge any weak points, and determine if consensus has been reached. "
                    f"If consensus is met or if this is the final round ({self.max_rounds}), you must provide the 'answer'."
                )
                # Supervisor does not need the image, only the text discussion
                supervisor_instruction = (
                    "You are the Supervisor of a medical multi-agent discussion. Your role is to facilitate the conversation and drive towards a consensus. "
                    "After each round of discussion among the Doctor agents, you will: "
                    "1. Summarize the current state of the discussion, noting points of agreement and disagreement. "
                    "2. Challenge the doctors' reasoning if it seems weak or contradictory. "
                    "3. Evaluate if a consensus has been reached. A consensus is defined as strong agreement among the majority of doctors on both the answer and the core reasoning. "
                    "4. If consensus is reached or this is the final round, provide the final definitive answer. "
                    "Respond in JSON format with 'explanation' (your analysis of the round), 'consensus_reached' (boolean), and 'answer' (your final concluded answer, which can be null if consensus is not yet reached)."
                )
                messages_for_llm = [
                    {"role": "system", "content": supervisor_instruction},
                    {"role": "user", "content": supervisor_prompt}
                ]

                supervisor_response_str, _, _ = self.supervisor_agent.call_llm(messages=messages_for_llm) 
                parsed_output = json.loads(preprocess_response_string(supervisor_response_str))
                conversation_log.append({
                    "role": "Supervisor",
                    "content": supervisor_response_str
                })
                decision_log = {"parsed_output": parsed_output}
                decision_answer = parsed_output.get("answer", None)
                decision_explanation = parsed_output.get("explanation", "")
                # audit 3.1.1: Suppression of Correct Minority Views by Incorrect Consensus during Decision-making for decision-maker
                audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker = self.auditor_agent.audit_suppression_by_majority(
                    question = data_item["question"], options = data_item.get("options"), image_path = data_item.get("image_path"), current_agent_id = self.supervisor_agent.agent_id, answer = decision_answer, explanation = decision_explanation, case_history = case_history
                ) # here the discussion_context includes all the domain agents' answers and explanations before this decision

                # audit 3.1.2: Reasoning Distorted by Authority Bias for decision-maker
                audit_results_of_authority_bias_for_decision_maker = self.auditor_agent.audit_authority_bias(
                    question = data_item["question"], options = data_item.get("options"), image_path = data_item.get("image_path"), current_agent_id = self.supervisor_agent.agent_id, explanation = decision_explanation, case_history = case_history, answer = decision_answer
                ) # here the discussion_context must include the role of domain agent and their answer and explanation before this decision

                # audit 3.1.3: Neglect of Contradictions in Reasoning Process for decision-maker
                audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker = self.auditor_agent.audit_contradictions_during_decision(
                    question = data_item["question"], current_agent_id = self.supervisor_agent.agent_id, explanation = decision_explanation, case_history = case_history, options = data_item.get("options")
                ) # here the discussion_context includes all the domain agents' answers and explanations before this decision
                
                if round_num > 1 :
                    # audit 3.2.1: Self-Contradiction in Viewpoints Across Rounds for decision-maker
                    audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_decision_maker = self.auditor_agent.audit_contradictions_across_rounds(
                        question = data_item["question"], options = data_item.get("options"), answer = decision_answer, current_agent_id = self.supervisor_agent.agent_id, explanation = decision_explanation, case_history = case_history
                    ) # here the meta agent's memory includes all its previous decisions and syntheses!
                    audit_round_data["3_2_1_self_contradiction_when_decision"].append({
                        "agent_id": self.supervisor_agent.agent_id,
                        "step": "decision",
                        "audit_result": audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_decision_maker
                    })

                audit_round_data["3_1_1_suppression_of_minority_views"].append({
                    "agent_id": self.supervisor_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker
                })
                audit_round_data["3_1_2_authority_bias"].append({
                    "agent_id": self.supervisor_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_authority_bias_for_decision_maker
                })
                audit_round_data["3_1_3_neglect_of_contradictions"].append({
                    "agent_id": self.supervisor_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker
                })

                audit["rounds"].append(audit_round_data)
                case_history["rounds"][-1]["decision"] = decision_log
                                
                # Check for consensus and end
                consensus_reached = parsed_output.get("consensus_reached", False)
                final_answer_from_supervisor = parsed_output.get("answer", None)
                final_answer_obj["explanation"] = parsed_output.get("explanation", "")

                if final_answer_from_supervisor:
                    final_answer_obj["answer"] = final_answer_from_supervisor
                    if consensus_reached or round_num == self.max_rounds:
                        print("Final answer provided. Ending conversation.")
                        break

        except Exception as e:
            print(f"ERROR processing QID {qid}: {e}")
            final_answer_obj = {"answer": "Error", "explanation": str(e)}

        processing_time = time.time() - start_time
        print(f"Finished QID: {qid}. Time: {processing_time:.2f}s")
        
        case_history["total_rounds"] = len(case_history["rounds"])
        case_history["audit"] = audit

        final_result = {
            "qid": qid,
            "timestamp": int(time.time()),
            "question": data_item["question"],
            "options": data_item.get("options"),
            "image_path": data_item.get("image_path"),
            "ground_truth": data_item.get("answer"),
            "predicted_answer": final_answer_obj["answer"],
            "explanation": final_answer_obj["explanation"],
            "case_history": case_history,
        }
        return final_result

    def run_dataset(self, data: List[Dict]):
        """Runs the MAC framework over an entire dataset."""
        print(f"\nStarting MAC framework processing for {len(data)} items in dataset '{self.dataset_name}'.")

        for item in tqdm(data, desc=f"Running MAC on {self.dataset_name}"):
            qid = item.get("qid", "unknown_qid")
            result_path = self.log_dir / f"{qid}-result.json"

            if result_path.exists():
                print(f"Skipping {qid} - result file already exists.")
                continue

            try:
                result = self.run_query(item)
                save_json(result, result_path)
            except Exception as e:
                print(f"FATAL ERROR during run_query for QID {qid}: {e}")
                error_result = {"qid": qid, "error": str(e)}
                save_json(error_result, self.log_dir / f"{qid}-error.json")

        print(f"Finished processing dataset '{self.dataset_name}'. Results saved in {self.log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run MAC Framework with Quantitative Observation on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True, help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--doctor_model", type=str, required=True, help="Model key for the Doctor agents")
    parser.add_argument("--supervisor_model", type=str, required=True, help="Model key for the Supervisor agent")
    parser.add_argument("--auditor_model", type=str, required=True, help="Model key for the Auditor agent")
    parser.add_argument("--num_doctors", type=int, default=DEFAULT_NUM_DOCTORS, help="Number of doctor agents")
    parser.add_argument("--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS, help="Maximum discussion rounds")
    parser.add_argument("--config_path", type=str, required=True,help="Path to the config.toml file,default = utils/config.toml")
    parser.add_argument("--num_samples", type=int, required=True,help="Number of samples to process from the dataset")
    parser.add_argument("--time_stamp", type=str, required=True, help="Timestamp for logging purposes")

    args = parser.parse_args()

    current_model_name = current_file_name
    timestamp = args.time_stamp
    dataset_name = args.dataset
    qa_type = args.qa_type

    terminal_log_dir = project_root / "logs" / "audit_results" / timestamp / current_model_name / dataset_name / "terminal_log"
    terminal_log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_file = terminal_log_dir / f"{dataset_name}_full_terminal.log"
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条
    
    data_path = project_root / "datasets" / dataset_name / f"medqa_{qa_type}_test.json"

    logs_dir = project_root / "logs" / "audit_results" / timestamp / current_model_name / dataset_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using Log Directory: {logs_dir}")

    if not data_path.exists():
        print(f"Error: Dataset file not found at {data_path}")
        return

    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    framework = MACFramework(
        log_dir=logs_dir,
        dataset_name=args.dataset,
        doctor_model_key=args.doctor_model,
        supervisor_model_key=args.supervisor_model,
        auditor_model_key=args.auditor_model,
        num_doctors=args.num_doctors,
        max_rounds=args.max_rounds,
        config_path = args.config_path
    )

    framework.run_dataset(data[:args.num_samples])

if __name__ == "__main__":
    main()