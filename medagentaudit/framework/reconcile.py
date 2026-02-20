"""
./medagentaudit/framework/reconcile.py
""" 
import os
import json
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import argparse
from tqdm import tqdm
from pathlib import Path
import sys
import traceback
from medagentaudit.utils.json_utils import load_json, save_jsonl, preprocess_response_string
from medagentaudit.utils.encode_image import encode_image
from medagentaudit.utils.logger import DualLogger
from medagentaudit.auditor.auditor_agent import AuditorAgent
from medagentaudit.core.base_agent import BaseAgent
from medagentaudit.common.agent_type import AgentType
from medagentaudit.common.medical_specialty import MedicalSpecialty
# Ensure project root is in path
current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))


class DiscussionPhase(Enum):
    """Enumeration of discussion phases in the Reconcile framework."""
    INITIAL = "initial"        # Initial answer generation
    DISCUSSION = "discussion"  # Multi-round discussion
    FINAL = "final"            # Final team answer

class ReconcileAgent(BaseAgent):
    """
    An agent participating in the Reconcile framework.
    This version is enhanced for quantitative observation.
    """
    def __init__(self, agent_id: str, model_key: str, config_path: str):
        super().__init__(agent_id = agent_id, agent_type=AgentType.RECONCILE, model_key=model_key, config_path=config_path)
        """
        Initialize a Reconcile agent.
        """
        self.discussion_history = []
        self.memory = []
        self.specialty = MedicalSpecialty.GENERAL_MEDICINE
        print(f"Initialized agent {self.agent_id} with model {self.model_name}")


    def generate_initial_response(self,
                                question: str,
                                options: Optional[Dict[str, str]] = None,
                                image_path: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate an initial response.
        """
        print(f"Agent {self.agent_id} generating initial response")

        system_message = {
            "role": "system",
            "content": (
                "You are a medical expert assistant. Analyze the following medical question "
                "and provide a clear answer along with detailed step-by-step reasoning. "
                "Based on your understanding, estimate your confidence in your answer "
                "on a scale from 0.0 to 1.0, where 1.0 means complete certainty."
            )
        }

        user_content = []
        if image_path and os.path.exists(image_path):
            try:
                base64_image = encode_image(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")

        question_text = question
        if options:
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_text = f"{question}\n\nOptions:\n{options_text}"

        prompt_text = (
            f"Case Description:\n{question_text}\n\n"
            "Please provide your complete analysis as a single, well-formatted JSON object "
            "strictly adhering to the four required fields: 'reasoning', 'answer', 'confidence'."
        )
        user_content.append({"type": "text", "text": prompt_text})
        user_message = {"role": "user", "content": user_content}

        response_text, reasoning_content, system_message, user_message = self.call_llm(system_message = system_message, user_message = user_message, response_format={"type": "json_object"})
        result = self._parse_response(response_text)
        user_message["content"] = [item for item in user_message["content"] if item.get("type") != "image_url"]

        step_log = {
            "llm_input":{
                "system_message": system_message,
                "user_message": user_message
            },
            "reasoning_content": reasoning_content
        }
        self.memory.append({"phase": DiscussionPhase.INITIAL.value, "response": result})
        return result, step_log

    def generate_discussion_response(self,
                                  question: str,
                                  discussion_prompt: str,
                                  current_round: int,
                                  options: Optional[Dict[str, str]] = None,
                                  image_path: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a discussion response, now with viewpoint attribution and KEU/CCP integration.
        """
        print(f"Agent {self.agent_id} generating discussion response for round {current_round}")

        system_message = {
            "role": "system",
            "content": (
                "You are a medical expert participating in a multi-agent discussion. "
                "Review the opinions from other experts, then provide your updated analysis. "
                "You may change your opinion if others' reasoning convinces you, or defend your position "
                "with clear explanations. Estimate your confidence in your answer on a scale from 0.0 to 1.0."
            )
        }

        user_content = []
        if image_path and os.path.exists(image_path):
            try:
                base64_image = encode_image(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")

        question_text = question
        if options:
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_text = f"{question}\n\nOptions:\n{options_text}"
        

        # Add question and discussion prompt
        prompt_text = (
            f"Original question: {question_text}\n\n"
            f"Discussion from other experts:\n{discussion_prompt}\n\n"
            f"Based on this discussion, provide your updated analysis in JSON format with the following fields:\n"
            f"- 'reasoning': your detailed step-by-step analysis\n"
            f"- 'answer': your final answer"
            f"\n- 'confidence': a number between 0.0 and 1.0 representing your confidence level\n\n"
        )

        user_content.append({"type": "text", "text": prompt_text})
        user_message = {"role": "user", "content": user_content}
        
        response_text, reasoning_content, system_message, user_message = self.call_llm(system_message = system_message, user_message = user_message, response_format={"type": "json_object"})
        result = self._parse_response(response_text)
        user_message["content"] = [item for item in user_message["content"] if item.get("type") != "image_url"]

        step_log = {
            "llm_input":{
                "system_message": system_message,
                "user_message": user_message
            },
            "reasoning_content": reasoning_content
        }
        step_log["discussion_context_provided"] = discussion_prompt

        self.memory.append({
            "phase": DiscussionPhase.DISCUSSION.value,
            "round": current_round,
            "response": result
        })
        return result, step_log

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured format.

        Args:
            response_text: The raw response text from the LLM

        Returns:
            A dictionary with reasoning, answer, and confidence
        """
        try:
            result = json.loads(preprocess_response_string(response_text))

            # Validate required fields
            if "reasoning" not in result:
                result["reasoning"] = "No reasoning provided"

            if "answer" not in result:
                result["answer"] = ""

            if "confidence" not in result:
                result["confidence"] = 0.0
            else:
                # Ensure confidence is a float between 0 and 1
                try:
                    result["confidence"] = float(result["confidence"])
                    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                except (ValueError, TypeError):
                    result["confidence"] = 0.0

            return result

        except json.JSONDecodeError:
            print(f"Agent {self.agent_id} failed to parse JSON response: {response_text[:100]}...")

            # Attempt to extract with simple parsing
            reasoning = ""
            answer = ""
            confidence = 0.0

            lines = response_text.split('\n')
            for line in lines:
                if line.lower().startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.lower().startswith("answer:"):
                    answer = line.split(":", 1)[1].strip()
                elif line.lower().startswith("confidence:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))
                    except (ValueError, IndexError):
                        confidence = 0.0

            # If basic parsing doesn't work, use the raw text
            if not reasoning:
                reasoning = response_text

            return {
                "reasoning": reasoning,
                "answer": answer,
                "confidence": confidence
            }


class ReconcileCoordinator:
    """
    The coordinator for the Reconcile framework, enhanced with auditing capabilities.
    """
    def __init__(self, agent_configs: List[Dict[str, str]], max_rounds: int = 3, auditor_model_key: str = "gemini-2.5-pro", conflict_analysis_model_key: str = "gemini-2.5-pro", config_path: str = "config.toml"):
        """
        Initialize the Reconcile coordinator with auditing agents.
        """
        self.agents = [ReconcileAgent(agent_id=cfg["agent_id"], model_key=cfg["model_key"], config_path=config_path) for cfg in agent_configs]
        self.max_rounds = max_rounds
        self.auditor_agent = AuditorAgent(agent_id="auditor", model_key=auditor_model_key, config_path=config_path, agent_type=AgentType.AUDITOR)
        print(f"Initialized ReconcileCoordinator with {len(self.agents)} agents, max_rounds={max_rounds}")

    def _group_answers(self, answers: List[Dict[str, Any]]) -> str:
        # (This method remains unchanged from the original file)
        groups = {}
        for ans in answers:
            answer_value = ans.get("answer", "")
            if not isinstance(answer_value, str): 
                answer_value = str(answer_value)
            answer_text = answer_value.strip().lower()
            if not answer_text: 
                continue
            
            confidence = ans.get("confidence", 0.0)
            if answer_text not in groups:
                groups[answer_text] = {"count": 0, "explanations": [], "confidence_sum": 0.0}
            groups[answer_text]["count"] += 1
            groups[answer_text]["explanations"].append(ans.get("reasoning", ""))
            groups[answer_text]["confidence_sum"] += confidence

        grouped_str = ""
        for ans_text, data in sorted(groups.items(), key=lambda item: item[1]['count'], reverse=True):
            avg_confidence = data["confidence_sum"] / data["count"] if data["count"] > 0 else 0
            grouped_str += f"Answer Option: '{ans_text.upper()}'\n"
            grouped_str += f"- Supporters: {data['count']}\n"
            grouped_str += f"- Average Confidence: {avg_confidence:.2f}\n"
            grouped_str += "- Supporting Reasons:\n"
            for i, exp in enumerate(data["explanations"]):
                exp_short = (exp[:400] + '...') if len(exp) > 400 else exp
                grouped_str += f"  - Reason {i+1}: {exp_short}\n"
            grouped_str += "\n"
        return grouped_str.strip()

    def _recalibrate(self, confidence: float) -> float:
        # (This method remains unchanged)
        if confidence == 1.0: 
            return 1.0
        elif confidence >= 0.9: 
            return 0.8
        elif confidence >= 0.8: 
            return 0.5
        elif confidence > 0.6: 
            return 0.3
        else: 
            return 0.1

    def _weighted_vote(self, answers: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        # (This method remains unchanged, but benefits from better logging)
        vote_weights = {}
        for ans in answers:
            answer_value = ans.get("answer", "")
            if not isinstance(answer_value, str): 
                answer_value = str(answer_value)
            answer = answer_value.strip()
            if not answer: 
                continue
            
            confidence = ans.get("confidence", 0.0)
            weight = self._recalibrate(confidence)
            key = answer.lower()
            if key not in vote_weights:
                vote_weights[key] = {"weight": 0, "original": answer, "supporters": 0}
            vote_weights[key]["weight"] += weight
            vote_weights[key]["supporters"] += 1

        if not vote_weights:
            return "", {"vote_weights": {}, "winner": "No valid answer", "note": "All agents provided empty answers."}

        winner_key = max(vote_weights, key=lambda k: vote_weights[k]["weight"])
        final_decision = vote_weights[winner_key]["original"]
        voting_log = {"vote_weights": vote_weights, "winner_key": winner_key, "final_decision": final_decision}
        return final_decision, voting_log

    def _check_consensus(self, answers: List[Dict[str, Any]]) -> bool:
        # (This method remains unchanged)
        valid_answers = []
        for ans in answers:
            answer_value = ans.get("answer", "")
            if not isinstance(answer_value, str): 
                answer_value = str(answer_value)
            answer_text = answer_value.strip()
            if answer_text: 
                valid_answers.append(answer_text.lower())
        return len(valid_answers) > 0 and len(set(valid_answers)) == 1

    def run_discussion(self,
                      qid: str,
                      question: str,
                      options: Dict[str, str] | None = None,
                      image_path: str | None = None,
                      task:str = "open_coding") -> Dict[str, Any]:
        """
        Run the complete discussion process with integrated quantitative observation mechanisms.
        """
        print(f"Starting discussion with {len(self.agents)} agents on question: {question}")
        start_time = time.time()

        discussion_history = []
        if task =="audit":
            audit = {"rounds": []}
            audit_round_data = {
                "round": 1,
                "1_1_1_factual_hallucination": [],
                "1_2_1_neglect_or_misinterpretation_of_modality_info": [],

                "2_1_1_role_assignment": [], 
                "2_1_2_domain_specific_knowledge_activation": [],
                
                "2_2_1_repetition_of_initial_views": [],
                "2_2_2_unresolved_conflicts": [],
                
                "3_1_1_suppression_of_minority_views": [],
                "3_1_2_authority_bias": [],
                "3_1_3_neglect_of_contradictions": [],
                "3_2_1_self_contradiction_when_decision": []
            }
        case_history = {"rounds": []}
        round_data = {"round": 1, "opinions": [], "synthesis": None, "reviews": [], "decision": None}
        case_history["rounds"].append(round_data)
        doctor_opinions = []
        # === Phase 1: Initial responses ===
        print("Phase 1: Generating initial responses")
        current_answers = []
        for agent in self.agents:
            resp, step_log = agent.generate_initial_response(question = question, options = options, image_path = image_path)
            current_answers.append(resp)
            initial_answer = resp.get("answer", "")
            initial_explanation = resp.get("reasoning", "")
            opinion_log = {
            "parsed_output":
            {
                "answer": initial_answer,
                "explanation": initial_explanation,
            },
            "reasoning_content": step_log.get("reasoning_content", "")
            }
            doctor_opinions.append(opinion_log["parsed_output"])
            if task == "audit":
                # audit 1.1.1 facutal hallucination
                audit_results_of_factual_hallucination = self.auditor_agent.audit_factual_hallucination(question = question, image_path=image_path, agent_id=agent.agent_id, specialty=agent.specialty, answer=initial_answer, explanation=initial_explanation)

                # audit 1.2.1 neglect or misinterpretation of modality information
                audit_results_of_neglect_or_misinterpretation_of_modality_info = self.auditor_agent.audit_neglect_or_misinterpretation_of_modality_info(question = question, image_path=image_path, agent_id=agent.agent_id, specialty=agent.specialty, answer=initial_answer, explanation=initial_explanation)

                # audit 2.1.2 domain-specific knowledge activation
                audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question, image_path=image_path, agent_id=agent.agent_id, specialty=agent.specialty, answer=initial_answer, explanation=initial_explanation)

                audit_round_data["1_1_1_factual_hallucination"].append({
                    "agent_id": agent.agent_id,
                    "specialty": agent.specialty.value,
                    "step": "analysis",
                    "audit_result": audit_results_of_factual_hallucination
                })
                
                audit_round_data["1_2_1_neglect_or_misinterpretation_of_modality_info"].append({
                    "agent_id": agent.agent_id,
                    "specialty": agent.specialty.value,
                    "step": "analysis",
                    "audit_result": audit_results_of_neglect_or_misinterpretation_of_modality_info
                })

                audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                    "agent_id": agent.agent_id,
                    "specialty": agent.specialty.value,
                    "step": "analysis",
                    "audit_result": audit_results_of_domain_specific_knowledge_activation
                })
            
            case_history["rounds"][-1]["opinions"].append({
                "agent_id": agent.agent_id,
                "specialty": agent.specialty.value,
                "log": opinion_log 
            })
            
            discussion_history.append({
                "phase": DiscussionPhase.INITIAL.value,
                "agent_id": agent.agent_id, "model_name": agent.model_name,
                "response": resp, "interaction_log": step_log
            })
            print(f"Agent {agent.agent_id} initial answer: {resp.get('answer', '')} (conf: {resp.get('confidence', 0.0):.2f})")

        # === Phase 2: Multi-round discussion (Review Phase) ===
        round_num = 0
        consensus_reached = False
        while round_num < self.max_rounds and not consensus_reached:
            round_num += 1
            print(f"Phase 2: Discussion round {round_num}/{self.max_rounds}")
            if round_num > 1:
                if task == "audit":
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

            discussion_prompt = self._group_answers(current_answers)

            new_answers = []
            for agent in self.agents:
                resp, step_log = agent.generate_discussion_response(
                    question, discussion_prompt, round_num, options, image_path
                )
                new_answers.append(resp)
                explanation = resp.get("reasoning", "")
                answer = resp.get("answer", "")
                review_log = {"parsed_output":{
                    "answer": answer,
                    "explanation": explanation,
                },
                "reasoning_content": step_log.get("reasoning_content", "")}
                if task == "audit":
                    # audit 2.1.2 Failure to Activate Specialist Knowledge During Role Execution during Collaborative discussion
                    audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question, image_path=image_path, agent_id=agent.agent_id, specialty=agent.specialty, answer=answer, explanation=explanation)

                    # audit 2.2.1 Repetition of Initial Views during Collaborative discussion 
                    audit_results_repetition_of_initial_views = self.auditor_agent.audit_repetition_of_initial_views(question=question, image_path = image_path, current_agent_id=agent.agent_id, current_answer = answer, current_explanation=explanation, case_history=case_history)
                    
                    # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
                    audit_results_of_unresolved_conflicts_during_review = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question=question, current_agent_id=agent.agent_id, current_answer = answer, current_explanation=explanation, case_history=case_history) 

                    audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                        "agent_id": agent.agent_id,
                        "specialty": agent.specialty.value,
                        "step": "review",
                        "audit_result": audit_results_of_domain_specific_knowledge_activation
                    })
                    audit_round_data["2_2_1_repetition_of_initial_views"].append({
                        "agent_id": agent.agent_id,
                        "specialty": agent.specialty.value,
                        "step": "review",
                        "audit_result": audit_results_repetition_of_initial_views
                    })
                    audit_round_data["2_2_2_unresolved_conflicts"].append({ 
                        "agent_id": agent.agent_id,
                        "specialty": agent.specialty.value,
                        "step": "review",
                        "audit_result": audit_results_of_unresolved_conflicts_during_review
                    })

                case_history["rounds"][-1]["reviews"].append({
                    "agent_id": agent.agent_id,
                    "specialty": agent.specialty.value,
                    "log": review_log
                })

                discussion_history.append({
                    "phase": DiscussionPhase.DISCUSSION.value, "round": round_num,
                    "agent_id": agent.agent_id, "model_name": agent.model_name,
                    "response": resp, "interaction_log": step_log
                })
                print(f"Agent {agent.agent_id} round {round_num} answer: {resp.get('answer', '')} (conf: {resp.get('confidence', 0.0):.2f})")
            if task == "audit":
                audit["rounds"].append(audit_round_data)
            current_answers = new_answers
            consensus_reached = self._check_consensus(current_answers)
            if consensus_reached:
                print("Consensus reached, ending discussion.")
                break

        # === Phase 3: Final team answer via weighted vote ===
        print("Phase 3: Generating final team answer")
        final_decision, voting_log = self._weighted_vote(current_answers)
        if task == "audit":
            case_history["audit"] = audit
        discussion_history.append({
            "phase": DiscussionPhase.FINAL.value,
            "final_decision": final_decision, "consensus_reached": consensus_reached,
            "rounds_completed": round_num, "final_round_agent_answers": current_answers,
            "voting_details": voting_log
        })

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Discussion completed in {processing_time:.2f} seconds. Final answer: {final_decision}")


        return {
            "final_decision": final_decision,
            "discussion_history": discussion_history,
            "processing_time": processing_time,
            "case_history": case_history
        }


def process_item(item: Dict[str, Any],
               agent_configs: List[Dict[str, str]],
               max_rounds: int = 3,
               auditor_model_key: str = "gemini-2.5-pro",
               conflict_analysis_model_key: str = "gemini-2.5-pro",
               config_path: str = "config.toml",
               task :str = "open_coding") -> Dict[str, Any]:
    """
    Process a single QA item with the Reconcile framework.
    """
    qid = item.get("qid", "unknown")
    print(f"Processing item {qid}")

    coordinator = ReconcileCoordinator(agent_configs, max_rounds, auditor_model_key, conflict_analysis_model_key, config_path = config_path)
    discussion_result = coordinator.run_discussion(
        qid=qid,
        question=item.get("question", ""),
        options=item.get("options"),
        image_path=item.get("image_path"),
        task = task
    )
    
    result = {
        "qid": qid,
        "timestamp": int(time.time()),
        "question": item.get("question", ""),
        "options": item.get("options"),
        "image_path": item.get("image_path"),
        "ground_truth": item.get("answer"),
        "predicted_answer": discussion_result["final_decision"],
        "case_history": discussion_result["case_history"],
    }
    return result


def main():
    """
    Main entry point for running the Reconcile framework with full observation logging.
    """
    parser = argparse.ArgumentParser(description="Run the Reconcile framework on medical QA datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--agents", nargs='+', required=True, help="List of agent model keys")
    parser.add_argument("--max_rounds", type=int, default=3, help="Maximum number of discussion rounds")
    parser.add_argument("--auditor_model", type=str, required=True, help="Model for the AuditorAgent and conflict agent.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to process.")
    parser.add_argument("--config_path", type=str, default="config.toml", help="Path to config file")
    parser.add_argument("--time_stamp", type=str, required=True, help="Timestamp for logging purposes")
    parser.add_argument("--task", type = str, required=True, help="audit or open_coding?")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")
    task = args.task
    print(f"Task: {task}")

    timestamp = args.time_stamp
    current_mas_name = current_file_name
    main_llm = args.agents[0]

    terminal_log_file = project_root / "logs" / f"{task}_results" / timestamp/ f"{current_mas_name}_{dataset_name}_{main_llm}_terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条

    data_path = project_root / "datasets" / "processed" / dataset_name / f"{task}" / f"medqa_{dataset_name.lower()}_{task}.json"
    
    output_file = project_root / "logs" / f"{task}_results" / timestamp / f"{current_mas_name}_{dataset_name}_{main_llm}.jsonl"
    error_output_file = project_root / "logs" / f"{task}_results" / timestamp / f"{current_mas_name}_{dataset_name}_{main_llm}_errors.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_output_file.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    agent_configs = [{"agent_id": f"agent_{idx}", "model_key": model_key} for idx, model_key in enumerate(args.agents, 1)]
    print(f"Configured {len(agent_configs)} agents: {[cfg['model_key'] for cfg in agent_configs]}")

    for item in tqdm(data[:args.num_samples], desc=f"Processing {args.dataset} ({args.task})"):
        qid = item.get("qid")

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

        if qid in existing_qids:
            print(f"Skipping {qid} - already processed")
            continue

        try:
            result = process_item(
                item,
                agent_configs,
                args.max_rounds,
                auditor_model_key=args.auditor_model,
                conflict_analysis_model_key=args.auditor_model, 
                config_path=args.config_path,
                task = task
            )
            save_jsonl(result, output_file)
        except Exception as e:
            error_traceback = traceback.format_exc()
            
            print(f"--- FATAL ERROR processing item {qid} ---", flush=True)
            print(error_traceback, flush=True)
            print(f"--- END OF ERROR for {qid} ---", flush=True)

            error_log = {
                "qid": qid,
                "error_type": str(type(e).__name__), # e.g., "IndexError"
                "error_message": str(e),            # e.g., "list index out of range"
                "traceback": error_traceback        # The full stack trace
            }
            save_jsonl(error_log, error_output_file)

if __name__ == "__main__":
    main()