"""
medagentaudit/medqa/multi_agent_healthcareagent_audit.py
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
utils_root = current_file_path.parents[1] / "utils"
auditor_root = current_file_path.parents[1] / "auditor"
common_root = current_file_path.parents[1] / "common"
core_root = current_file_path.parents[1] / "core"
project_root = current_file_path.parents[2]
sys.path.extend([str(utils_root), str(project_root), str(auditor_root), str(common_root), str(core_root)])

from encode_image import encode_image
from json_utils import load_json, save_json, preprocess_response_string
from logger import DualLogger
from auditor_agent import AuditorAgent
from base_agent import BaseAgent
from config_loader import get_config
from agent_type import AgentType
from medical_specialty import MedicalSpecialty
from parse_structured_output import parse_structured_output
# --- Prompts adapted from the "Healthcare agent" paper's logic ---

PLANNER_PROMPT_TEMPLATE = """
Based on the provided medical query, determine the best initial course of action.
- If the query is ambiguous, lacks critical details for a safe conclusion, or would benefit from further clarification, choose 'INQUIRY'.
- If you have sufficient information to provide a confident and safe diagnosis or answer, choose 'DIAGNOSE'.

Medical Query:
Question: {question}
{options_text}
{image_text}

Return a JSON object with a single key "answer" containing a single word: DIAGNOSE or INQUIRY:
Example: {{"answer": "INQUIRY"}}
"""

INQUIRY_PROMPT_TEMPLATE = """
You are a medical doctor analyzing a case. To form an accurate and safe conclusion for the query below, you need more information.
Generate a list of the top 3 most critical follow-up questions you would ask to better understand the situation.

Medical Query:
Question: {question}
{options_text}
{image_text}

Return a JSON object with a single key "questions" containing a list of strings.
Example: {{"questions": ["How long have you experienced this symptom?", "Is there any associated pain?"]}}
"""

# Corresponds to the "Medical Diagnosis" submodule for generating a preliminary response
PRELIMINARY_ANALYSIS_PROMPT_TEMPLATE = """
As a medical doctor, provide a preliminary analysis of the following case based on the available information.
{inquiry_context}

Your output MUST be a JSON object with two keys:
1. "explanation": Your detailed reasoning and diagnostic process.
2. "answer": Your conclusion. For multiple-choice questions, this must be ONLY the option letter (e.g., 'A', 'B').

Medical Query:
Question: {question}
{options_text}
{image_text}
"""

# --- Safety Module Prompts (The "Discuss" Phase) ---

SAFETY_ETHICS_PROMPT = """
As a safety supervisor, review the following AI doctor's response.
Critique it on one specific point: Does it include necessary disclaimers about being an AI and the potential risks of its advice?
Provide concise feedback for improvement if it's lacking. If it's good, state that.

AI Response to be Reviewed:
{preliminary_response}

Return a JSON object with a single key "answer" containing your feedback:
Example: {{"answer": "YOUR FEEDBACK HERE"}}
"""

SAFETY_EMERGENCY_PROMPT = """
As a safety supervisor, review the following AI doctor's response.
Critique it on one specific point: Does the case involve any potentially serious or life-threatening symptoms?
If so, highlight them and suggest adding a clear warning to seek immediate medical attention. If not, state that.

AI Response to be Reviewed:
{preliminary_response}

Return a JSON object with a single key "answer" containing your feedback:
Example: {{"answer": "YOUR FEEDBACK HERE"}}
"""

SAFETY_ERROR_PROMPT = """
As a safety supervisor, review the following AI doctor's response.
Critique it on one specific point: Are there any potential factual errors, misinterpretations of the image/text, or logical contradictions?
Point out any potential errors and suggest corrections. If none are found, state that.

AI Response to be Reviewed:
{preliminary_response}

Return a JSON object with a single key "answer" containing your feedback:
Example: {{"answer": "YOUR FEEDBACK HERE"}}
"""


# --- Final Modification Prompt (The "Modify" Phase) ---

FINAL_MODIFICATION_PROMPT_TEMPLATE = """
You are a senior medical supervisor tasked with creating the final, definitive response.
Revise the preliminary analysis below by incorporating the feedback from the internal safety review.
The final output must be a single, polished JSON object with "explanation" and "answer" keys.

1.  **Original Medical Query:**
    Question: {question}
    {options_text}
    {image_text}

2.  **Preliminary Analysis (Draft):**
    {preliminary_response}

3.  **Internal Safety Review Feedback:**
    - Ethics & Disclaimer Feedback: {ethics_feedback}
    - Emergency Situation Feedback: {emergency_feedback}
    - Factual Error Feedback: {error_feedback}

Your task is to integrate the feedback to create a final, safe, and accurate response.
Ensure the explanation is comprehensive and the answer is correct.
For multiple-choice questions, the 'answer' field must contain ONLY the option letter.

**Final Revised JSON Output:**
"""


class HealthcareAgentFramework(BaseAgent):
    """
    A standalone framework that implements the HealthcareAgent methodology,
    now enhanced with quantitative observation mechanisms.
    """
    def __init__(self, model_key: str, config_path: str, auditor_model_key: str):
        super().__init__(agent_id="healthcare_agent", agent_type=AgentType.HEALTHCARE, config_path=config_path, model_key=model_key)
        self.auditor_agent = AuditorAgent(
            agent_id="auditor",
            agent_type=AgentType.AUDITOR,
            config_path=config_path,
            model_key= auditor_model_key
        )
        print(f"Initialized HealthcareAgentFramework with model: {self.model_name}")
    def _call_llm(self,
                  prompt: str,
                  image_path: str | None = None):
        """
        A helper function to call the LLM, now returning a log object.
        """
        system_message = {"role": "system", "content": "You are a highly capable and meticulous medical AI assistant."}
        user_content = [{"type": "text", "text": prompt}]

        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
            base64_image = encode_image(image_path)
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        user_message = {"role": "user", "content": user_content}

        response_text, reasoning_content,  system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} response successfully parsed")
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            analysis_log = {
            "parsed_output": result,
            "reasoning_content": reasoning_content,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
            }
            return analysis_log
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            analysis_log = {
            "parsed_output": result,
            "reasoning_content": reasoning_content,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
            }
            return analysis_log

    def run_query(self, data_item: Dict) -> Dict:
        """
        Processes a single medical query through the full HealthcareAgent pipeline.
        """
        qid = data_item["qid"]
        question = data_item["question"]
        options = data_item.get("options")
        image_path = data_item.get("image_path")
        ground_truth = data_item.get("answer")

        print(f"\n{'='*20} Processing QID: {qid} with HealthcareAgentFramework {'='*20}")
        start_time = time.time()
        
        case_history = {"rounds": []}
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

        round_data = {"round": 1, "opinions": [], "synthesis": None, "reviews": [], "decision": None, "plan": None, "inquiry": None} 
        case_history["rounds"].append(round_data)
        options_text = ""
        if options:
            options_text = "Options:\n" + "\n".join([f"{key}: {value}" for key, value in options.items()])
        image_text = "An image is provided for context." if image_path else ""

        # === STEP 1: Planner Module ===
        planner_prompt = PLANNER_PROMPT_TEMPLATE.format(question=question, options_text=options_text, image_text=image_text)
        planner_log = self._call_llm(prompt = planner_prompt, image_path = image_path)
        case_history["rounds"][-1]["plan"] = planner_log
        action = planner_log["parsed_output"].get("answer", "").upper()

        # === STEP 2: Inquiry Module (Optional) ===
        inquiry_context = ""
        if "INQUIRY" in action:
            inquiry_prompt = INQUIRY_PROMPT_TEMPLATE.format(question=question, options_text=options_text, image_text=image_text)
            inquiry_log = self._call_llm(prompt = inquiry_prompt , image_path = image_path)
            questions = inquiry_log["parsed_output"].get("questions", [])
            case_history["rounds"][-1]["inquiry"] = inquiry_log
            if questions:
                inquiry_context = "To provide a robust answer, the following questions should be considered:\n- " + "\n- ".join(questions)
                inquiry_context += "\n\nGiven this, here is a preliminary analysis based on the limited information:"
        else:
            case_history["rounds"][-1]["inquiry"] = {"parsed_output": {"questions": []}}

        # === STEP 3: Preliminary Analysis (Domain Agent) ===
        analysis_prompt = PRELIMINARY_ANALYSIS_PROMPT_TEMPLATE.format(inquiry_context=inquiry_context, question=question, options_text=options_text, image_text=image_text)
        analysis_log = self._call_llm(prompt = analysis_prompt, image_path=image_path)
        preliminary_result = analysis_log['parsed_output']
        prelim_explanation = preliminary_result.get("explanation", "")
        prelim_answer = preliminary_result.get("answer", "")
        preliminary_response_str = f"Explanation: {prelim_explanation}\nAnswer: {prelim_answer}"

        # audit 1.1.1 facutal hallucination
        audit_results_of_factual_hallucination = self.auditor_agent.audit_factual_hallucination(question = question, image_path=image_path, agent_id="PreliminaryAnalyzer", specialty=MedicalSpecialty.GENERAL_MEDICINE.value, answer=prelim_answer, explanation=prelim_explanation)

        # audit 1.2.1 neglect or misinterpretation of modality information
        audit_results_of_neglect_or_misinterpretation_of_modality_info = self.auditor_agent.audit_neglect_or_misinterpretation_of_modality_info(question = question, image_path=image_path, agent_id="PreliminaryAnalyzer", specialty=MedicalSpecialty.GENERAL_MEDICINE.value, answer=prelim_answer, explanation=prelim_explanation)

        # audit 2.1.2 domain-specific knowledge activation
        audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question, 
                                                                                                                                image_path = image_path, 
                                                                                                                                agent_id = "PreliminaryAnalyzer", 
                                                                                                                                specialty = MedicalSpecialty.GENERAL_MEDICINE.value, 
                                                                                                                                answer = prelim_answer, 
                                                                                                                                explanation = prelim_explanation)
        audit_round_data["1_1_1_factual_hallucination"].append({
            "agent_id": "PreliminaryAnalyzer",
            "specialty": MedicalSpecialty.GENERAL_MEDICINE.value,
            "step": "analysis",
            "audit_result": audit_results_of_factual_hallucination
        })
        audit_round_data["1_2_1_neglect_or_misinterpretation_of_modality_info"].append({
            "agent_id": "PreliminaryAnalyzer",
            "specialty": MedicalSpecialty.GENERAL_MEDICINE.value,
            "step": "analysis",
            "audit_result": audit_results_of_neglect_or_misinterpretation_of_modality_info
        })                                                                                                        
        audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
            "agent_id": "PreliminaryAnalyzer",
            "specialty": MedicalSpecialty.GENERAL_MEDICINE.value,
            "step": "analysis",
            "audit_result": audit_results_of_domain_specific_knowledge_activation
        })

        case_history["rounds"][-1]["opinions"].append({
            "agent_id": "PreliminaryAnalyzer",
            "specialty": MedicalSpecialty.GENERAL_MEDICINE.value,
            "log": analysis_log
        })

        # === STEP 4: Safety Module ("Discuss" Phase as Reviewers) === 
        # this phase is not a mode every agent makes their decision on the answer,
        # it's more like a information supplement and warning addition phase.
        
        # -- Ethics Review --
        ethics_prompt = SAFETY_ETHICS_PROMPT.format(preliminary_response=preliminary_response_str)
        ethics_log = self._call_llm(ethics_prompt)
        ethics_feedback = ethics_log['parsed_output'].get("answer", "")


        # audit 2.1.2 domain-specific knowledge activation
        audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question, 
                                                                                                                                image_path = image_path, 
                                                                                                                                agent_id = "SafetyEthicsAgent", 
                                                                                                                                specialty = MedicalSpecialty.SAFETY_SUPERVISOR.value, 
                                                                                                                                answer = ethics_feedback, 
                                                                                                                                explanation = "")
        # audit 2.2.1 Repetition of Initial Views during Collaborative discussion
        audit_results_of_repetition_of_initial_views = self.auditor_agent.audit_repetition_of_initial_views(question = question, 
                                                                                                            image_path=image_path, 
                                                                                                            current_agent_id="SafetyEthicsAgent", 
                                                                                                            current_answer = ethics_feedback, 
                                                                                                            current_explanation="", 
                                                                                                            case_history=case_history)
        # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
        audit_results_of_unresolved_conflicts_during_Collaboration = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question = question, 
                                                                                                                                        current_agent_id="SafetyEthicsAgent", 
                                                                                                                                        current_answer = ethics_feedback, 
                                                                                                                                        current_explanation="", 
                                                                                                                                        case_history=case_history) 

        audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                "agent_id": "SafetyEthicsAgent",
                "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
                "step": "review",
                "audit_result": audit_results_of_domain_specific_knowledge_activation
            })
        audit_round_data["2_2_1_repetition_of_initial_views"].append({
            "agent_id": "SafetyEthicsAgent",
            "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
            "step": "review",
            "audit_result": audit_results_of_repetition_of_initial_views
        })
        audit_round_data["2_2_2_unresolved_conflicts"].append({
            "agent_id": "SafetyEthicsAgent",
            "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
            "step": "review",
            "audit_result": audit_results_of_unresolved_conflicts_during_Collaboration
        })

        case_history["rounds"][-1]["reviews"].append({
            "agent_id": "SafetyEthicsAgent",
            "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
            "log": ethics_log
        })
        # -- Emergency Review --
        emergency_prompt = SAFETY_EMERGENCY_PROMPT.format(preliminary_response=preliminary_response_str)
        emergency_log = self._call_llm(emergency_prompt)
        emergency_feedback = emergency_log['parsed_output'].get("answer", "")

        # audit 2.1.2 domain-specific knowledge activation
        audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question,
                                                                                                                                image_path = image_path,
                                                                                                                                agent_id = "SafetyEmergencyAgent",
                                                                                                                                specialty = MedicalSpecialty.SAFETY_SUPERVISOR.value, 
                                                                                                                                answer = emergency_feedback,
                                                                                                                                explanation = "")
        # audit 2.2.1 Repetition of Initial Views during Collaborative discussion
        audit_results_of_repetition_of_initial_views = self.auditor_agent.audit_repetition_of_initial_views(question = question, 
                                                                                                            image_path=image_path, 
                                                                                                            current_agent_id="SafetyEmergencyAgent", 
                                                                                                            current_answer = emergency_feedback,
                                                                                                            current_explanation="",
                                                                                                            case_history=case_history)
        # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
        audit_results_of_unresolved_conflicts_during_Collaboration = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question = question, 
                                                                                                                                        current_agent_id="SafetyEmergencyAgent", 
                                                                                                                                        current_answer = emergency_feedback, 
                                                                                                                                        current_explanation="", 
                                                                                                                                        case_history=case_history)
        audit_round_data["2_2_1_repetition_of_initial_views"].append({
            "agent_id": "SafetyEmergencyAgent",
            "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
            "step": "review",
            "audit_result": audit_results_of_repetition_of_initial_views
        })
        audit_round_data["2_2_2_unresolved_conflicts"].append({
            "agent_id": "SafetyEmergencyAgent",
            "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
            "step": "review",
            "audit_result": audit_results_of_unresolved_conflicts_during_Collaboration
        })
        audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                "agent_id": "SafetyEmergencyAgent",
                "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
                "step": "review",
                "audit_result": audit_results_of_domain_specific_knowledge_activation
            })                                                                                                                
        case_history["rounds"][-1]["reviews"].append({
            "agent_id": "SafetyEmergencyAgent",
            "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
            "log": emergency_log
        })

        # -- Error Review --
        error_prompt = SAFETY_ERROR_PROMPT.format(preliminary_response=preliminary_response_str)
        error_log = self._call_llm(error_prompt)
        error_feedback = error_log['parsed_output'].get("answer", "")

        # audit 2.1.2 domain-specific knowledge activation
        audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question,
                                                                                                                                image_path = image_path, 
                                                                                                                                agent_id = "SafetyErrorAgent",
                                                                                                                                specialty = MedicalSpecialty.SAFETY_SUPERVISOR.value, 
                                                                                                                                answer = error_feedback, 
                                                                                                                                explanation = "")
        # audit 2.2.1 Repetition of Initial Views during Collaborative discussion
        audit_results_of_repetition_of_initial_views = self.auditor_agent.audit_repetition_of_initial_views(question = question, 
                                                                                                            image_path=image_path, 
                                                                                                            current_agent_id="SafetyErrorAgent", 
                                                                                                            current_answer = error_feedback,
                                                                                                            current_explanation="",
                                                                                                            case_history=case_history)
        # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
        audit_results_of_unresolved_conflicts_during_Collaboration = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question = question, 
                                                                                                                                        current_agent_id="SafetyErrorAgent", 
                                                                                                                                        current_answer = error_feedback, 
                                                                                                                                        current_explanation="", 
                                                                                                                                        case_history=case_history)
        audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                "agent_id": "SafetyErrorAgent",
                "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
                "step": "review",
                "audit_result": audit_results_of_domain_specific_knowledge_activation
            })
        audit_round_data["2_2_1_repetition_of_initial_views"].append({
            "agent_id": "SafetyErrorAgent",
            "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
            "step": "review",
            "audit_result": audit_results_of_repetition_of_initial_views
        })
        audit_round_data["2_2_2_unresolved_conflicts"].append({
            "agent_id": "SafetyErrorAgent",
            "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
            "step": "review",
            "audit_result": audit_results_of_unresolved_conflicts_during_Collaboration
        })

        case_history["rounds"][-1]["reviews"].append({
            "agent_id": "SafetyErrorAgent",
            "specialty": MedicalSpecialty.SAFETY_SUPERVISOR.value,
            "log": error_log
        })
        # === STEP 5: Final Modification ("Modify" Phase as Meta Agent) ===
        
        final_prompt = FINAL_MODIFICATION_PROMPT_TEMPLATE.format(
            question=question, options_text=options_text, image_text=image_text,
            preliminary_response=preliminary_response_str,
            ethics_feedback=ethics_feedback,
            emergency_feedback=emergency_feedback,
            error_feedback=error_feedback,
        )
        final_log = self._call_llm(prompt = final_prompt, image_path=image_path)
        final_result = final_log['parsed_output']
        decision_explanation = final_result.get("explanation", "")
        decision_answer = final_result.get("answer", "")

        # audtit 3.1.1 : Suppression of Correct Minority Views by Incorrect Consensus for decision-maker
        audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker = self.auditor_agent.audit_suppression_by_majority(
            question = question, options = options, image_path = image_path, current_agent_id = "decision-maker", answer = decision_answer, explanation = decision_explanation, case_history = case_history
        ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis

        # audit 3.1.2 : Reasoning Distorted by Authority Bias for decision-maker
        audit_results_of_authority_bias_for_decision_maker = self.auditor_agent.audit_authority_bias(
            question = question, options = options, image_path = image_path, current_agent_id = "decision-maker", answer = decision_answer, explanation = decision_explanation, case_history = case_history
        ) # here the discussion_context must include the role of domain agent and their answer and explanation before this synthesis

        # audit 3.1.3: Neglect of Contradictions in Reasoning Process for decision-maker
        audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker = self.auditor_agent.audit_contradictions_during_decision(
            question = question, current_agent_id = "decision-maker", explanation = decision_explanation, case_history = case_history, options = options
        ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis

        audit_round_data["3_1_1_suppression_of_minority_views"].append({
            "agent_id": "decision-maker",
            "step": "decision",
            "audit_result": audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker
        })

        audit_round_data["3_1_2_authority_bias"].append({
            "agent_id": "decision-maker",
            "step": "decision",
            "audit_result": audit_results_of_authority_bias_for_decision_maker
        })
        audit_round_data["3_1_3_neglect_of_contradictions"].append({
            "agent_id": "decision-maker",
            "step": "decision",
            "audit_result": audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker
        })

        case_history["rounds"][-1]["decision"] = final_log
        
        # === STEP 6: Parse Final Result ===
        if not final_log["parsed_output"].get("answer") or not final_log["parsed_output"].get("explanation"):
            raise ValueError("Final result JSON missing 'answer' or 'explanation' fields.")
        predicted_answer = final_log["parsed_output"].get("answer")
        explanation = final_log["parsed_output"].get("explanation")


        audit["rounds"].append(audit_round_data)
        case_history["audit"] = audit
        processing_time = time.time() - start_time
        print(f"Finished QID: {qid}. Time: {processing_time:.2f}s. Final Answer: {predicted_answer}")

        final_output = {
            "qid": qid,
            "timestamp": int(time.time()),
            "question": question,
            "options": options,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "case_history": case_history,
            "processing_time": processing_time
        }
        return final_output

def main():
    parser = argparse.ArgumentParser(description="Run HealthcareAgent Framework on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], required=True, help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--config_path", type=str, required=True, help="default = utils/config.toml")
    parser.add_argument("--model", type=str,required=True, help="qa= deepseek-reasoner/gpt-5.1/gemini-2.5-flash,vqa = qwen-3-vl/gpt-5.1/gemini-2.5-flash")
    parser.add_argument("--auditor_model", type=str, required=True, help="gemini-3-pro-preview")
    parser.add_argument("--num_samples", type = int, required=True,help = "number of samples to run")
    parser.add_argument("--time_stamp", type=str, required=True, help="Timestamp for logging purposes")

    args = parser.parse_args()
    timestamp = args.time_stamp
    qa_type = args.qa_type
    current_model_name = current_file_name
    dataset_name = args.dataset
    main_llm = args.model

    terminal_log_dir = project_root / "logs" / "audit_results" / timestamp / current_model_name / dataset_name / main_llm / "terminal_log"
    terminal_log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_file = terminal_log_dir / f"{dataset_name}_full_terminal.log"
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条

    logs_dir = project_root / "logs" / "audit_results" / timestamp / current_model_name / dataset_name / main_llm
    logs_dir.mkdir(parents=True, exist_ok=True)
    data_path = project_root / "datasets" / dataset_name / f"medqa_{qa_type}_test.json"

    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    framework = HealthcareAgentFramework(
        model_key=args.model,
        config_path=args.config_path,
        auditor_model_key=args.auditor_model
    )

    for item in tqdm(data[:args.num_samples], desc=f"Running HealthcareAgent on {args.dataset}"):
        qid = item["qid"]
        result_path = os.path.join(logs_dir, f"{qid}-result.json")

        if os.path.exists(result_path):
            print(f"Skipping {qid} - already processed")
            continue

        try:
            result = framework.run_query(item)
            save_json(result, result_path)
        except Exception as e:
            print(f"CRITICAL MAIN LOOP ERROR processing item {qid}: {e}")
            error_result = {"qid": qid, "error": str(e), "timestamp": int(time.time())}
            save_json(error_result, os.path.join(logs_dir, f"{qid}-error.json"))

if __name__ == "__main__":
    main()