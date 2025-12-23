"""
medagentboard/medqa/multi_agent_healthcareagent_audit.py
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

current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
utils_root = current_file_path.parents[1] / "utils"
project_root = current_file_path.parents[2]
sys.path.extend([str(utils_root), str(project_root)])

from encode_image import encode_image
from json_utils import load_json, save_json, preprocess_response_string
from dual_logger import DualLogger
from auditor_agent import AuditorAgent
from BaseAgent import BaseAgent
from config import get_config
from agent_type import AgentType
from medical_specialty import MedicalSpecialty
from parse_structured_output import parse_structured_output
# --- Prompts adapted from the "Healthcare agent" paper's logic ---

PLANNER_PROMPT_TEMPLATE = """
Based on the provided medical query, determine the best initial course of action.
- If the query is ambiguous, lacks critical details for a safe conclusion, or would benefit from further clarification, choose 'INQUIRY'.
- If you have sufficient information to provide a confident and safe diagnosis or answer, choose 'DIAGOOSE'.

Medical Query:
Question: {question}
{options_text}
{image_text}

Respond with a single word: DIAGNOSE or INQUIRY.
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

Your Feedback:
"""

SAFETY_EMERGENCY_PROMPT = """
As a safety supervisor, review the following AI doctor's response.
Critique it on one specific point: Does the case involve any potentially serious or life-threatening symptoms?
If so, highlight them and suggest adding a clear warning to seek immediate medical attention. If not, state that.

AI Response to be Reviewed:
{preliminary_response}

Your Feedback:
"""

SAFETY_ERROR_PROMPT = """
As a safety supervisor, review the following AI doctor's response.
Critique it on one specific point: Are there any potential factual errors, misinterpretations of the image/text, or logical contradictions?
Point out any potential errors and suggest corrections. If none are found, state that.

AI Response to be Reviewed:
{preliminary_response}

Your Feedback:
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
    def __init__(self, model_key: str, auditor_model_key: str, conflict_model_key: str,config_path: str):
        super().__init__(agent_id="healthcare_agent", agent_type=AgentType.DOMAIN, config_path=config_path, model_key=model_key) # TODO, this agent type need to be specified!
        print(f"Initialized HealthcareAgentFramework with model: {self.model_name}")
    def _call_llm(self,
                  prompt: str,
                  image_path: str | None):
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

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

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

        options_text = ""
        if options:
            options_text = "Options:\n" + "\n".join([f"{key}: {value}" for key, value in options.items()])
        image_text = "An image is provided for context." if image_path else ""

        try:
            # === STEP 1: Planner Module ===
            planner_prompt = PLANNER_PROMPT_TEMPLATE.format(question=question, options_text=options_text, image_text=image_text)
            action, planner_log = self._call_llm(planner_prompt, "Planner", image_path, expect_json=True)
            action = action.strip().upper()
            case_history["steps"].append({"step": "1_Planner", "log": planner_log, "decision": action})

            # === STEP 2: Inquiry Module (Optional) ===
            inquiry_context = ""
            if "INQUIRY" in action:
                inquiry_prompt = INQUIRY_PROMPT_TEMPLATE.format(question=question, options_text=options_text, image_text=image_text)
                inquiry_response_str, inquiry_log = self._call_llm(inquiry_prompt, "InquiryAgent", image_path, expect_json=True)
                inquiry_result = json.loads(preprocess_response_string(inquiry_response_str))
                questions = inquiry_result.get("questions", [])
                case_history["steps"].append({"step": "2_Inquiry", "log": inquiry_log, "generated_questions": questions})
                if questions:
                    inquiry_context = "To provide a robust answer, the following questions should be considered:\n- " + "\n- ".join(questions)
                    inquiry_context += "\n\nGiven this, here is a preliminary analysis based on the limited information:"
            else:
                 case_history["steps"].append({"step": "2_Inquiry", "generated_questions": "Skipped as per planner's decision."})

            # === STEP 3: Preliminary Analysis (Domain Agent) ===
            analysis_prompt = PRELIMINARY_ANALYSIS_PROMPT_TEMPLATE.format(inquiry_context=inquiry_context, question=question, options_text=options_text, image_text=image_text)
            preliminary_response_str, analysis_log = self._call_llm(analysis_prompt, "PreliminaryAnalyzer", image_path, expect_json=True)
            preliminary_result = json.loads(preprocess_response_string(preliminary_response_str))
            
            
            all_keus_for_audit = [{"keu_id": k, "content": v.content} for k,v in audit_trail["keus"].items()]
            if all_keus_for_audit:
                key_status_map = self.auditor_agent.identify_key_evidential_units(question, all_keus_for_audit)
                for keu_id, is_key in key_status_map.items():
                    if keu_id in audit_trail["keus"]:
                        audit_trail["keus"][keu_id].is_key = is_key

            # Mechanism 3: Audit the contribution
            prelim_explanation = preliminary_result.get("explanation", "")
            contribution_audit = self.auditor_agent.audit_domain_agent_contribution(question, "PreliminaryAnalyzer", MedicalSpecialty.GENERAL_MEDICINE, prelim_explanation)
            risk_audit = self.auditor_agent.audit_risk_and_quality("PreliminaryAnalyzer", prelim_explanation, image_path)
            audit_trail["collaboration_audits"]["preliminary_analysis"] = {**contribution_audit, **risk_audit}
            # --- END: Mechanism 1 & 3 Auditing ---
            
            analysis_log['parsed_output'] = preliminary_result
            case_history["steps"].append({"step": "3_Preliminary_Analysis", "log": analysis_log})

            # === STEP 4: Safety Module ("Discuss" Phase as Reviewers) === 
            # this phase is not a mode every agent makes their decision on the answer,
            # it's more like a information supplement and warning addition phase.
            keu_list_text = "\n".join([f"- {k}: '{v.content}'" for k, v in audit_trail["keus"].items()]) or "No KEUs were extracted."
            ccp_text = "No conflicts identified yet." # No conflicts before review
            
            # -- Ethics Review --
            ethics_prompt = SAFETY_ETHICS_PROMPT.format(preliminary_response=preliminary_response_str, keu_list_text=keu_list_text, ccp_text=ccp_text)
            ethics_feedback_str, ethics_log = self._call_llm(ethics_prompt, "SafetyEthicsAgent", expect_json=True)
            ethics_feedback = json.loads(preprocess_response_string(ethics_feedback_str)).get("feedback", "")
            risk_audit_ethics = self.auditor_agent.audit_risk_and_quality("SafetyEthicsAgent", ethics_feedback, image_path)
            audit_trail["collaboration_audits"]["ethics_review"] = risk_audit_ethics

            # -- Emergency Review --
            emergency_prompt = SAFETY_EMERGENCY_PROMPT.format(preliminary_response=preliminary_response_str, keu_list_text=keu_list_text, ccp_text=ccp_text)
            emergency_feedback_str, emergency_log = self._call_llm(emergency_prompt, "SafetyEmergencyAgent", expect_json=True)
            emergency_feedback = json.loads(preprocess_response_string(emergency_feedback_str)).get("feedback", "")
            risk_audit_emergency = self.auditor_agent.audit_risk_and_quality("SafetyEmergencyAgent", emergency_feedback, image_path)
            audit_trail["collaboration_audits"]["emergency_review"] = risk_audit_emergency

            # -- Error Review --
            error_prompt = SAFETY_ERROR_PROMPT.format(preliminary_response=preliminary_response_str, keu_list_text=keu_list_text, ccp_text=ccp_text)
            error_feedback_str, error_log = self._call_llm(error_prompt, "SafetyErrorAgent", expect_json=True)
            error_feedback = json.loads(preprocess_response_string(error_feedback_str)).get("feedback", "")
            risk_audit_error = self.auditor_agent.audit_risk_and_quality("SafetyErrorAgent", error_feedback, image_path)
            audit_trail["collaboration_audits"]["error_review"] = risk_audit_error
            
            case_history["steps"].append({
                "step": "4_Safety_Review",
                "ethics_feedback": {"log": ethics_log, "feedback": ethics_feedback},
                "emergency_feedback": {"log": emergency_log, "feedback": emergency_feedback},
                "error_feedback": {"log": error_log, "feedback": error_feedback}
            })
            
            # --- START: Mechanism 4: Identify conflicts after review ---
            review_contributions = [
                {'agent_id': 'PreliminaryAnalyzer', 'specialty': MedicalSpecialty.GENERAL_MEDICINE.value, 'text': prelim_explanation},
                {'agent_id': 'SafetyEthicsAgent', 'specialty': MedicalSpecialty.SAFETY_ETHICS.value, 'text': ethics_feedback},
                {'agent_id': 'SafetyEmergencyAgent', 'specialty': MedicalSpecialty.EMERGENCY_MEDICINE.value, 'text': emergency_feedback},
                {'agent_id': 'SafetyErrorAgent', 'specialty': MedicalSpecialty.FACTUAL_ACCURACY.value, 'text': error_feedback}
            ]
            new_ccps = self.auditor_agent.identify_critical_conflicts(review_contributions, "preliminary analysis vs. safety reviews")
            audit_trail["ccps"]["round_1"] = []
            for ccp in new_ccps:
                ccp['ccp_id'] = f"CCP-{ccp_counter}"
                ccp['status'] = 'unresolved'
                audit_trail["ccps"]["round_1"].append(ccp)
                ccp_counter += 1
            all_unresolved_ccps.extend(audit_trail["ccps"]["round_1"])
            # --- END: Mechanism 4 ---
            

            # === STEP 5: Final Modification ("Modify" Phase as Meta Agent) ===
            ccp_text_for_prompt = "\n".join([f"- {c['ccp_id']}: {c['conflict_summary']}" for c in all_unresolved_ccps]) or "No critical conflicts identified."
            
            final_prompt = FINAL_MODIFICATION_PROMPT_TEMPLATE.format(
                question=question, options_text=options_text, image_text=image_text,
                preliminary_response=preliminary_response_str,
                ethics_feedback=ethics_feedback,
                emergency_feedback=emergency_feedback,
                error_feedback=error_feedback,
                keu_list_text=keu_list_text,
                ccp_text=ccp_text_for_prompt
            )
            final_response_str, final_log = self._call_llm(final_prompt, "FinalModifier", image_path, expect_json=True)
            
            # --- START: Mechanism 1, 3, 4 Auditing on Final Step ---
            final_result_json = json.loads(preprocess_response_string(final_response_str))
            final_explanation = final_result_json.get("explanation", "")
            
            # Mechanism 1: Track KEU presence in final decision
            for keu_id, keu in audit_trail["keus"].items():
                if keu_id in final_explanation or keu.content in final_explanation:
                    keu.present_in_final_decision = True
            
            # Mechanism 3: Audit final decision quality and risk
            quality_audit_final = self.auditor_agent.audit_single_argument_quality(question, final_explanation)
            risk_audit_final = self.auditor_agent.audit_risk_and_quality("FinalModifier", final_explanation, image_path=image_path)
            audit_trail["collaboration_audits"]["final_decision"] = {**quality_audit_final, **risk_audit_final}

            # Mechanism 4: Check if conflicts were resolved
            for ccp in all_unresolved_ccps:
                was_addressed, reasoning = self.analysis_llm.check_if_conflict_was_addressed(ccp, final_explanation)
                if was_addressed:
                    ccp['status'] = 'resolved'
                    ccp['resolution_reasoning'] = reasoning
            # --- END: Auditing Final Step ---
            
            final_log['parsed_output'] = final_result_json
            case_history["steps"].append({"step": "5_Final_Modification", "log": final_log})
            
            # === STEP 6: Parse Final Result ===
            if not final_result_json.get("answer") or not final_result_json.get("explanation"):
                raise ValueError("Final result JSON missing 'answer' or 'explanation' fields.")
            predicted_answer = final_result_json.get("answer")
            explanation = final_result_json.get("explanation")

        except Exception as e:
            print(f"FATAL ERROR during query processing for QID {qid}: {e}")
            predicted_answer = "Framework Error"
            explanation = str(e)
            case_history["error"] = str(e)

        processing_time = time.time() - start_time
        print(f"Finished QID: {qid}. Time: {processing_time:.2f}s. Final Answer: {predicted_answer}")

        # Serialize KEU objects before saving
        if "keus" in audit_trail and audit_trail["keus"]:
            audit_trail["keus"] = {k: v.to_dict() for k, v in audit_trail["keus"].items()}

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
    current_model_name = current_file_name.split("_")[2]

    terminal_log_dir = project_root / "logs" / "observation" / timestamp / current_model_name / args.dataset / "terminal_log"
    terminal_log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_file = terminal_log_dir / f"{args.dataset}__full_terminal.log"
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    logs_dir = project_root / "logs" / "observation" / current_model_name / args.dataset
    logs_dir.mkdir(parents=True, exist_ok=True)
    data_path = f"./my_datasets/processed/medqa/{args.dataset}/medqa_{args.qa_type}_test.json"

    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    framework = HealthcareAgentFramework(
        model_key=args.model,
        auditor_model_key=args.auditor_model,
        config_path=args.config_path
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