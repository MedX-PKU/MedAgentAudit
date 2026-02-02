"""
./medagentaudit/framework/medagent.py
"""

from openai import OpenAI
import os
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
sys.path.extend([str(utils_root), str(project_root), str(auditor_root), str(common_root), str(core_root)])

from encode_image import encode_image
from json_utils import load_json, save_jsonl, preprocess_response_string
from logger import DualLogger
from config_loader import get_config
from auditor_agent import AuditorAgent
from base_agent import BaseAgent
from agent_type import AgentType
from medical_specialty import MedicalSpecialty
from parse_structured_output import parse_structured_output


class ExpertGathererAgent(BaseAgent):
    """Agent responsible for gathering domain experts based on medical questions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-vl-max", config_path: str = "config.toml"):
        super().__init__(agent_id=agent_id, agent_type=AgentType.EXPERT_GATHERER, model_key=model_key, config_path=config_path)
        print(f"Initializing expert gatherer agent, ID: {agent_id}, Model: {model_key}")

    def gather_question_domain_experts(self, question: str) -> Tuple[List[MedicalSpecialty], Dict[str, Any]]:
        """
        Gather relevant domain experts for a medical question.

        Args:
            question: Medical question to analyze

        Returns:
            A tuple containing:
            - List of MedicalSpecialty enums representing relevant experts
            - The full log of the LLM call
        """
        print(f"Expert gatherer {self.agent_id} gathering question domain experts")
        
        available_specialties = ", ".join([s.value for s in MedicalSpecialty])

        system_message = {
            "role": "system",
            "content": f"""You are a medical expert who specializes in categorizing a specific medical scenario into specific areas of medicine.
Your task is to complete the following steps:
1. Carefully read the medical scenario presented in the question.
2. Based on the medical scenario, identify the three most relevant and distinct medical specialties from the provided list to form a multidisciplinary team.
3. Your output MUST be a JSON object with a single key 'fields', which is a list containing exactly three specialty names as strings.
Available Specialties: {available_specialties}"""
        }

        user_message = {
            "role": "user",
            "content": f"Review this medical question and determine the three most appropriate medical specialties required to provide the answer. Ensure the selected specialties are distinct and cover different aspects of the problem:\n\n{question}"
        }

        response_text, reasoning_content, system_msg, user_msg = self.call_llm(system_message = system_message, user_message = user_message)
        log = {
            "reasoning_content": reasoning_content,
            "llm_input": {"system_message": system_msg, "user_message": user_msg},
            "raw_output": response_text
        }
        
        try:
            result = json.loads(preprocess_response_string(response_text))
            log["parsed_output"] = result
            specialties_str = result.get("fields", [])
            
            valid_specialties = []
            for spec_str in specialties_str:
                try:
                    # 精确匹配Enum
                    matched_enum = next(s for s in MedicalSpecialty if s.value.lower() == spec_str.lower().strip())
                    valid_specialties.append(matched_enum)
                except StopIteration:
                    print(f"Warning: Could not map specialty '{spec_str}' to known MedicalSpecialty enum.")

            # 如果识别不足3个，则进行补充
            if len(valid_specialties) < 3:
                print(f"Warning: Only {len(valid_specialties)} specialties identified. Adding defaults.")
                defaults = [
                    MedicalSpecialty.INTERNAL_MEDICINE,
                    MedicalSpecialty.RADIOLOGY,
                    MedicalSpecialty.SURGERY,
                    MedicalSpecialty.PEDIATRICS,
                    MedicalSpecialty.CARDIOLOGY
                ]
                for default_spec in defaults:
                    if default_spec not in valid_specialties:
                        valid_specialties.append(default_spec)
                    if len(valid_specialties) >= 3:
                        break
            
            return valid_specialties[:3], log

        except json.JSONDecodeError:
            print("Expert gatherer response is not valid JSON, using fallback specialties")
            fallback_specialties = [MedicalSpecialty.INTERNAL_MEDICINE, MedicalSpecialty.RADIOLOGY, MedicalSpecialty.PEDIATRICS]
            return fallback_specialties, log


class DoctorAgent(BaseAgent):
    """Doctor agent with a medical specialty."""

    def __init__(self,
                 agent_id: str,
                 specialty: MedicalSpecialty,
                 model_key: str = "qwen-vl-max", config_path: str = "config.toml"):
        super().__init__(agent_id=agent_id, agent_type=AgentType.DOCTOR, model_key=model_key, config_path=config_path)
        self.specialty = specialty
        print(f"Initializing {specialty.value} doctor agent, ID: {agent_id}, Model: {model_key}")

    def analyze_case(self,
                     question: str,
                     options: Optional[Dict[str, str]] = None,
                     image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a medical case. Returns a full log dictionary.
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) analyzing case with model: {self.model_key}")
        
        system_message = {
            "role": "system",
            "content": f"""You are a doctor specializing in {self.specialty.value}.
Analyze the medical case and provide your professional opinion on the question. 
Your output must be a JSON object with three fields: 
1. 'explanation': Your detailed reasoning.
2. 'answer': Your final conclusion. For multiple-choice questions, this MUST be the single option letter (e.g., 'A', 'B')."""}

        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        options_text = ""
        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
        
        question_with_options = f"{question}{options_text}"

        user_content.append({
            "type": "text",
            "text": f"{question_with_options}\n\nProvide your analysis in the specified JSON format."
        })

        user_message = {"role": "user", "content": user_content}
        response_text, reasoning_content, system_msg, user_msg = self.call_llm(system_message = system_message, user_message = user_message)
        user_msg["content"] = [item for item in user_msg["content"] if item.get("type") != "image_url"]

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} response successfully parsed")
        except json.JSONDecodeError:
            print(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
        
        return {"parsed_output": result, "llm_input": {"system_message": system_msg, "user_message": user_msg}, "reasoning_content": reasoning_content}

    def review_synthesis(self,
                         question: str,
                         synthesis: Dict[str, Any],
                         options: Optional[Dict[str, str]] = None,
                         image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Review the meta agent's synthesis. Returns a full log dictionary.
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) reviewing synthesis with model: {self.model_key}")
        
        system_message = {
            "role": "system",
            "content": f"You are a doctor specializing in {self.specialty.value}, participating in a multidisciplinary team consultation. "
                      f"Review the synthesis report and determine if you agree with it. "
                      f"Your output should be in JSON format, including 'agree' (yes/no), 'reason' (rationale for your decision), "
                      f"and 'answer' (your suggested answer if you disagree) fields."
        }

        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        text_content = {
            "type": "text",
            "text": f"Synthesized report:\n{synthesis.get('reason', '')}\n\n"
                    f"Do you agree with this synthesized report? Provide your response in JSON format with the following fields:\n"
                    f"1. 'agree': 'yes' or 'no'\n"
                    f"2. 'reason': Your rationale for agreeing or disagreeing\n"
                    f"3. 'answer': If you disagree, provide your suggested answer"
        }
        user_content.append(text_content)

        user_message = {"role": "user", "content": user_content}
        response_text, reasoning_content, system_msg, user_msg = self.call_llm(system_message = system_message, user_message = user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} review successfully parsed")

            # Normalize agree field
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]
            log = {"parsed_output": result, "reasoning_content": reasoning_content}
            return log
        except json.JSONDecodeError:
            # Fallback parsing
            print(f"Doctor {self.agent_id} review is not valid JSON, using fallback parsing")
            lines = response_text.strip().split('\n')
            result = {}

            for line in lines:
                if "agree" in line.lower():
                    result["agree"] = "true" in line.lower() or "yes" in line.lower()
                elif "explanation" in line.lower():
                    result["explanation"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif "answer" in line.lower():
                    result["answer"] = line.split(":", 1)[1].strip() if ":" in line else line

            # Ensure required fields
            if "agree" not in result:
                result["agree"] = False
            if "explanation" not in result:
                result["explanation"] = "No explanation provided"

            return result


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-vl-max", config_path: str = "config.toml"):
        super().__init__(agent_id=agent_id, agent_type=AgentType.SYNTHESIZER, model_key=model_key, config_path=config_path)
        print(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    def synthesize_opinions(self,
                           question: str,
                           doctor_opinions: List[Dict[str, Any]],
                           doctor_specialties: List[MedicalSpecialty],
                           current_round: int,
                           options: Dict[str, str] | None = None) -> Dict[str, Any]:
        """
        Synthesize multiple doctors' opinions. Returns a full log.
        """
        print(f"Meta agent synthesizing opinions with model: {self.model_key}")

        system_message = {
            "role": "system",
            "content": """You are a medical consensus coordinator facilitating a multidisciplinary team consultation.
Synthesize the opinions of multiple specialist doctors into a coherent analysis and a suggested consensus conclusion.
Consider each doctor's expertise and perspective, and weigh their opinions accordingly.
Your output should be in JSON format, including 'explanation' (synthesized reasoning) and 'answer' (suggested consensus conclusion) fields."""
        }
        
        if options:
            system_message["content"] += "\nFor multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) that best represents the consensus view."

        formatted_opinions = []
        for i, (opinion, specialty) in enumerate(zip(doctor_opinions, doctor_specialties)):
            formatted_opinion = f"Doctor {i+1} ({specialty.value}):\n"
            formatted_opinion += f"Explanation: {opinion.get('explanation', '')}\n"
            formatted_opinion += f"Answer: {opinion.get('answer', '')}\n"
            formatted_opinions.append(formatted_opinion)
        opinions_text = "\n".join(formatted_opinions)

        # Prepare user message with all opinions
        user_content = []
        user_content.append({
            "type": "text",
            "text": f"Round {current_round} Doctors' Opinions:\n{opinions_text}\n\n"
        })
        user_message = {
            "role": "user",
            "content": user_content
        }

        response_text, reasoning_content, system_msg, user_msg = self.call_llm(system_message = system_message, user_message = user_message)
        user_msg["content"] = [item for item in user_msg["content"] if item.get("type") != "image_url"]

        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Meta agent synthesis successfully parsed")
        except json.JSONDecodeError:
            print("Meta agent synthesis is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
        
        return {"parsed_output": result, "reasoning_content": reasoning_content, "llm_input": {"system_message": system_msg, "user_message": user_msg}}


class DecisionMakingAgent(BaseAgent):
    """Decision making agent that gives final answers based on synthesized opinions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-max-latest", config_path: str = "config.toml"):
        super().__init__(agent_id=agent_id, agent_type=AgentType.DECISION_MAKER, model_key=model_key, config_path=config_path)
        print(f"Initializing decision making agent, ID: {agent_id}, Model: {model_key}")

    def make_decision(self,
                      question: str,
                      synthesis: Dict[str, Any],
                      doctor_reviews: List[Dict[str, Any]],
                      doctor_specialties: List[MedicalSpecialty],
                      options: Dict[str, str] | None = None,
                      image_path: str | None = None) -> Dict[str, Any]:
        """
        Make a final decision based on synthesized report and doctor reviews. Returns a full log.
        """
        print(f"Decision making agent generating final answer")
        
        system_message = {
            "role": "system",
            "content": """You are a senior medical attending physician responsible for making the final, authoritative decision based on a multidisciplinary team's consultation.
Your task is to review the synthesized report and all specialists' final reviews to determine the single best answer.
Your output must be in JSON format, including 'explanation' (your final reasoning for the decision) and 'answer' (the definitive final conclusion) fields."""
        }
        
        if options:
            system_message["content"] += "\nFor multiple choice questions, ensure your 'answer' field contains ONLY the single option letter (A, B, C, etc.) that represents your final decision."

        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })

        options_text = ""
        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
        question_with_options = f"{question}{options_text}"

        synthesis_text = f"Synthesized Report from Coordinator:\nExplanation: {synthesis.get('explanation', '')}\nSuggested Answer: {synthesis.get('answer', '')}"
        
        formatted_reviews = []
        for i, (review, specialty) in enumerate(zip(doctor_reviews, doctor_specialties)):
            review_text = f"Doctor {i+1} ({specialty.value}):\n"
            review_text += f"Agree with synthesis: {'Yes' if review.get('agree', False) else 'No'}\n"
            review_text += f"Final Viewpoint: {review.get('current_viewpoint', '')}\n"
            review_text += f"Reason: {review.get('reason', '')}\n"
            formatted_reviews.append(review_text)
        reviews_text = "\n".join(formatted_reviews)
        
        text_content = {
            "type": "text",
            "text": f"""Question: {question_with_options}
{synthesis_text}
Specialists' Final Reviews on the Synthesis:
{reviews_text}
Your reasoning in the 'explanation' field must demonstrate a final, decisive synthesis.
Based on ALL available information presented above, provide your final decision. Your response must be in JSON format, including 'explanation' and 'answer' fields."""
        }
        user_content.append(text_content)

        user_message = {"role": "user", "content": user_content}
        response_text, reasoning_content, system_msg, user_msg = self.call_llm(system_message = system_message, user_message = user_message)
        user_msg["content"] = [item for item in user_msg["content"] if item.get("type") != "image_url"]

        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Decision making agent response successfully parsed")
        except json.JSONDecodeError:
            print("Decision making agent response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            
        return {"parsed_output": result, "reasoning_content": reasoning_content, "llm_input": {"system_message": system_msg, "user_message": user_msg}}

class MDTConsultation:
    """Multi-disciplinary team consultation coordinator."""

    def __init__(self,
                max_rounds: int = 1, # MedAgent原框架为单轮，此处设为1以保持一致，可调
                model_key: str = "qwen-max-latest",
                meta_model_key: str = "qwen-max-latest",
                decision_model_key: str = "qwen-max-latest",
                auditor_model_key: str = "gemini-2.5-pro",
                config_path: str = "config.toml"):
        self.max_rounds = max_rounds
        self.model_key = model_key
        self.config_path = config_path
        # Initialize agents
        self.expert_gatherer = ExpertGathererAgent(agent_id="expert_gatherer", model_key=model_key, config_path=config_path)
        self.meta_agent = MetaAgent(agent_id="meta", model_key=meta_model_key, config_path=config_path)
        self.decision_agent = DecisionMakingAgent(agent_id="decision", model_key=decision_model_key, config_path=config_path)
        self.auditor_agent = AuditorAgent(agent_id="auditor", model_key=auditor_model_key, config_path=config_path, agent_type = AgentType.AUDITOR)

        self.doctor_agents: List[DoctorAgent] = []
        self.doctor_specialties: List[MedicalSpecialty] = []
        print(f"Initialized MDT consultation, max_rounds={max_rounds}, doctor_model={model_key}")

    def _initialize_doctor_agents(self, specialties: List[MedicalSpecialty]):
        self.doctor_agents = []
        for idx, specialty in enumerate(specialties, 1):
            agent_id = f"doctor_{idx}"
            self.doctor_agents.append(DoctorAgent(agent_id=agent_id, specialty=specialty, model_key=self.model_key, config_path=self.config_path))
        self.doctor_specialties = specialties

    def run_consultation(self,
                            qid: str,
                            question: str,
                            options: Dict[str, str] | None = None,
                            image_path: str | None = None,
                            task:str = "open_coding") -> Dict[str, Any]:
            start_time = time.time()
            print(f"Starting MDT consultation for case {qid}")
            print(f"Question: {question}")
            if options: print(f"Options: {options}")

            # Step 1: Gather relevant domain experts
            specialties, gatherer_log = self.expert_gatherer.gather_question_domain_experts(question)
            print(f"Gathered specialties: {[s.value for s in specialties]}")
            self._initialize_doctor_agents(specialties)
            if task =="audit":
                # audit 2.1.1 role assignment
                audit_results_of_role_assignment = self.auditor_agent.audit_role_assignment(question=question, image_path=image_path, specialties=specialties)
                audit = {"rounds": []}
            current_round = 0
            case_history = {
                "rounds" : [],
            }
            consensus_reached = False
            final_decision_log = None # 初始化 final_decision_log

            while current_round < self.max_rounds and not consensus_reached:
                current_round += 1
                if task == "audit":
                    audit_round_data = {
                        "round": current_round,
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
                    audit_round_data["2_1_1_role_assignment"].append({
                        "specialties": [s.value for s in specialties],
                        "step": "role_assignment",
                        "audit_result": audit_results_of_role_assignment
                    })

                round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": [], "decision": None}
                case_history["rounds"].append(round_data)

                print(f"Starting round {current_round}")
                # Step 2: Each doctor analyzes the case
                doctor_opinion_parsed_outputs = []
                for i, doctor in enumerate(self.doctor_agents):
                    print(f"Doctor {i+1} ({doctor.specialty.value}) analyzing case")
                    opinion_log = doctor.analyze_case(question, options, image_path)
                    parsed_output = opinion_log["parsed_output"]
                    doctor_opinion_parsed_outputs.append(parsed_output)
                    print(f"Doctor {i+1} opinion: {parsed_output.get('answer', '')}")
                    if task == "audit":
                        # audit 1.1.1 facutal hallucination
                        audit_results_of_factual_hallucination = self.auditor_agent.audit_factual_hallucination(question = question, image_path=image_path, agent_id=doctor.agent_id, specialty=doctor.specialty, answer=parsed_output.get('answer', ''), explanation=parsed_output.get('explanation', ''))

                        # audit 1.2.1 neglect or misinterpretation of modality information
                        audit_results_of_neglect_or_misinterpretation_of_modality_info = self.auditor_agent.audit_neglect_or_misinterpretation_of_modality_info(question = question, image_path=image_path, agent_id=doctor.agent_id, specialty=doctor.specialty, answer=parsed_output.get('answer', ''), explanation=parsed_output.get('explanation', ''))

                        # audit 2.1.2 Failure to Activate Specialist Knowledge During Role Execution during Collaborative discussion
                        audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question, image_path=image_path, agent_id=doctor.agent_id, specialty=doctor.specialty, answer=parsed_output.get('answer', ''), explanation=parsed_output.get('explanation', ''))
                        
                        if current_round > 1:
                            # audit 2.2.1 Repetition of Initial Views during Collaborative discussion
                            audit_results_of_repetition_of_initial_views = self.auditor_agent.audit_repetition_of_initial_views(question = question, image_path=image_path, current_agent_id=doctor.agent_id, current_answer = parsed_output.get('answer', ''), current_explanation=parsed_output.get('explanation', ''), case_history=case_history) 
                            # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
                            audit_results_of_unresolved_conflicts_during_Collaboration = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question = question, current_agent_id=doctor.agent_id, current_answer = parsed_output.get('answer', ''), current_explanation=parsed_output.get('explanation', ''), case_history=case_history)
                            audit_round_data["2_2_1_repetition_of_initial_views"].append({
                                "agent_id": doctor.agent_id,
                                "specialty": doctor.specialty.value,
                                "step": "analysis",
                                "audit_result": audit_results_of_repetition_of_initial_views
                            })
                            audit_round_data["2_2_2_unresolved_conflicts"].append({ 
                                "agent_id": doctor.agent_id,
                                "specialty": doctor.specialty.value,
                                "step": "analysis",
                                "audit_result": audit_results_of_unresolved_conflicts_during_Collaboration
                            })

                        audit_round_data["1_1_1_factual_hallucination"].append({
                            "agent_id": doctor.agent_id,
                            "specialty": doctor.specialty.value,
                            "step": "analysis",
                            "audit_result": audit_results_of_factual_hallucination
                        })
                        
                        audit_round_data["1_2_1_neglect_or_misinterpretation_of_modality_info"].append({
                            "agent_id": doctor.agent_id,
                            "specialty": doctor.specialty.value,
                            "step": "analysis",
                            "audit_result": audit_results_of_neglect_or_misinterpretation_of_modality_info
                        })

                        audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                            "agent_id": doctor.agent_id,
                            "specialty": doctor.specialty.value,
                            "step": "analysis",
                            "audit_result": audit_results_of_domain_specific_knowledge_activation
                        })

                    case_history["rounds"][-1]["opinions"].append({
                        "agent_id": doctor.agent_id,
                        "specialty": doctor.specialty.value,
                        "log": opinion_log 
                    }) # after audit 2.2.1, we log the current opinion in case repetition 

                    print(f"Doctor {i+1} opinion: {opinion_log['parsed_output'].get('answer', '')}")


                # Step 3: Meta agent synthesizes opinions
                print("Meta agent synthesizing opinions")
                synthesis_log = self.meta_agent.synthesize_opinions(question=question, doctor_opinions=doctor_opinion_parsed_outputs, doctor_specialties=self.doctor_specialties, options=options, current_round=current_round)
                synthesis_parsed_output = synthesis_log["parsed_output"]
                synthesis_explanation = synthesis_parsed_output.get("explanation", "")
                synthesis_answer = synthesis_parsed_output.get("answer", "")
                if task == "audit":
                    # audtit 3.1.1 : Suppression of Correct Minority Views by Incorrect Consensus for synthesizer
                    audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_synthesizer = self.auditor_agent.audit_suppression_by_majority(
                        question = question, options = options, image_path = image_path, current_agent_id = self.meta_agent.agent_id, answer = synthesis_answer, explanation = synthesis_explanation, case_history = case_history
                    ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis

                    # audit 3.1.2 : Reasoning Distorted by Authority Bias for synthesizer
                    audit_results_of_authority_bias_for_synthesizer = self.auditor_agent.audit_authority_bias(
                        question = question, options = options, image_path = image_path, current_agent_id = self.meta_agent.agent_id, answer = synthesis_answer, explanation = synthesis_explanation, case_history = case_history
                    ) # here the discussion_context must include the role of domain agent and their answer and explanation before this synthesis

                    # audit 3.1.3: Neglect of Contradictions in Reasoning Process for synthesizer
                    audit_results_of_neglect_of_contradictions_in_reasoning_process_for_synthesizer = self.auditor_agent.audit_contradictions_during_decision(
                        question = question, current_agent_id = self.meta_agent.agent_id, explanation = synthesis_explanation, case_history = case_history, options = options
                    ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis
                    if current_round > 1:
                        # audit 3.2.1: Self-Contradiction in Viewpoints Across Rounds for synthesizer
                        audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_synthesizer = self.auditor_agent.audit_contradictions_across_rounds(
                            question = question, answer = synthesis_answer, explanation = synthesis_explanation, case_history = case_history, current_agent_id = self.meta_agent.agent_id, options = options
                        ) # here the meta_agent.memory includes all the previous syntheses and decisions
                        audit_round_data["3_2_1_self_contradiction_when_decision"].append({
                            "agent_id": self.meta_agent.agent_id,
                            "step": "synthesis",
                            "audit_result": audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_synthesizer
                        })
                    audit_round_data["3_1_1_suppression_of_minority_views"].append({
                        "agent_id": self.meta_agent.agent_id,
                        "step": "synthesis",
                        "audit_result": audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_synthesizer
                    })
                    audit_round_data["3_1_2_authority_bias"].append({
                        "agent_id": self.meta_agent.agent_id,
                        "step": "synthesis",
                        "audit_result": audit_results_of_authority_bias_for_synthesizer
                    })
                    audit_round_data["3_1_3_neglect_of_contradictions"].append({
                        "agent_id": self.meta_agent.agent_id,
                        "step": "synthesis",
                        "audit_result": audit_results_of_neglect_of_contradictions_in_reasoning_process_for_synthesizer
                    })

                case_history["rounds"][-1]["synthesis"] = synthesis_log # after synthesizer then log, in case repetition
                # Step 4: Doctors review synthesis
                doctor_review_parsed_outputs = []
                all_agree = True
                for i, doctor in enumerate(self.doctor_agents):
                    print(f"Doctor {i+1} ({doctor.specialty.value}) reviewing synthesis")
                    review_log = doctor.review_synthesis(question, synthesis_parsed_output, options, image_path)
                    review_parsed_output = review_log["parsed_output"]
                    review_reason = review_parsed_output.get("reason", "")
                    review_outcome = review_parsed_output.get("answer", "")
                    doctor_review_parsed_outputs.append(review_parsed_output)
                    agrees = review_parsed_output.get('agree', False)
                    all_agree = all_agree and agrees
                    print(f"Doctor {i+1} agrees: {'Yes' if agrees else 'No'}")
                    if task == "audit":
                        # audit 2.1.2 Failure to Activate Specialist Knowledge During Role Execution during Collaborative discussion
                        audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question, image_path=image_path, agent_id=doctor.agent_id, specialty=doctor.specialty, answer=review_outcome, explanation=review_reason)

                        # audit 2.2.1 Repetition of Initial Views during Collaborative discussion
                        audit_results_repetition_of_initial_views = self.auditor_agent.audit_repetition_of_initial_views(question=question, image_path = image_path, current_agent_id=doctor.agent_id, current_answer = review_outcome, current_explanation=review_reason, case_history=case_history)
                        
                        # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
                        audit_results_of_unresolved_conflicts_during_review = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question=question, current_agent_id=doctor.agent_id, current_answer = review_outcome, current_explanation=review_reason, case_history=case_history)

                        audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                            "agent_id": doctor.agent_id,
                            "specialty": doctor.specialty.value,
                            "step": "review",
                            "audit_result": audit_results_of_domain_specific_knowledge_activation
                        })
                        audit_round_data["2_2_1_repetition_of_initial_views"].append({
                            "agent_id": doctor.agent_id,
                            "specialty": doctor.specialty.value,
                            "step": "review",
                            "audit_result": audit_results_repetition_of_initial_views
                        })
                        audit_round_data["2_2_2_unresolved_conflicts"].append({ 
                            "agent_id": doctor.agent_id,
                            "specialty": doctor.specialty.value,
                            "step": "review",
                            "audit_result": audit_results_of_unresolved_conflicts_during_review
                        })

                    case_history["rounds"][-1]["reviews"].append({
                        "agent_id": doctor.agent_id,
                        "specialty": doctor.specialty.value,
                        "log": review_log 
                    })
                if task == "audit":
                    audit["rounds"].append(audit_round_data)

                consensus_reached = all_agree

                if consensus_reached or current_round == self.max_rounds:
                    print("Proceeding to final decision.")
                    
                    # Step 5: Decision making agent provides final answer
                    final_decision_log = self.decision_agent.make_decision(question, synthesis_parsed_output, doctor_review_parsed_outputs, self.doctor_specialties, options, image_path)
                    decision_explanation = final_decision_log.get("parsed_output", {}).get("explanation", "")
                    decision_answer = final_decision_log.get("parsed_output", {}).get("answer", "")
                    break 
                else:
                    print("No consensus reached, continuing to next round.")

            if final_decision_log:
                print(f"Final answer: {final_decision_log['parsed_output'].get('answer', 'N/A')}")
            else:
                print("Consultation ended without a final decision being generated.")
            if task == "audit":
                # audit 3.1.1: Suppression of Correct Minority Views by Incorrect Consensus during Decision-making for decision-maker
                audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker = self.auditor_agent.audit_suppression_by_majority(
                    question = question, options = options, image_path = image_path, current_agent_id = self.decision_agent.agent_id, answer = decision_answer, explanation = decision_explanation, case_history = case_history
                ) # here the discussion_context includes all the domain agents' answers and explanations before this decision

                # audit 3.1.2: Reasoning Distorted by Authority Bias for decision-maker
                audit_results_of_authority_bias_for_decision_maker = self.auditor_agent.audit_authority_bias(
                    question = question, options = options, image_path = image_path, current_agent_id = self.decision_agent.agent_id, explanation = decision_explanation, case_history = case_history, answer = decision_answer
                ) # here the discussion_context must include the role of domain agent and their answer and explanation before this decision

                # audit 3.1.3: Neglect of Contradictions in Reasoning Process for decision-maker
                audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker = self.auditor_agent.audit_contradictions_during_decision(
                    question = question, current_agent_id = self.decision_agent.agent_id, explanation = decision_explanation, case_history = case_history, options = options
                ) # here the discussion_context includes all the domain agents' answers and explanations before this decision
                if current_round > 1:
                    # audit 3.2.1: Self-Contradiction in Viewpoints Across Rounds for decision-maker
                    audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_decision_maker = self.auditor_agent.audit_contradictions_across_rounds(
                        question = question, options = options, answer = decision_answer, current_agent_id = self.decision_agent.agent_id, explanation = decision_explanation, case_history = case_history
                    ) # here the meta agent's memory includes all its previous decisions and syntheses!
                    audit_round_data["3_2_1_self_contradiction_when_decision"].append({
                        "agent_id": self.decision_agent.agent_id,
                        "step": "decision",
                        "audit_result": audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_decision_maker
                    })
                audit_round_data["3_1_1_suppression_of_minority_views"].append({
                    "agent_id": self.decision_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker
                })
                audit_round_data["3_1_2_authority_bias"].append({
                    "agent_id": self.decision_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_authority_bias_for_decision_maker
                })
                audit_round_data["3_1_3_neglect_of_contradictions"].append({
                    "agent_id": self.decision_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker
                })


            case_history["rounds"][-1]["decision"] = final_decision_log
            if task == "audit":
                case_history["audit"] = audit
            # Finalize history
            processing_time = time.time() - start_time
            
            case_history.update({
                "final_decision_log": final_decision_log,
                "consensus_reached": consensus_reached,
                "total_rounds": current_round,
                "processing_time": processing_time
            })

            return case_history

def process_input(item, model_key, meta_model_key, decision_model_key, auditor_model_key, config_path, task:str):
    """Process a single input data item."""
    mdt = MDTConsultation(
        max_rounds=3,
        model_key=model_key,
        meta_model_key=meta_model_key,
        decision_model_key=decision_model_key,
        auditor_model_key=auditor_model_key,
        config_path=config_path
    )
    return mdt.run_consultation(
        qid=item.get("qid"),
        question=item.get("question"),
        options=item.get("options"),
        image_path=item.get("image_path"),
        task = task
    )


def main():
    parser = argparse.ArgumentParser(description="Run MDT consultation on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name")
    parser.add_argument("--model", required=True, type=str, help="Model for doctor agents")
    parser.add_argument("--meta_model", required=True, type=str, help="Model for meta agent")
    parser.add_argument("--decision_model", required=True, type=str, help="Model for decision agent")
    parser.add_argument("--auditor_model", type=str, required=True, help="Model for auditor agent")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to process")
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
    main_llm = args.model
    
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

    for item in tqdm(data[:args.num_samples], desc=f"Running MDT on {args.dataset}"):
        qid = item["qid"]
        existing_qids = set()
        if output_file.exists():
            print(f"Output file {output_file} already exists. Appending new results.")
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
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
            full_case_history = process_input(
                item,
                model_key=args.model,
                meta_model_key=args.meta_model,
                decision_model_key=args.decision_model,
                auditor_model_key=args.auditor_model,
                config_path=args.config_path,
                task = task
            )

            final_decision_log = full_case_history.get("final_decision_log", {})
            final_decision_parsed = final_decision_log.get("parsed_output", {})
            predicted_answer = final_decision_parsed.get("answer", "Error: No answer found")
            print(f"Predicted answer for {qid}: {predicted_answer}")

            item_result = {
                "qid": qid,
                "timestamp": int(time.time()),
                "question": item["question"],
                "options": item.get("options"),
                "image_path": item.get("image_path"),
                "ground_truth": item.get("answer"),
                "predicted_answer": predicted_answer,
                "case_history": full_case_history,
            }
            save_jsonl(item_result, output_file)

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            error_log = {"qid": qid, "error": str(e)}
            save_jsonl(error_log, error_output_file)


if __name__ == "__main__":
    main()