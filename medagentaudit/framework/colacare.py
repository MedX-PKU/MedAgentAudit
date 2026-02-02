"""
medagentaudit/medqa/multi_agent_colacare_audit.py
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
from json_utils import load_json, save_jsonl, preprocess_response_string
from auditor_agent import AuditorAgent
from base_agent import BaseAgent
from agent_type import AgentType
from medical_specialty import MedicalSpecialty
from parse_structured_output import parse_structured_output
class DoctorAgent(BaseAgent):
    """Doctor agent with a medical specialty."""

    def __init__(self,
                 agent_id: str,
                 specialty: MedicalSpecialty,
                 config_path: str,
                 model_key: str = "qwen-vl-max"):
        """
        Initialize a doctor agent.
        """
        super().__init__(agent_id=agent_id, agent_type=AgentType.DOMAIN, config_path=config_path, model_key=model_key)
        self.specialty = specialty
        print(f"Initializing {specialty.value} doctor agent, ID: {agent_id}, Model: {model_key}")

    def analyze_case(self,
                     question: str,
                     image_path: str | None,
                     options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze a medical case.
        Returns:
            A dictionary containing full log of the analysis.
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) analyzing case with model: {self.model_key}")

        system_message = {
            "role": "system",
            "content": f"You are a doctor specializing in {self.specialty.value}. "
                       f"Analyze the medical case and provide your professional opinion on the question. "
        }

        if options:
            system_message["content"] += (
                f"For multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) "
                f"that best matches your conclusion. Be specific about which option you are selecting."
            )

        user_content = []

        if image_path:

            base64_image = encode_image(image_path)
            image_url_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            user_content.append(image_url_content)

        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        text_content = {
            "type": "text",
            "text": f"{question_with_options}\n\nProvide your analysis in JSON format, including 'explanation' (detailed reasoning) ,'answer' (clear conclusion) , and 'IU' (a list of information units)  fields."
        }
        text_content["text"] += (
            f"Each IU in the list should be a string representing a single piece of object you perceive from the input text or image(if we have the image)"
            f"(e.g., 'A 2cm nodule is visible in the upper left lung lobe.', 'The patient's white blood cell count is 15,000/µL.')"
        )
        user_content.append(text_content)

        user_message = {
            "role": "user",
            "content": user_content,
        }

        response_text, reasoning_content, system_msg, user_msg = self.call_llm(system_message = system_message, user_message = user_message, response_format={"type": "json_object"})

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


    def review_synthesis(self,
                         question: str,
                         synthesis: Dict[str, Any],
                         options: Optional[Dict[str, str]] = None,
                         image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Review the meta agent's synthesis.
        Returns:
            A dictionary containing full log of the review.
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) reviewing synthesis with model: {self.model_key}")

        current_round = len(self.memory) // 2 + 1
        own_analysis = None
        for mem in reversed(self.memory):
            if mem["type"] == "analysis":
                own_analysis = mem["content"]
                break

        system_message = {
            "role": "system",
            "content": f"You are a doctor specializing in {self.specialty.value}, participating in round {current_round} of a multidisciplinary team consultation. "
                      f"Review the synthesis of multiple doctors' opinions and determine if you agree with the conclusion. "
                      f"Consider your previous analysis and the MetaAgent's synthesized opinion to decide whether to agree or provide a different perspective. "
                      f"Your output should be in JSON format, including 'agree' (boolean or 'yes'/'no'), 'reason' (rationale for your decision), "
                      f"and 'answer' (your suggested answer if you disagree; if you agree, you can repeat the synthesized answer) fields."
        }

        user_content = []

        if image_path:
            base64_image = encode_image(image_path)
            image_url_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            user_content.append(image_url_content)

        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        own_analysis_text = ""
        if own_analysis:
            own_analysis_text = f"Your previous analysis:\nExplanation: {own_analysis.get('explanation', '')}\nAnswer: {own_analysis.get('answer', '')}\n\n"

        synthesis_text = f"Synthesized explanation: {synthesis.get('explanation', '')}\n"
        synthesis_text += f"Suggested answer: {synthesis.get('answer', '')}"


        user_content.append({
            "type": "text",
            "text": f"Original question: {question_with_options}\n\n"
                  f"{own_analysis_text}"
                  f"{synthesis_text}\n\n"
                  f"Do you agree with this synthesized result? Please provide your response in JSON format, including:\n"
                  f"1. 'agree': 'yes'/'no'\n"
                  f"2. 'reason': Your rationale for agreeing or disagreeing\n"
                  f"3. 'answer': Your supported answer (can be the synthesized answer if you agree, or your own suggested answer if you disagree)"
        })

        user_message = {
            "role": "user",
            "content": user_content,
        }

        response_text, reasoning_content, system_msg, user_msg = self.call_llm(system_message = system_message, user_message = user_message, response_format={"type": "json_object"}) 

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} review successfully parsed")
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]
        except json.JSONDecodeError:
            print(f"Doctor {self.agent_id} review is not valid JSON, using fallback parsing")
            lines = response_text.strip().split('\n')
            result = {}
            for line in lines:
                if "agree" in line.lower():
                    result["agree"] = "true" in line.lower() or "yes" in line.lower()
                elif "reason" in line.lower():
                    result["reason"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif "answer" in line.lower():
                    result["answer"] = line.split(":", 1)[1].strip() if ":" in line else line
            if "agree" not in result:
                result["agree"] = False
            if "reason" not in result:
                result["reason"] = "No reason provided"
            if "answer" not in result:
                if own_analysis and "answer" in own_analysis:
                    result["answer"] = own_analysis["answer"]
                else:
                    result["answer"] = synthesis.get("answer", "No answer provided")

        self.memory.append({
            "type": "review",
            "round": current_round,
            "content": result
        })

        review_log = {
            "parsed_output": result,
            "reasoning_content": reasoning_content,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
        }
        return review_log 


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions."""

    def __init__(self, agent_id: str, config_path: str, model_key: str = "qwen-max-latest"):
        """
        Initialize a meta agent.
        """
        super().__init__(agent_id=agent_id, agent_type=AgentType.META, config_path=config_path, model_key = model_key)
        print(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    def synthesize_opinions(self,
                            question: str,
                            doctor_opinions: List[Dict[str, Any]],
                            doctor_specialties: List[MedicalSpecialty],
                            current_round: int = 1,
                            options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Synthesize multiple doctors' opinions.
        Returns:
            A dictionary containing full log of the synthesis.
        """
        print(f"Meta agent synthesizing round {current_round} opinions with model: {self.model_key}")

        system_message = {
            "role": "system",
            "content": f"You are a medical consensus coordinator facilitating round {current_round} of a multidisciplinary team consultation. "
                       "Synthesize the opinions of multiple specialist doctors into a coherent analysis and conclusion. "
                       "Consider each doctor's expertise and perspective, and weigh their opinions accordingly. "
                       "Your output should be in JSON format, including 'explanation' (synthesized reasoning) and "
                       "'answer' (consensus conclusion) fields."
        }

        if options:
            system_message["content"] += (
                " For multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) "
                "that best represents the consensus view. Be specific about which option you are selecting."
            )

        formatted_opinions = []
        for i, (opinion, specialty) in enumerate(zip(doctor_opinions, doctor_specialties)):
            formatted_opinion = f"Doctor {i+1} ({specialty.value}):\n"
            formatted_opinion += f"Explanation: {opinion.get('explanation', '')}\n"
            formatted_opinion += f"Answer: {opinion.get('answer', '')}\n"
            formatted_opinions.append(formatted_opinion)
        opinions_text = "\n".join(formatted_opinions)

        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        user_content = []
        text_content = (
            f"Question: {question_with_options}\n\n"
            f"Round {current_round} Doctors' Opinions:\n{opinions_text}\n\n"
            f"Please synthesize these opinions into a consensus view. Provide your synthesis in JSON format, including "
            f"'explanation' (comprehensive reasoning) and 'answer' (clear conclusion) fields."
        )
        user_content.append({
            "type":"text",
            "text":text_content
        })

        user_message = {
            "role":"user",
            "content": user_content
        }
        response_text, reasoning_content, system_msg, user_msg = self.call_llm(system_message = system_message, user_message = user_message, response_format={"type": "json_object"})

        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Meta agent synthesis successfully parsed")
        except json.JSONDecodeError:
            print("Meta agent synthesis is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)

        self.memory.append({
            "type": "synthesis",
            "round": current_round,
            "content": result
        })

        synthesis_log = {
            "parsed_output": result,
            "reasoning_content": reasoning_content,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
        }
        return synthesis_log

    def make_final_decision(self,
                            question: str,
                            doctor_reviews: List[Dict[str, Any]],
                            doctor_specialties: List[MedicalSpecialty],
                            current_synthesis: Dict[str, Any],
                            current_round: int,
                            max_rounds: int,
                            image_path:Optional[str] = None,
                            options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a final decision based on doctor reviews.
        Returns:
            A dictionary containing full log of the decision.
        """
        print(f"Meta agent making round {current_round} decision with model: {self.model_key}")

        all_agree = all(review.get('agree', False) for review in doctor_reviews)
        reached_max_rounds = current_round >= max_rounds

        system_message = {
            "role": "system",
            "content": "You are a medical consensus coordinator making a final decision. "
        }
        if all_agree:
            system_message["content"] += "All doctors agree with your synthesis, generate a final report."
        elif reached_max_rounds:
            system_message["content"] += (
                f"Maximum number of discussion rounds ({max_rounds}) reached without full consensus. "
                f"Make a final decision using majority opinion approach."
            )
        else:
            system_message["content"] += (
                "Not all doctors agree with your synthesis, but a decision for the current round is needed."
            )

        system_message["content"] += (
            " Your output should be in JSON format, including 'explanation' (final reasoning) and "
            "'answer' (final conclusion) fields."
        )

        if options:
            system_message["content"] += (
                " For multiple choice questions, ensure your 'answer' field contains the option letter (A, B, C, etc.) "
                "that represents the final decision. Be specific about which option you are selecting."
            )

        formatted_reviews = []
        for i, (review, specialty) in enumerate(zip(doctor_reviews, doctor_specialties)):
            formatted_review = f"Doctor {i+1} ({specialty.value}):\n"
            formatted_review += f"Agree: {'Yes' if review.get('agree', False) else 'No'}\n"
            formatted_review += f"Reason: {review.get('reason', '')}\n"
            formatted_review += f"Answer: {review.get('current_viewpoint', '')}\n"
            formatted_reviews.append(formatted_review)

        reviews_text = "\n".join(formatted_reviews)

        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            question_with_options = f"{question}\n{options_text}"
        else:
            question_with_options = question

        current_synthesis_text = (
            f"Current synthesized explanation: {current_synthesis.get('explanation', '')}\n"
            f"Current suggested answer: {current_synthesis.get('answer', '')}"
        )

        decision_type = "final" if all_agree or reached_max_rounds else "current round"

        previous_syntheses = []
        for i, mem in enumerate(self.memory):
            if mem["type"] == "synthesis" and mem["round"] < current_round:
                prev = f"Round {mem['round']} synthesis:\n"
                prev += f"Explanation: {mem['content'].get('explanation', '')}\n"
                prev += f"Answer: {mem['content'].get('answer', '')}"
                previous_syntheses.append(prev)

        previous_syntheses_text = "\n\n".join(previous_syntheses) if previous_syntheses else "No previous syntheses available."

            
        user_content =[]
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type":"image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        text_content = (
            f"Question: {question_with_options}\n\n" 
            f"{current_synthesis_text}\n\n" 
            f"Doctor Reviews:\n{reviews_text}\n\n" 
            f"Previous Rounds:\n{previous_syntheses_text}\n\n"
            f"Please provide your {decision_type} decision, "
            f"in JSON format, including 'explanation' and 'answer' fields."
        )
        user_content.append({
            "type":"text",
            "text":text_content
        })
        user_message = {
            "role":"user",
            "content": user_content
        }

        response_text, reasoning_content, system_msg, user_msg = self.call_llm(system_message = system_message, user_message = user_message, response_format={"type": "json_object"})

        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Meta agent final decision successfully parsed")
        except json.JSONDecodeError:
            print("Meta agent final decision is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)

        self.memory.append({
            "type": "decision",
            "round": current_round,
            "final": all_agree or reached_max_rounds,
            "content": result
        })

        decision_log = {
            "parsed_output": result,
            "reasoning_content": reasoning_content,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
        }
        return decision_log 

class MDTConsultation:
    """Multi-disciplinary team consultation coordinator."""

    def __init__(self,
                 max_rounds: int = 3,
                 doctor_configs: List[Dict] = None,
                 meta_model_key: str = None,
                 auditor_model_key: str = None,
                 config_path: str = "config.toml"):
        """
        Initialize MDT consultation.
        """
        self.max_rounds = max_rounds
        self.doctor_configs = doctor_configs or [
            {"specialty": MedicalSpecialty.INTERNAL_MEDICINE, "model_key": "qwen-vl-max"},
            {"specialty": MedicalSpecialty.SURGERY, "model_key": "qwen-vl-max"},
            {"specialty": MedicalSpecialty.RADIOLOGY, "model_key": "qwen-vl-max"},
        ]
        self.meta_model_key = meta_model_key

        self.doctor_agents = []
        for idx, config in enumerate(self.doctor_configs, 1):
            agent_id = f"doctor_{idx}"
            specialty = config["specialty"]
            model_key = config.get("model_key", "qwen-vl-max")
            doctor_agent = DoctorAgent(agent_id=agent_id, specialty=specialty, config_path=config_path, model_key=model_key)
            self.doctor_agents.append(doctor_agent)

        self.meta_agent = MetaAgent(agent_id="meta",config_path=config_path, model_key=meta_model_key)
        self.auditor_agent = AuditorAgent(agent_id="auditor", config_path = config_path, model_key= auditor_model_key, agent_type = AgentType.AUDITOR)

        self.doctor_specialties = [doctor.specialty for doctor in self.doctor_agents]

        doctor_info_parts = []
        for config in self.doctor_configs:
            model_name = config.get('model_key', 'default')
            specialty_name = config['specialty'].value
            doctor_info_parts.append(f"{specialty_name} ({model_name})")
        doctor_info = ", ".join(doctor_info_parts)
        print(f"Initialized MDT consultation, max_rounds={max_rounds}, doctors: [{doctor_info}], meta_model={meta_model_key}")

    def run_consultation(self,
                         qid: str,
                         question: str,
                         options: Optional[Dict[str, str]] = None,
                         image_path: Optional[str] = None,
                         task: str = "audit") -> Dict[str, Any]:
        """
        Run the MDT consultation process.
        Returns:
            A dictionary containing the complete case history and final result.
        """
        start_time = time.time()
        print(f"Starting MDT consultation for case {qid}")
        print(f"Question: {question}")
        if options:
            print(f"Options: {options}")

        case_history = {"rounds": []}
        audit = {"rounds": []}
        current_round = 0
        final_decision_log = None
        consensus_reached = False
        decision_log = None # Initialize decision_log to handle cases with no rounds
        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            print(f"Starting round {current_round}")
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
            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": [], "decision": None}
            case_history["rounds"].append(round_data)
            # Step 1: Each doctor analyzes the case
            doctor_opinion_parsed_outputs = []

            specialties = []
            for doctor in self.doctor_agents:
                specialties.append(doctor.specialty.value)
            if task == "audit":
                # audit 2.1.1 role assignment
                audit_results_of_role_assignment = self.auditor_agent.audit_role_assignment(question=question, image_path=image_path, specialties=specialties)
                audit_round_data["2_1_1_role_assignment"].append({
                    "specialties": specialties,
                    "step": "role_assignment",
                    "audit_result": audit_results_of_role_assignment
                })
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) analyzing case")
                opinion_log = doctor.analyze_case(question = question, options = options, image_path=image_path)
                parsed_output = opinion_log["parsed_output"]
                reasoning_content = opinion_log["reasoning_content"]
                explanation = parsed_output.get("explanation", "")
                answer = parsed_output.get("answer", "")
                if task == "audit":
                    # audit 1.1.1 facutal hallucination
                    audit_results_of_factual_hallucination = self.auditor_agent.audit_factual_hallucination(question = question, image_path=image_path, agent_id=doctor.agent_id, specialty=doctor.specialty, answer=answer, explanation=explanation)

                    # audit 1.2.1 neglect or misinterpretation of modality information
                    audit_results_of_neglect_or_misinterpretation_of_modality_info = self.auditor_agent.audit_neglect_or_misinterpretation_of_modality_info(question = question, image_path=image_path, agent_id=doctor.agent_id, specialty=doctor.specialty, answer=answer, explanation=explanation)

                    # audit 2.1.2 domain-specific knowledge activation
                    audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question, image_path=image_path, agent_id=doctor.agent_id, specialty=doctor.specialty, answer=answer, explanation=explanation)

                    if current_round > 1:
                        # audit 2.2.1 Repetition of Initial Views during Collaborative discussion
                        audit_results_of_repetition_of_initial_views = self.auditor_agent.audit_repetition_of_initial_views(question = question, image_path=image_path, current_agent_id=doctor.agent_id, current_answer = answer, current_explanation=explanation, case_history=case_history) 
                        # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
                        audit_results_of_unresolved_conflicts_during_Collaboration = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question = question, current_agent_id=doctor.agent_id, current_answer = answer, current_explanation=explanation, case_history=case_history) 

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

                doctor_opinion_parsed_outputs.append(parsed_output)

                case_history["rounds"][-1]["opinions"].append({
                    "agent_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "log": opinion_log
                }) # after audit 2.2.1, we log the current opinion in case repetition 
                
                print(f"Doctor {i+1} opinion: {opinion_log['parsed_output'].get('answer', '')}")

            # Step 2: Meta agent synthesizes opinions
            print("Meta agent synthesizing opinions")
            synthesis_log = self.meta_agent.synthesize_opinions(
                question, doctor_opinion_parsed_outputs, self.doctor_specialties,
                current_round=current_round, options=options
            )
            synthesis_parsed_output = synthesis_log["parsed_output"]
            synthesis_explanation = synthesis_parsed_output.get("explanation", "")
            synthesis_answer = synthesis_parsed_output.get("answer", "")
            print(f"Meta agent synthesis: {synthesis_parsed_output.get('answer', '')}")
            if task == "audit":
                # audit 2.2.2 : Unresolved Conflicts during Collaborative discussion for synthesizer
                audit_results_of_unresolved_conflicts_during_collaboration_for_synthesizer = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(
                    question=question, current_agent_id=self.meta_agent.agent_id, current_explanation=synthesis_explanation, case_history=case_history, current_answer = synthesis_answer
                )

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
                        question = question, answer = synthesis_answer, explanation = synthesis_explanation, case_history = case_history, options = options, current_agent_id = self.meta_agent.agent_id
                    ) # here the meta_agent.memory includes all the previous syntheses and decisions
                    audit_round_data["3_2_1_self_contradiction_when_decision"].append({
                        "agent_id": self.meta_agent.agent_id,
                        "step": "synthesis",
                        "audit_result": audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_synthesizer
                    })
                audit_round_data["2_2_2_unresolved_conflicts"].append({ 
                    "agent_id": self.meta_agent.agent_id,
                    "step": "synthesis",
                    "audit_result": audit_results_of_unresolved_conflicts_during_collaboration_for_synthesizer
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

            # Step 3: Doctors review synthesis
            doctor_review_parsed_outputs = []
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) reviewing synthesis")
                review_log = doctor.review_synthesis(question, synthesis_parsed_output, options=options, image_path=image_path)
                review_parsed_output = review_log["parsed_output"]
                review_reason = review_parsed_output.get("reason", "")
                review_outcome = "agrees" if review_parsed_output.get("agree", False) else "disagrees"
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

                doctor_review_parsed_outputs.append(review_parsed_output)

                agrees = review_parsed_output.get('agree', False)
                all_agree = all_agree and agrees
                print(f"Doctor {i+1} agrees: {'Yes' if agrees else 'No'}")

            # Step 4: Meta agent makes decision based on reviews
            decision_log = self.meta_agent.make_final_decision(
                question, doctor_review_parsed_outputs, self.doctor_specialties,
                synthesis_parsed_output, current_round, self.max_rounds, image_path = image_path, options=options
            )
            decision_explanation = decision_log['parsed_output'].get("explanation", "")
            decision_answer = decision_log['parsed_output'].get("answer", "")
            if task == "audit":
                # audit 3.1.1: Suppression of Correct Minority Views by Incorrect Consensus during Decision-making for decision-maker
                audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker = self.auditor_agent.audit_suppression_by_majority(
                    question = question, options = options, image_path = image_path, current_agent_id = self.meta_agent.agent_id, answer = decision_answer, explanation = decision_explanation, case_history = case_history
                ) # here the discussion_context includes all the domain agents' answers and explanations before this decision

                # audit 3.1.2: Reasoning Distorted by Authority Bias for decision-maker
                audit_results_of_authority_bias_for_decision_maker = self.auditor_agent.audit_authority_bias(
                    question = question, options = options, image_path = image_path, current_agent_id = self.meta_agent.agent_id, explanation = decision_explanation, case_history = case_history, answer = decision_answer
                ) # here the discussion_context must include the role of domain agent and their answer and explanation before this decision

                # audit 3.1.3: Neglect of Contradictions in Reasoning Process for decision-maker
                audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker = self.auditor_agent.audit_contradictions_during_decision(
                    question = question, current_agent_id = self.meta_agent.agent_id, explanation = decision_explanation, case_history = case_history, options = options
                ) # here the discussion_context includes all the domain agents' answers and explanations before this decision
                if current_round > 1:
                    # audit 3.2.1: Self-Contradiction in Viewpoints Across Rounds for decision-maker
                    audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_decision_maker = self.auditor_agent.audit_contradictions_across_rounds(
                        question = question, options = options, answer = decision_answer, current_agent_id = self.meta_agent.agent_id, explanation = decision_explanation, case_history = case_history
                    ) # here the meta agent's memory includes all its previous decisions and syntheses!
                    audit_round_data["3_2_1_self_contradiction_when_decision"].append({
                        "agent_id": self.meta_agent.agent_id,
                        "step": "decision",
                        "audit_result": audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_decision_maker
                    })
                audit_round_data["3_1_1_suppression_of_minority_views"].append({
                    "agent_id": self.meta_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker
                })
                audit_round_data["3_1_2_authority_bias"].append({
                    "agent_id": self.meta_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_authority_bias_for_decision_maker
                })
                audit_round_data["3_1_3_neglect_of_contradictions"].append({
                    "agent_id": self.meta_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker
                })

                audit["rounds"].append(audit_round_data)
            case_history["rounds"][-1]["decision"] = decision_log

            if all_agree:
                consensus_reached = True
                final_decision_log = decision_log
                print("Consensus reached")
            else:
                print("No consensus reached, continuing to next round")
                if current_round == self.max_rounds:
                    final_decision_log = decision_log
        if not final_decision_log:
            final_decision_log = decision_log

        final_decision_parsed = final_decision_log['parsed_output'] if final_decision_log else {}
        print(f"Final decision: {final_decision_parsed.get('answer', 'N/A')}")

        processing_time = time.time() - start_time
        if task == "audit":
            case_history["audit"] = audit
        case_history["final_decision_log"] = final_decision_log
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        case_history["processing_time"] = processing_time

        agent_final_states = {
            "meta_agent": {
                "id": self.meta_agent.agent_id,
                "memory": self.meta_agent.memory
            },
            "doctor_agents": [
                {
                    "id": doc.agent_id,
                    "specialty": doc.specialty.value,
                    "memory": doc.memory
                } for doc in self.doctor_agents
            ]
        }
        case_history["agent_final_states"] = agent_final_states

        return case_history

def process_input(item, doctor_configs=None, config_path=None, meta_model_key="qwen-max-latest",auditor_model_key="gemini-2.5-pro",conflict_analysis_model_key="deepseek-reasoner", task: str ="audit"):
    """
    Process a single input data item.
    """
    qid = item.get("qid")
    question = item.get("question")
    options = item.get("options")
    image_path = item.get("image_path")

    mdt = MDTConsultation(
        max_rounds=3,
        doctor_configs=doctor_configs,
        meta_model_key=meta_model_key,
        auditor_model_key=auditor_model_key,
        config_path=config_path
    )

    result_history = mdt.run_consultation(
        qid=qid,
        question=question,
        options=options,
        image_path=image_path,
        task=task
    )
    return result_history


def main():
    parser = argparse.ArgumentParser(description="Run MDT consultation on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name,like PathVQA,VQA-RAD")
    parser.add_argument("--doctor_models", nargs='+', required=True, help="for qa, use deepseek-reasoner,for vqa,use qwen3-vl")
    parser.add_argument("--meta_model", type=str, required=True, help="same as doctor agent")
    parser.add_argument("--auditor_model", type=str, required=True, help="gemini-3-pro-preview") # auditor model is the conflict model
    parser.add_argument("--config_path", type=str, required=True,help="Path to the config.toml file,default = utils/config.toml")
    parser.add_argument("--num_samples", type=int, required=True,help="Number of samples to process from the dataset")
    parser.add_argument("--time_stamp", type=str, required=True, help="Timestamp for logging purposes")
    parser.add_argument("--task", type = str, required=True, help="audit or open_coding?")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")

    task = args.task
    print(f"Task: {task}")

    timestamp = args.time_stamp
    current_mas_name = current_file_name

    main_llm = args.meta_model

    terminal_log_file = project_root / "logs" / f"{task}_results" / timestamp/ f"{current_mas_name}_{dataset_name}_{main_llm}_terminal_log" / "terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条

    data_path = project_root / "datasets" / "processed" / dataset_name / f"{task}" / f"medqa_{dataset_name.lower()}_{task}.json"
    
    output_file = project_root / "logs" / f"{task}_results" / timestamp / f"{current_mas_name}_{dataset_name}_{main_llm}.jsonl"
    error_output_file = project_root / "logs" / f"{task}_results" / timestamp / f"{current_mas_name}_{dataset_name}_{main_llm}_errors.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_output_file.parent.mkdir(parents=True, exist_ok=True)
    
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

    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    doctor_specialties = [
        MedicalSpecialty.INTERNAL_MEDICINE,
        MedicalSpecialty.SURGERY,
        MedicalSpecialty.RADIOLOGY # single-instance object
    ]

    if len(args.doctor_models) > len(doctor_specialties):
        print(f"Warning: More doctor models ({len(args.doctor_models)}) provided than specialties ({len(doctor_specialties)}). Extra models will not be used.")

    doctor_configs = []
    num_doctors_to_configure = min(len(args.doctor_models), len(doctor_specialties))
    for i in range(num_doctors_to_configure):
        doctor_configs.append({
            "specialty": doctor_specialties[i],
            "model_key": args.doctor_models[i]
        })

    doctor_model_names = [config["model_key"] for config in doctor_configs]
    print(f"Configuring {len(doctor_configs)} doctors with models: {doctor_model_names}") 

    for item in tqdm(data[:args.num_samples], desc=f"Running MDT consultation on {dataset_name}"): 
        qid = item["qid"]

        if qid in existing_qids:
            print(f"Skipping {qid} - already processed")
            continue

        try:
            full_case_history = process_input(
                item,
                doctor_configs=doctor_configs,
                config_path=args.config_path,
                meta_model_key=args.meta_model,
                auditor_model_key=args.auditor_model,
                conflict_analysis_model_key=args.auditor_model,
                task = task
            )

            final_decision_log = full_case_history.get("final_decision_log", {})
            print("Final decision log:", final_decision_log)  # Debugging line
            final_decision_parsed = final_decision_log.get("parsed_output", {})
            print("Final decision parsed:", final_decision_parsed)  # Debugging line
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
                "case_history": full_case_history, # This now contains the full, detailed log
            }

            save_jsonl(item_result, str(output_file))
            existing_qids.add(qid)

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            # Optionally, save an error log
            error_log = {
                "qid": qid,
                "error": str(e)
            }
            save_jsonl(error_log, str(error_output_file))
if __name__ == "__main__":
    main()