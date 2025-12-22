# multi_agent_colacare_full_log.py
"""
medagentaudit/medqa/multi_agent_colacare.py
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
utils_root = current_file_path.parents[1] / "utils"
project_root = current_file_path.parents[2]
sys.path.extend([str(utils_root), str(project_root)])
from config import get_config
from dual_logger import DualLogger
from encode_image import encode_image
from json_utils import load_json, save_json, preprocess_response_string
from keu import KEU
from analysishelper import AnalysisHelperLLM
from auditor_agent import AuditorAgent
from BaseAgent import BaseAgent
class MedicalSpecialty(Enum):
    """Medical specialty enumeration."""
    INTERNAL_MEDICINE = "Internal Medicine"
    SURGERY = "Surgery"
    RADIOLOGY = "Radiology"


class AgentType(Enum):
    """Agent type enumeration."""
    DOCTOR = "Doctor"
    META = "Coordinator"
    AUDITOR = "Auditor"

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
        super().__init__(agent_id=agent_id, agent_type=AgentType.DOCTOR, config_path=config_path, model_key=model_key)
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

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} response successfully parsed")
            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1, # 每轮包含分析和复审，所以要这样操作
                "content": result
            })
            analysis_log = {
            "parsed_output": result,
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
            }
            return analysis_log # logging，返回了完整的系统和用户消息。
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
            return analysis_log # logging，返回了完整的系统和用户消息。


    # MODIFICATION START: Changed return type to a dictionary for comprehensive logging.
    def review_synthesis(self,
                         question: str,
                         synthesis: Dict[str, Any],
                         audit_trail: Dict[str, Any],
                         ccp_text: str = "",
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
                    f"Your output must be a JSON object, including:"
                    f"1. 'agree': boolean (true/false)."
                    f"2. 'current_viewpoint': Your current final answer after this review (e.g., 'A', 'B')."
                    f"3. 'viewpoint_changed': boolean, true if your 'current_viewpoint' is different from your initial analysis's answer."
                    f"4. 'justification_type': A string, must be one of ['evidence_based', 'consensus_based']. Choose 'evidence_based' if your decision is primarily driven by specific KEU facts. Choose 'consensus_based' if your decision is primarily to align with the synthesized opinion or majority view."
                    f"5. 'cited_references': A list of strings containing the KEU-IDs or Agent-IDs that influenced your decision."
                    f"6. 'reason': Your detailed textual explanation for your decision."
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

        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed so far:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"

        text_content = {
            "type": "text",
            "text": f"Original question: {question_with_options}\n\n"
                    f"{own_analysis_text}"
                    f"Synthesized Opinion for Review:\n{synthesis_text}\n\n"
                    f"Available Key Evidential Units (KEUs):\n{keu_list_text}\n\n"
                    f"Available Critical Consensus Points (CCPs):\n{ccp_text}\n\n" 
                    f"Pay attention to the potential conflicts (CCPs) listed above, as addressing them in your 'reason' field will strengthen your argument. "
                    f"Please provide your comprehensive review.\nYour 'reason' field MUST reference the KEU-IDs that support your decision. \n"
                    f"Your response MUST be a single JSON object, strictly adhering to the 6-field structure defined in your system instructions. "
                    f"Pay close attention to correctly populating 'viewpoints_changed', 'justification_type', and 'cited_references'."
        }
        user_content.append(text_content)

        user_message = {
            "role": "user",
            "content": user_content,
        }

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message) # logging,获取了完善的系统和用户消息

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
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
        }
        return review_log # logging,记录了完整的系统和用户消息


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions."""

    def __init__(self, agent_id: str, config_path: str, model_key: str = "qwen-max-latest"):
        """
        Initialize a meta agent.
        """
        super().__init__(agent_id=agent_id, agent_type=AgentType.META, config_path=config_path, model_key = model_key)
        print(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    # MODIFICATION START: Changed return type to a dictionary for comprehensive logging.
    def synthesize_opinions(self,
                            question: str,
                            doctor_opinions: List[Dict[str, Any]],
                            doctor_specialties: List[MedicalSpecialty],
                            audit_trail: Dict[str, Any],
                            ccp_text: str = "",
                            current_round: int = 1,
                            options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        # MODIFICATION END
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

        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed so far:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"

        user_content = []
        text_content = (
            f"Question: {question_with_options}\n\n"
            f"Round {current_round} Doctors' Opinions:\n{opinions_text}\n\n"
            f"Available KEUs (Key Evidential Units):\n{keu_list_text}\n"
            f"Available CCPs (Critical Consensus Points):\n{ccp_text}\n\n"
            f"Note that potential conflicts (CCPs) have been identified; a robust synthesis must acknowledge or resolve these points.\n\n"
            f"**CRITICAL INSTRUCTION:** Your task is to synthesize these diverse opinions into a single, coherent analysis. In your 'explanation', you **MUST selectively cite only the most important KEU-IDs** that support your synthesized view. **DO NOT simply list all available KEUs.** Your goal is to demonstrate a deep understanding by building a new, consolidated argument from the strongest evidence (e.g., 'Synthesizing the specialists' views, the consensus leans towards X, primarily supported by the crucial findings in KEU-2 and KEU-5...').\n\n"
            f"Provide your synthesis in JSON format, including 'explanation' (comprehensive reasoning) and 'answer' (clear conclusion) fields."
        )
        user_content.append({
            "type":"text",
            "text":text_content
        })

        user_message = {
            "role":"user",
            "content": user_content
        }
        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

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
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
        }
        return synthesis_log # logging，返回了完整的系统和用户消息。

    def make_final_decision(self,
                            question: str,
                            doctor_reviews: List[Dict[str, Any]],
                            doctor_specialties: List[MedicalSpecialty],
                            current_synthesis: Dict[str, Any],
                            current_round: int,
                            max_rounds: int,
                            audit_trail: Dict[str, Any],
                            image_path:Optional[str] = None,
                            options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    # MODIFICATION END
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

        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed so far:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"
            
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
            f"Doctor Reviews on this synthesis:\n{reviews_text}\n\n"
            f"{keu_list_text}\n\n"
            f"**CRITICAL INSTRUCTION:** Your reasoning in the 'explanation' field must demonstrate synthesis, not just summarization. To do this, you **MUST selectively cite only the most pivotal KEU-IDs** that form the core basis of your conclusion. **DO NOT list or repeat all available KEUs.** Your task is to build an argument using the strongest evidence. (e.g., 'Based on the critical findings in KEU-1 and KEU-3, I conclude...').\n\n"
            f"History of Previous Rounds:\n{previous_syntheses_text}\n\n"
            f"Based on ALL available information presented above, provide your {decision_type} decision. Your explanation should be grounded in the evidence and reasoning from the synthesis and reviews. "
            f"Your response must be in JSON format, including 'explanation' and 'answer' fields."
        )
        user_content.append({
            "type":"text",
            "text":text_content
        })
        user_message = {
            "role":"user",
            "content": user_content
        }

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

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
            "llm_input": {
                "system_message": system_msg,
                "user_message": user_msg
            }
        }
        return decision_log # logging，返回了完整的系统和用户消息。

class MDTConsultation:
    """Multi-disciplinary team consultation coordinator."""

    def __init__(self,
                 max_rounds: int = 3,
                 doctor_configs: List[Dict] = None,
                 meta_model_key: str = None,
                 auditor_model_key: str = None,
                 conflict_analysis_model_key: str = None,
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
        self.auditor_agent = AuditorAgent(agent_id="auditor", config_path = config_path, model_key= auditor_model_key)

        # 初始化 AnalysisHelperLLM，使其在整个咨询流程中可用
        self.analysis_llm = AnalysisHelperLLM(config_path=config_path, model_key=conflict_analysis_model_key) 

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
                         image_path: Optional[str] = None) -> Dict[str, Any]:
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
        current_round = 0
        final_decision_log = None
        consensus_reached = False
        decision_log = None # Initialize decision_log to handle cases with no rounds
        audit_trail = {
            "keus": {},  # Dict[str, KEU] # TODO
            "viewpoints": {doc.agent_id: [] for doc in self.doctor_agents}, # TODO
            "collaboration_audits": {}, # TODO
            "ccps": {} # TODO
        }
        all_unresolved_ccps = []
        ccp_counter = 0
        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            print(f"Starting round {current_round}")

            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": [], "decision": None} 

            # Step 1: Each doctor analyzes the case
            doctor_opinion_parsed_outputs = []

            if current_round == 1:
                keu_counter = 0 # TODO

            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) analyzing case")
                opinion_log = doctor.analyze_case(question, options, image_path)
                parsed_output = opinion_log["parsed_output"]
                explanation = parsed_output.get("explanation", "")
                answer = parsed_output.get("answer", "")
                # audit 2.1.1 role assignment
                audit_results_of_role_assignment = self.auditor_agent.audit_role_assignment(question=question, image_path=image_path, agent_id=doctor.agent_id, specialty=doctor.specialty, answer=answer, explanation=explanation)

                # audit 2.1.2 domain-specific knowledge activation
                audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question, image_path, doctor.agent_id, doctor.specialty, answer, explanation)

                if current_round > 1:
                    # audit 2.2.1 Repetition of Initial Views during Collaborative discussion
                    audit_results_of_role_assignment = self.auditor_agent.audit_repetition_of_initial_views(question = question, image_path=image_path, current_agent_id=doctor.agent_id, current_explanation=explanation, case_history=case_history) 
                    # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
                    audit_results_of_unresolved_conflicts_during_Collaboration = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question = question, current_agent_id=doctor.agent_id, current_explanation=explanation, case_history=case_history) 

                round_data["opinions"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "log": opinion_log 
                }) # after audit 2.2.1, we log the current opinion in case repetition 
                
                step_id = f"round_1_analysis_{doctor.agent_id}"
                audit_trail["collaboration_audits"][step_id] = {**contribution_audit, **risk_audit}
                doctor_opinion_parsed_outputs.append(parsed_output)


                print(f"Doctor {i+1} opinion: {opinion_log['parsed_output'].get('answer', '')}")

            # Step 2: Meta agent synthesizes opinions
            print("Meta agent synthesizing opinions")
            synthesis_log = self.meta_agent.synthesize_opinions(
                question, doctor_opinion_parsed_outputs, self.doctor_specialties,
                audit_trail=audit_trail, ccp_text=ccp_text_for_prompt, current_round=current_round, options=options
            )
            synthesis_parsed_output = synthesis_log["parsed_output"]
            synthesis_explanation = synthesis_parsed_output.get("explanation", "")
            synthesis_answer = synthesis_parsed_output.get("answer", "")
            print(f"Meta agent synthesis: {synthesis_parsed_output.get('answer', '')}")

            # audit 2.2.2 : Unresolved Conflicts during Collaborative discussion for synthesizer
            audit_results_of_unresolved_conflicts_during_collaboration_for_synthesizer = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(
                question=question, current_agent_id=self.meta_agent.agent_id, current_explanation=synthesis_explanation, case_history=case_history
            )

            # audtit 3.1.1 : Suppression of Correct Minority Views by Incorrect Consensus for synthesizer
            audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_synthesizer = self.auditor_agent.audit_suppression_by_majority(
                question = question, options = options, image_path = image_path, current_agent_id = self.meta_agent.agent_id, synthesis_answer = synthesis_answer, synthesis_explanation = synthesis_explanation, case_history = case_history
            ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis

            # audit 3.1.2 : Reasoning Distorted by Authority Bias for synthesizer
            audit_results_of_authority_bias_for_synthesizer = self.auditor_agent.audit_authority_bias(
                question = question, options = options, image_path = image_path, current_agent_id = self.meta_agent.agent_id, synthesis_answer = synthesis_answer, synthesis_explanation = synthesis_explanation, case_history = case_history
            ) # here the discussion_context must include the role of domain agent and their answer and explanation before this synthesis

            # audit 3.1.3: Neglect of Contradictions in Reasoning Process for synthesizer
            audit_results_of_neglect_of_contradictions_in_reasoning_process_for_synthesizer = self.auditor_agent.audit_contradictions_during_decision(
                question, options, synthesis_answer, synthesis_explanation, discussion_context
            ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis

            # audit 3.2.1: Self-Contradiction in Viewpoints Across Rounds for synthesizer
            audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_synthesizer = self.auditor_agent.audit_self_contradiction_across_rounds(
                question, synthesis_answer, synthesis_explanation, self.meta_agent.memory
            ) # here the meta_agent.memory includes all the previous syntheses and decisions

            round_data["synthesis"] = synthesis_log # after synthesizer then log, in case repetition


            # 机制三：审计元智能体风险规避层级
            synthesis_risk_audit = self.auditor_agent.audit_risk_and_quality(self.meta_agent.agent_id, synthesis_explanation, image_path)
            step_id = f"round_{current_round}_synthesis"
            audit_trail["collaboration_audits"][step_id] = synthesis_risk_audit


            # 机制一：记录keu在synthesis中的出现和引用情况。
            for keu_id, keu in audit_trail["keus"].items():
                if keu_id in synthesis_explanation or keu.content in synthesis_explanation:
                    keu.present_in_synthesis[current_round] = True
                    keu.cited_by.append({
                        "agent_id": self.meta_agent.agent_id,
                        "round": current_round,
                        "action": "synthesis"
                    })
                else:
                    keu.present_in_synthesis[current_round] = False
            # Step 3: Doctors review synthesis
            doctor_review_parsed_outputs = []
            # 机制2修改：初始化观点追踪器，确保每个医生的观点都有记录
            if "viewpoints" not in audit_trail:
                audit_trail["viewpoints"] = {doc.agent_id: [] for doc in self.doctor_agents}
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty.value}) reviewing synthesis")
                review_log = doctor.review_synthesis(question, synthesis_parsed_output, audit_trail=audit_trail, ccp_text=ccp_text_for_prompt, options=options, image_path=image_path)
                review_parsed_output = review_log["parsed_output"]
                review_reason = review_parsed_output.get("reason", "")
                review_outcome = "agrees" if review_parsed_output.get("agree", False) else "disagrees"
                cited_refs = review_parsed_output.get("cited_references", [])

                # audit 2.1.2 Failure to Activate Specialist Knowledge During Role Execution during Collaborative discussion
                contribution_audit = self.auditor_agent.audit_domain_specific_knowledge_activation(question, doctor.agent_id, doctor.specialty, review_reason)

                # audit 2.2.1 Repetition of Initial Views during Collaborative discussion 
                audit_results_repetition_of_initial_views = self.auditor_agent.audit_repetition_of_initial_views(question=question, image_path = image_path, current_agent_id=doctor.agent_id, current_explanation=review_reason, case_history=case_history)
                
                # audit 2.2.2 Unresolved Conflicts during Collaborative discussion
                audit_results_of_unresolved_conflicts_during_review = self.auditor_agent.audit_unresolved_conflicts_during_Collaboration(question=question, current_agent_id=doctor.agent_id, current_explanation=review_reason, case_history=case_history) 

                round_data["reviews"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "log": review_log 
                })

                step_id = f"round_{current_round}_review_{doctor.agent_id}"
                audit_trail["collaboration_audits"][step_id] = {**contribution_audit, **risk_audit}

                # 检查机制1在review过程中的引用和反驳情况
                for keu_id, keu in audit_trail["keus"].items():
                    # 检查是否被引用
                    if keu_id in cited_refs or keu_id in review_reason or keu.content in review_reason:
                        keu.cited_by.append({
                            "agent_id": doctor.agent_id,
                            "round": current_round,
                            "action": "review"
                        })
                    # 检查是否被反驳 (这里用一个简单的关键词启发式，精确判断在后分析阶段)
                    # 我们将主要的反驳判断逻辑放在 analyze_failures.py 中
                    if not review_parsed_output.get("agree", True):
                        # 如果医生不同意synthesis，且在他的理由中提到了某个KEU，我们初步标记为潜在反驳
                        if keu_id in review_reason or keu.content in review_reason:
                             keu.rebuttals.append({
                                 "agent_id": doctor.agent_id,
                                 "round": current_round,
                                 "reason": review_reason
                             })

                doctor_review_parsed_outputs.append(review_parsed_output)

                # 机制二：在review后记录观点变化
                review_viewpoint_entry = {
                    "step": f"round_{current_round}_review",
                    "viewpoint": review_parsed_output.get("current_viewpoint"),
                    "viewpoint_changed": review_parsed_output.get("viewpoint_changed", False),
                    "justification_type": review_parsed_output.get("justification_type", "unknown"),
                    "cited_references": review_parsed_output.get("cited_references", []),
                }
                audit_trail["viewpoints"][doctor.agent_id].append(review_viewpoint_entry)
                agrees = review_parsed_output.get('agree', False)
                all_agree = all_agree and agrees
                print(f"Doctor {i+1} agrees: {'Yes' if agrees else 'No'}")

            
            # 机制三：在元智能体决策前，审计论据综合质量得分
            overall_quality_audit = self.auditor_agent.audit_overall_quality_for_decision(
                question, doctor_review_parsed_outputs, self.doctor_specialties
            )
            audit_trail["collaboration_audits"][f"round_{current_round}_pre_decision_quality"] = overall_quality_audit


            # 机制四：每轮复审结束后，更新和发现冲突
            # 收集本轮的所有讨论文本
            round_discussion_text = round_data["synthesis"]["parsed_output"]["explanation"]
            for review in round_data["reviews"]:
                round_discussion_text += "\n" + review["log"]["parsed_output"].get("reason", "")
            
            # 判断遗留的冲突是否被解决
            resolved_this_round_log = []
            still_unresolved = []
            for ccp in all_unresolved_ccps:
                was_addressed, resolution_reasoning = self.analysis_llm.check_if_conflict_was_addressed(ccp, round_discussion_text)
                if was_addressed:
                    ccp['status'] = 'resolved'
                    ccp['round_resolved'] = current_round
                    ccp['resolution_reasoning'] = resolution_reasoning  # Store the reasoning
                    resolved_this_round_log.append(ccp)
                else:
                    still_unresolved.append(ccp)
            all_unresolved_ccps = still_unresolved
            print(f"Round {current_round}: {len(resolved_this_round_log)} CCP(s) were resolved.")
            # 基于本轮复审的文本，发现新冲突
            # 注意：这里的输入应该是 review 的文本，因为新的冲突往往产生于对 synthesis 的异议
            review_contributions = []
            for review_log in round_data["reviews"]:
                review_contributions.append({
                    'agent_id': review_log["doctor_id"],
                    'specialty': review_log["specialty"],
                    'text': review_log["log"]["parsed_output"].get("reason", "")
                })

            # 无条件调用冲突识别
            new_ccps = self.auditor_agent.identify_critical_conflicts(
                review_contributions,
                context_description="doctors' review reasons"
            )

            # 4. 将新发现的冲突加入追踪列表
            if current_round not in audit_trail["ccps"]:
                audit_trail["ccps"][current_round] = []
                
            for ccp in new_ccps:
                ccp['ccp_id'] = f"CCP-{ccp_counter}" # 分配唯一ID
                ccp['round_identified'] = current_round
                ccp['status'] = 'unresolved'
                ccp['round_resolved'] = None
                audit_trail["ccps"][current_round].append(ccp)
                ccp_counter += 1

            all_unresolved_ccps.extend(new_ccps)

            # Step 4: Meta agent makes decision based on reviews
            # MODIFICATION START: Capture the full log from the agent.
            decision_log = self.meta_agent.make_final_decision(
                question, doctor_review_parsed_outputs, self.doctor_specialties,
                synthesis_parsed_output, current_round, self.max_rounds, audit_trail, image_path = image_path, options=options
            )
            decision_explanation = decision_log['parsed_output'].get("explanation", "")
            decision_answer = decision_log['parsed_output'].get("answer", "")

            # audit 3.1.1: Suppression of Correct Minority Views by Incorrect Consensus during Decision-making for decision-maker
            audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker = self.auditor_agent.audit_suppression_by_majority(
                question = question, options = options, image_path = image_path, answer = decision_answer, explanation = decision_explanation, case_history = case_history
            ) # here the discussion_context includes all the domain agents' answers and explanations before this decision

            # audit 3.1.2: Reasoning Distorted by Authority Bias for decision-maker
            audit_results_of_authority_bias_for_decision_maker = self.auditor_agent.audit_authority_bias(
                question = question, options = options, image_path = image_path, answer = decision_answer, explanation = decision_explanation, case_history = case_history
            ) # here the discussion_context must include the role of domain agent and their answer and explanation before this decision

            # audit 3.1.3: Neglect of Contradictions in Reasoning Process for decision-maker
            audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker = self.auditor_agent.audit_contradictions_during_decision(
                question, options, decision_answer, decision_explanation, discussion_context
            ) # here the discussion_context includes all the domain agents' answers and explanations before this decision

            # audit 3.2.1: Self-Contradiction in Viewpoints Across Rounds for decision-maker
            audit_results_of_self_contradiction_in_viewpoints_across_rounds_for_decision_maker = self.auditor_agent.audit_contradictions_across_rounds(
                question, options, decision_answer, decision_explanation, self.meta_agent.memory
            ) # here the meta agent's memory includes all its previous decisions and syntheses!

            round_data["decision"] = decision_log # Store the decision log for this round

            decision_explanation = decision_log.get("parsed_output", {}).get("explanation", "")
            
            step_id = f"round_{current_round}_decision"
            audit_trail["collaboration_audits"][step_id] = {**decision_risk_audit, **decision_quality_audit}

            case_history["rounds"].append(round_data)

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

        # 机制1:记录keu在最终决策中的出现情况
        if final_decision_log:
            final_explanation = final_decision_log.get("parsed_output", {}).get("explanation", "")
            # --- 机制一升级：记录KEU在最终决策中的出现情况 ---
            for keu_id, keu in audit_trail["keus"].items():
                if keu_id in final_explanation or keu.content in final_explanation:
                    keu.present_in_final_decision = True
                    
        final_decision_parsed = final_decision_log['parsed_output'] if final_decision_log else {}
        print(f"Final decision: {final_decision_parsed.get('answer', 'N/A')}")

        processing_time = time.time() - start_time
        if "keus" in audit_trail and audit_trail["keus"]:
            serializable_keus = {keu_id: keu.to_dict() 
                                 for keu_id, keu in audit_trail["keus"].items()}
            audit_trail["keus"] = serializable_keus
        case_history["final_decision_log"] = final_decision_log
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        case_history["processing_time"] = processing_time
        case_history['audit_trail'] = audit_trail

        # MODIFICATION START: Add final state of all agent memories to the log.
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
        # MODIFICATION END

        return case_history


def parse_structured_output(response_text: str) -> Dict[str, str]:
    """
    Parse LLM response to extract structured output as a fallback.
    """
    try:
        parsed = json.loads(preprocess_response_string(response_text))
        return parsed
    except json.JSONDecodeError:
        lines = response_text.strip().split('\n')
        result = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace("'", "").replace('"', '')
                value = value.strip()
                result[key] = value

        if "explanation" not in result:
            result["explanation"] = "No structured explanation found in response"
        if "answer" not in result:
            result["answer"] = "No structured answer found in response"

        return result


def process_input(item, doctor_configs=None, config_path=None, meta_model_key="qwen-max-latest",auditor_model_key="gemini-2.5-pro",conflict_analysis_model_key="deepseek-reasoner"):
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
        conflict_analysis_model_key = conflict_analysis_model_key,
        config_path=config_path
    )

    result_history = mdt.run_consultation(
        qid=qid,
        question=question,
        options=options,
        image_path=image_path,
    )
    return result_history


def main():
    parser = argparse.ArgumentParser(description="Run MDT consultation on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name,like PathVQA,VQA-RAD")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], default="mc", help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--doctor_models", nargs='+', required=True, help="for qa, use deepseek-reasoner,for vqa,use qwen3-vl")
    parser.add_argument("--meta_model", type=str, required=True, help="gpt-5.1/gemini-2.5-flash")
    parser.add_argument("--auditor_model", type=str, required=True, help="gemini-3-pro-preview") # auditor model is the conflict model
    parser.add_argument("--config_path", type=str, required=True,help="Path to the config.toml file,default = utils/config.toml")
    parser.add_argument("--num_samples", type=int, required=True,help="Number of samples to process from the dataset")
    parser.add_argument("--test_mode", type=bool, help="If set, log will be saved to a test-specific directory.")
    args = parser.parse_args()

    test_mode = args.test_mode
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if test_mode:
        print("!!! TEST MODE ENABLED: Logs will be saved to test-specific directories !!!")
        terminal_log_dir = project_root / "logs" / "observation" / "test" / "terminal_log" / "ColaCare" / args.dataset
    else:
        terminal_log_dir = project_root / "logs" / "observation" / "terminal_log" / "ColaCare" / args.dataset
    terminal_log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_file = terminal_log_dir / f"{args.dataset}_{timestamp}_full_terminal.log"
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条

    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")
    qa_type = args.qa_type
    print(f"QA Format: {qa_type}")

    if test_mode:
        logs_dir = project_root / "logs" / "observation" / "test" / "ColaCare" / dataset_name
    else:
        logs_dir = project_root / "logs" / "observation" / "ColaCare" / dataset_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved to: {logs_dir}")

    data_path = f"./my_datasets/processed/medqa/{dataset_name}/medqa_{qa_type}_test.json"
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
        
        log_file_path = logs_dir / f"{qid}-result.json"

        if log_file_path.exists():
            print(f"Skipping {qid} - already processed")
            continue

        try:
            full_case_history = process_input(
                item,
                doctor_configs=doctor_configs,
                config_path=args.config_path,
                meta_model_key=args.meta_model,
                auditor_model_key=args.auditor_model,
                conflict_analysis_model_key=args.auditor_model
            )

            # MODIFICATION START: The final decision is now nested inside the log.
            final_decision_log = full_case_history.get("final_decision_log", {})
            print("Final decision log:", final_decision_log)  # Debugging line
            final_decision_parsed = final_decision_log.get("parsed_output", {})
            print("Final decision parsed:", final_decision_parsed)  # Debugging line
            predicted_answer = final_decision_parsed.get("answer", "Error: No answer found")
            print(f"Predicted answer for {qid}: {predicted_answer}")
            # MODIFICATION END

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

            save_json(item_result, log_file_path)

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            # Optionally, save an error log
            error_log = {
                "qid": qid,
                "error": str(e)
            }
            save_json(error_log, logs_dir / f"{qid}-error.json")


if __name__ == "__main__":
    main()