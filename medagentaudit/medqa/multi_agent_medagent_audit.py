"""
medagentboard/medqa/multi_agent_medagent.py
这个框架整个流程就是：输入任务开始-> ExpertGathererAgent 识别招募专家 -> 多个 DoctorAgent 分析 -> MetaAgent 综合 -> 多个 DoctorAgent 评审 -> DecisionMakerAgent 最终决策

This version has been modified to include extensive logging for detailed analysis
of the multi-agent collaboration process, as requested for the WWW2026 research proposal.
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from medagentaudit.utils.encode_image import encode_image
from medagentaudit.utils.json_utils import load_json, save_json, preprocess_response_string
from medagentaudit.utils.keu import KEU
from medagentaudit.utils.analysishelper import AnalysisHelperLLM
from medagentaudit.utils.dual_logger import DualLogger
from medagentaudit.utils.config import get_config


class MedicalSpecialty(Enum):
    """Medical specialty enumeration."""
    PEDIATRICS = "Pediatrics"
    CARDIOLOGY = "Cardiology"
    PULMONOLOGY = "Pulmonology"
    NEONATOLOGY = "Neonatology"
    GENETICS = "Genetics"
    # 补充ColaCare中的专科，以增加ExpertGathererAgent的选择范围
    INTERNAL_MEDICINE = "Internal Medicine"
    SURGERY = "Surgery"
    RADIOLOGY = "Radiology"


class AgentType(Enum):
    """Agent type enumeration."""
    DOCTOR = "Doctor"
    META = "Coordinator"
    DECISION_MAKER = "Decision Maker"
    EXPERT_GATHERER = "Expert Gatherer"
    AUDITOR = "Auditor" # 新增


class BaseAgent:
    """Base class for all agents."""

    def __init__(self,
                 agent_id: str,
                 agent_type: AgentType,
                 config_path: str,
                 model_key: str = "qwen-vl-max"):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent
            model_key: LLM model to use
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.memory = [] # 为每个Agent实例添加memory用于追踪自身历史

        self.llm = get_config(config_path=config_path,active_llm=model_key).llm
        self.client = OpenAI(
            api_key=self.llm.api_key,
            base_url=self.llm.base_url,
            timeout= self.llm.timeout,
        )
        self.model_name = self.llm.model_name
        print(f"Initialized {self.agent_type.value} agent with ID: {self.agent_id}, Model: {self.model_name}")

    def call_llm(self,
                 system_message: Dict[str, str],
                 user_message: Dict[str, Any],
                 max_retries: int = 3) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message containing question and optional image
            max_retries: Maximum number of retry attempts

        Returns:
            A tuple containing:
            - LLM response text
            - The system message sent to the LLM
            - The user message sent to the LLM
        """
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM, system message: {system_message['content'][:50]}...",flush=True)
                if hasattr(self.llm, 'reasoning') and self.llm.reasoning: # for model like gpt-5.1
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[system_message, user_message],
                        response_format={"type": "json_object"},
                        extra_body={"enable_thinking": True},
                        reasoning_effort=self.llm.reasoning.effort,
                        stream=self.llm.stream,
                    )
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[system_message, user_message],
                        response_format={"type": "json_object"},
                        extra_body={"enable_thinking": True},
                        stream=self.llm.stream,
                    )
                if not self.llm.stream:
                    response = completion.choices[0].message.content
                else:
                    response_chunks = []
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            response_chunks.append(chunk.choices[0].delta.content)

                    response = "".join(response_chunks)
                if not response.strip():
                    raise ValueError("Empty response from LLM")
                print(f"Agent {self.agent_id} received response: {response[:50]}...")
                return response, system_message, user_message
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise RuntimeError(f"CRITICAL: Agent {self.agent_id} failed after {max_retries} attempts. Reason: {str(e)}")
                time.sleep(1)


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
        
        # 扩展专科列表以供LLM选择
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

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)
        log = {
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
        
        # [TODO 已解决] system_message要强制输出KEU列表
        system_message = {
            "role": "system",
            "content": f"""You are a doctor specializing in {self.specialty.value}. 
Analyze the medical case and provide your professional opinion on the question. 
Your output must be a JSON object with three fields: 
1. 'explanation': Your detailed reasoning.
2. 'answer': Your final conclusion. For multiple-choice questions, this MUST be the single option letter (e.g., 'A', 'B').
3. 'keus': A list of key evidential units. Each KEU in the list must be a string representing a single, verifiable piece of evidence from the case (e.g., 'A 2cm nodule is visible in the upper left lung lobe.', 'The patient's white blood cell count is 15,000/µL.')."""
        }

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
        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} response successfully parsed")
        except json.JSONDecodeError:
            print(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            if "keus" not in result: # 确保fallback时也有keus字段
                result["keus"] = []
        
        self.memory.append({"type": "analysis", "content": result})
        return {"parsed_output": result, "llm_input": {"system_message": system_msg, "user_message": user_msg}}

    def review_synthesis(self,
                         question: str,
                         synthesis: Dict[str, Any],
                         audit_trail: Dict[str, Any],
                         ccp_text: str = "",
                         options: Optional[Dict[str, str]] = None,
                         image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Review the meta agent's synthesis. Returns a full log dictionary.
        """
        print(f"Doctor {self.agent_id} ({self.specialty.value}) reviewing synthesis with model: {self.model_key}")
        
        own_analysis = self.memory[0]['content'] if self.memory else {}

        # [TODO 已解决] system_message格式要变化一下以方便观察机制二
        system_message = {
            "role": "system",
            "content": f"""You are a doctor specializing in {self.specialty.value}, participating in a multidisciplinary team consultation. 
Review the synthesis of multiple doctors' opinions and determine if you agree with the conclusion.
Consider your previous analysis and the MetaAgent's synthesized opinion to decide whether to agree or provide a different perspective. 
Your output must be a JSON object, including:
1. 'agree': boolean (true/false).
2. 'current_viewpoint': Your current final answer after this review (e.g., 'A', 'B').
3. 'viewpoint_changed': boolean, true if your 'current_viewpoint' is different from your initial analysis's answer.
4. 'justification_type': A string, must be one of ['evidence_based', 'consensus_based']. Choose 'evidence_based' if your decision is primarily driven by specific KEU facts. Choose 'consensus_based' if your decision is primarily to align with the synthesized opinion or majority view.
5. 'cited_references': A list of strings containing the KEU-IDs or Agent-IDs that influenced your decision.
6. 'reason': Your detailed textual explanation for your decision."""
        }

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

        own_analysis_text = f"Your previous analysis:\nExplanation: {own_analysis.get('explanation', '')}\nAnswer: {own_analysis.get('answer', '')}\n\n"
        synthesis_text = f"Synthesized explanation: {synthesis.get('explanation', '')}\nSuggested answer: {synthesis.get('answer', '')}"
        
        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed so far:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"

        # [TODO 已解决] user_message要把keu_list,ccp_list放进去
        text_content = {
            "type": "text",
            "text": f"""Original question: {question_with_options}

{own_analysis_text}
Synthesized Opinion for Review:
{synthesis_text}

Available Key Evidential Units (KEUs):
{keu_list_text}

Available Critical Consensus Points (CCPs):
{ccp_text}

Pay attention to the potential conflicts (CCPs) listed above, as addressing them in your 'reason' field will strengthen your argument.
Please provide your comprehensive review. Your 'reason' field MUST reference the KEU-IDs that support your decision.
Your response MUST be a single JSON object, strictly adhering to the 6-field structure defined in your system instructions.
Pay close attention to correctly populating 'viewpoint_changed', 'justification_type', and 'cited_references'."""
        }
        user_content.append(text_content)

        user_message = {"role": "user", "content": user_content}
        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} review successfully parsed")
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]
        except json.JSONDecodeError:
            print(f"Doctor {self.agent_id} review is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            # 确保关键字段存在
            if 'agree' not in result: result['agree'] = False
            if 'current_viewpoint' not in result: result['current_viewpoint'] = own_analysis.get('answer')
            if 'viewpoint_changed' not in result: result['viewpoint_changed'] = (result['current_viewpoint'] != own_analysis.get('answer'))
            if 'justification_type' not in result: result['justification_type'] = 'unknown'
            if 'cited_references' not in result: result['cited_references'] = []
            if 'reason' not in result: result['reason'] = "Fallback: No reason provided."

        self.memory.append({"type": "review", "content": result})
        return {"parsed_output": result, "llm_input": {"system_message": system_msg, "user_message": user_msg}}


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions."""

    def __init__(self, agent_id: str, model_key: str = "qwen-vl-max", config_path: str = "config.toml"):
        super().__init__(agent_id=agent_id, agent_type=AgentType.META, model_key=model_key, config_path=config_path)
        print(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    def synthesize_opinions(self,
                           question: str,
                           doctor_opinions: List[Dict[str, Any]],
                           doctor_specialties: List[MedicalSpecialty],
                           audit_trail: Dict[str, Any],
                           ccp_text: str = "",
                           options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
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
        
        options_text = ""
        if options:
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
        question_with_options = f"{question}{options_text}"

        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed so far:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"

        # [TODO 已解决] user_message要将keu与ccp注入
        user_message = {
            "role": "user",
            "content": f"""Question: {question_with_options}

Doctors' Opinions:
{opinions_text}

Available KEUs (Key Evidential Units):
{keu_list_text}

Available CCPs (Critical Consensus Points):
{ccp_text}

Note that potential conflicts (CCPs) have been identified; a robust synthesis must acknowledge or resolve these points.

**CRITICAL INSTRUCTION:** Your task is to synthesize these diverse opinions into a single, coherent analysis. In your 'explanation', you **MUST selectively cite only the most important KEU-IDs** that support your synthesized view. **DO NOT simply list all available KEUs.** Your goal is to demonstrate a deep understanding by building a new, consolidated argument from the strongest evidence (e.g., 'Synthesizing the specialists' views, the consensus leans towards X, primarily supported by the crucial findings in KEU-2 and KEU-5...').

Provide your synthesis in JSON format, including 'explanation' (comprehensive reasoning) and 'answer' (clear suggested conclusion) fields."""
        }

        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Meta agent synthesis successfully parsed")
        except json.JSONDecodeError:
            print("Meta agent synthesis is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
        
        self.memory.append({"type": "synthesis", "content": result})
        return {"parsed_output": result, "llm_input": {"system_message": system_msg, "user_message": user_msg}}


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
                      audit_trail: Dict[str, Any],
                      options: Optional[Dict[str, str]] = None,
                      image_path: Optional[str] = None) -> Dict[str, Any]:
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
        
        keu_list_text = "\n\nKey Evidential Units (KEUs) available:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"

        # [TODO 已解决] user_message中要把keu与ccp注入
        text_content = {
            "type": "text",
            "text": f"""Question: {question_with_options}

{synthesis_text}

Specialists' Final Reviews on the Synthesis:
{reviews_text}

{keu_list_text}

**CRITICAL INSTRUCTION:** Your reasoning in the 'explanation' field must demonstrate a final, decisive synthesis. To do this, you **MUST selectively cite only the most pivotal KEU-IDs** that form the core basis of your conclusion. **DO NOT list or repeat all available KEUs.** Your task is to build a definitive argument using the strongest evidence.

Based on ALL available information presented above, provide your final decision. Your response must be in JSON format, including 'explanation' and 'answer' fields."""
        }
        user_content.append(text_content)

        user_message = {"role": "user", "content": user_content}
        response_text, system_msg, user_msg = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Decision making agent response successfully parsed")
        except json.JSONDecodeError:
            print("Decision making agent response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            
        self.memory.append({"type": "decision", "content": result})
        return {"parsed_output": result, "llm_input": {"system_message": system_msg, "user_message": user_msg}}

class AuditorAgent(BaseAgent):
    def __init__(self, agent_id: str = "auditor", model_key: str = "gemini-2.5-pro", config_path: str = "config.toml"):
        super().__init__(agent_id=agent_id, agent_type=AgentType.AUDITOR, model_key= model_key, config_path=config_path)
        print(f"Initializing auditor agent, ID: {agent_id}, Model: {model_key}")

    def audit_domain_agent_contribution(self, question: str, agent_id: str, specialty: MedicalSpecialty, explanation: str) -> Dict[str, Any]:
        """在领域智能体发言后，评估其角色知识一致性和专家相关性。"""
        print(f"Auditor Agent: Auditing Domain Agent Contribution for {agent_id}...")
        system_message = {
            "role": "system",
            "content": """You are an expert in medical epistemology and collaborative intelligence. Your task is to analyze an argument from a specialist AI doctor and assess two key dimensions of their contribution.
You MUST provide a JSON object with two classifications:
1.  **`specialized_insight_emergence`**: Classify the degree to which the argument demonstrates the emergence of insights unique to the agent's assigned specialty, beyond general medical knowledge.
    - **"High"**: The reasoning presents a perspective, interpretation, or piece of knowledge that is highly specific to the assigned role and would likely not be offered by other specialists. It represents a unique, valuable contribution.
    - **"Medium"**: The reasoning contains some specialty-specific elements but is largely grounded in shared or overlapping medical knowledge.
    - **"Low"**: The reasoning is generic, lacks a distinct specialty perspective, and could have been generated by a generalist agent. No unique insight has emerged.
2.  **`expertise_relevance_category`**: Classify the relevance of this agent's specialty to the overall question.
    - **"Core"**: The specialty is central to diagnosing the problem.
    - **"Relevant"**: The specialty provides important, but not central, insights.
    - **"Ancillary"**: The specialty is only tangentially related.
Provide a concise `auditor_reasoning` explaining your choices."""
        }
        user_message = {
            "role": "user",
            "content": f"Medical Question: \"{question}\"\n\nAgent: {agent_id} (Specialty: {specialty.value})\nArgument/Explanation:\n\"{explanation}\"\n\nPlease provide your audit in the specified JSON format."
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try: return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError): return {}

    def audit_risk_and_quality(self, agent_id: str, explanation: str, image_path:str = None) -> Dict[str, Any]:
        """在任何智能体发言后，评估其论据的风险规避类别。"""
        print(f"Auditor Agent: Auditing Risk and Quality for {agent_id}'s argument...")
        system_message = {
            "role": "system",
            "content": """You are a senior attending physician specializing in emergency medicine and patient triage. Your task is to analyze a medical argument and classify its implied **Diagnostic Urgency Level**.
This level reflects how quickly the argument suggests action should be taken, especially when faced with potentially high-risk diagnoses.
You MUST provide a JSON object with one classification:
- **`diagnostic_urgency_level`**:
  - **"Immediate (STAT)"**: The argument demands immediate, urgent action to investigate or rule out a high-risk, time-sensitive condition. This aligns with the 'worst-case-first' principle.
  - **"Standard (Routine)"**: The argument proposes a standard, routine diagnostic workup based on the most probable causes. It is diligent but not urgent.
  - **"Delayed (Deferrable)"**: The argument suggests a passive or delayed course of action, such as "watchful waiting" or follow-up at a later date, downplaying the need for immediate investigation.
Provide a concise `auditor_reasoning` for your choice."""
        }
        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            image_url_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            user_content.append(image_url_content)
        
        text_content = (
            f"Argument from Agent {agent_id}:\n\"{explanation}\"\n\nPlease provide your risk audit in the specified JSON format."
            )
        user_content.append({"type": "text", "text": text_content})
        user_message = {"role": "user", "content": user_content}
        response_text, _, _ = self.call_llm(system_message, user_message)
        try: return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError): return {}

    def audit_overall_quality_for_decision(self, question: str, doctor_reviews: List[Dict[str, Any]], specialties: List[MedicalSpecialty]) -> List[Dict[str, Any]]:
        """在决策前，对每个领域智能体的当前论据进行综合质量评估。"""
        print("Auditor Agent: Auditing overall argument quality for decision-making...")
        system_message = {
            "role": "system",
            "content": """You are a lead physician and medical logician. Your task is to provide an **Overall Quality Category** for several arguments, to inform a final decision.
The Overall Quality considers all factors: logical soundness, evidence support, expertise relevance, and clinical safety.
- **"High"**: A very strong, reliable argument. It is logical, evidence-based, safe, and comes from a relevant perspective.
- **"Medium"**: A decent argument with some strengths but also notable weaknesses (e.g., logical gaps, ignores some risks).
- **"Low"**: A weak or dangerous argument that should be treated with caution.
For each doctor, you MUST provide a JSON object with:
1.  **`agent_id`**: The doctor's ID.
2.  **`overall_quality_category`**: "High", "Medium", or "Low".
3.  **`auditor_reasoning`**: A concise justification.
Your final output MUST be a JSON list of these objects."""
        }
        arguments_text = ""
        for i, review in enumerate(doctor_reviews):
            specialty_value = specialties[i].value
            arguments_text += f"\n---\nAgent ID: doctor_{i+1} (Specialty: {specialty_value}):\n"
            arguments_text += f"Supported Answer: {review.get('current_viewpoint', 'N/A')}\n"
            arguments_text += f"Reasoning: {review.get('reason', 'N/A')}\n"
        user_message = { "role": "user", "content": f"Medical Question: {question}\n\nArguments:\n{arguments_text}\n\nPlease provide the overall quality audit as a JSON list." }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try: return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError): return []

    def audit_single_argument_quality(self, question: str, explanation: str, image_path: str = None, domain_agent_quality: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """对单个论据（特别是元智能体和决策智能体的）进行综合质量评估。"""
        print("Auditor Agent: Auditing single argument's overall quality...")
        system_message = {
            "role": "system",
            "content": """You are a lead physician and medical logician. Your task is to provide an **Overall Quality Category** for a given medical argument.
The Overall Quality considers all factors: logical soundness, evidence support, and clinical safety.
- **"High"**: A very strong, reliable argument. It is logical, evidence-based, safe, and provides a comprehensive justification.
- **"Medium"**: A decent argument with some strengths but also notable weaknesses (e.g., logical gaps, ignores some risks, superficial reasoning).
- **"Low"**: A weak or dangerous argument that should be treated with caution.
You MUST provide a JSON object with:
1.  **`overall_quality_category`**: "High", "Medium", or "Low".
2.  **`auditor_reasoning`**: A concise justification for your rating."""
        }

        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            image_url_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            user_content.append(image_url_content)
        if domain_agent_quality:
            individual_quality_text = ""
            for audit in domain_agent_quality:
                individual_quality_text += f"- Agent: {audit.get('agent_id', 'Unknown')}\n"
                individual_quality_text += f"  Quality: {audit.get('overall_quality_category', 'N/A')}\n"
                individual_quality_text += f"  Auditor Reasoning: {audit.get('auditor_reasoning', 'No reasoning provided')}" + "\n"
            text_content = (
                f"Medical Question: \"{question}\"\n\n"
                f"Argument to Evaluate:\n\"{explanation}\"\n\n"
                f"We have evaluated the quality of individual doctors' arguments as follows:\n\"{individual_quality_text}\"\n\n"
                f"Please provide the overall quality audit as a JSON object."
            )
        else:
            text_content = (
                f"Medical Question: \"{question}\"\n\n"
                f"Argument to Evaluate:\n\"{explanation}\"\n\n"
                f"Please provide the overall quality audit as a JSON object."
            )
        user_content.append({"type": "text", "text": text_content})
        user_message = {"role": "user", "content": user_content}
        response_text, _, _ = self.call_llm(system_message, user_message)
        try: return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError): return {}
    
    def identify_critical_conflicts(self, contributions: List[Dict[str, Any]], context_description: str) -> List[Dict[str, Any]]:
        """从一系列文本贡献中识别关键冲突点。"""
        print(f"Auditor Agent: Identifying critical conflict points (CCPs) from {context_description}...")
        valid_contributions = [c for c in contributions if c.get("text", "").strip()]
        if not valid_contributions: return []
        system_message = {
            "role": "system",
            "content": """You are a meticulous and logical medical debate moderator. Your sole task is to read the provided arguments and identify direct, substantive contradictions about verifiable facts or core interpretations.
You MUST ignore minor differences in phrasing. Focus only on clear conflicts (e.g., Feature A is present vs. Feature A is absent; Diagnosis X is likely vs. Diagnosis X is unlikely).
Your final output MUST be a single JSON object containing a single key: "conflicts". The value of "conflicts" must be a list of conflict objects. 
Each conflict object must have the following structure:
- "conflicting_agents": A list of the agent_ids involved in this specific conflict.
- "conflict_summary": A brief, one-sentence summary of the core disagreement.
- "conflicting_statements": A list of objects, each detailing the specific statement, with keys "agent_id" and "statement_content".
If there are no conflicts, return a JSON object with an empty list: {"conflicts": []}."""
        }
        context_text = f"Please analyze the following {context_description} for conflicts:\n\n"
        for contrib in valid_contributions:
            context_text += f"--- Argument from {contrib['agent_id']} ({contrib.get('specialty', 'N/A')}) ---\n{contrib['text']}\n\n"
        user_message = { "role": "user", "content": context_text }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            parsed_response = json.loads(preprocess_response_string(response_text))
            conflicts = parsed_response.get("conflicts", [])
            print(f"Auditor Agent: Found {len(conflicts)} critical conflict point(s).")
            return conflicts
        except (json.JSONDecodeError, TypeError):
            print("Auditor Agent: Error parsing CCP response from LLM.")
            return []

    def identify_key_evidential_units(self, question: str, doctor_opinions: List[Dict[str, Any]], doctor_agents: List[DoctorAgent], all_keus: List[Dict], image_path: str = None) -> Dict[str, bool]:
        """从所有提出的KEU中，结合医生们的初步分析，判断哪些是“关键”证据。"""
        print("Auditor Agent: Identifying KEY evidential units with full context...")
        system_message = {
            "role": "system",
            "content": """You are a senior medical expert with exceptional diagnostic acumen. Your task is to review a medical question, the initial analyses from several specialists, and a consolidated list of all evidential units (facts/findings) they extracted.
Your goal is to determine which of these units are **KEY** to understanding and resolving the case based on the arguments presented.
A **KEY** evidential unit is one that is:
- Directly foundational to a doctor's primary conclusion.
- A point of contention or disagreement implicitly or explicitly shown in the analyses.
- Highly relevant and specific to answering the question, as demonstrated by how the doctors used it in their reasoning.
- Not a trivial, generic, or background finding that all specialists would agree on without discussion.
Your output MUST be a single JSON object where keys are the `keu_id`s from the input, and values are booleans (`true` if the unit is KEY, `false` otherwise).
Example: {"KEU-0": true, "KEU-1": false, "KEU-2": true}"""
        }
        opinions_context = "Here are the initial analyses from the specialists:\n\n"
        for i, opinion in enumerate(doctor_opinions):
            agent = doctor_agents[i]
            opinions_context += f"--- Analysis from {agent.agent_id} ({agent.specialty.value}) ---\n"
            opinions_context += f"Explanation: {opinion.get('explanation', 'N/A')}\nAnswer: {opinion.get('answer', 'N/A')}\n\n"
        keu_list_text = "\n".join([f"- {keu['keu_id']}: \"{keu['content']}\"" for keu in all_keus])
        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            image_url_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            user_content.append(image_url_content)
        
        text_content = (
            f"**Medical Question:**\n \"{question}\"\n\n**Doctors' Analyses:**\n{opinions_context}**Consolidated List of All Evidential Units to Evaluate:**\n{keu_list_text}\n\nBased on the doctors' analyses, please provide your judgment on which of these are KEY units in the specified JSON format."
        )
        user_content.append({"type": "text", "text": text_content})
        user_message = {"role": "user", "content": user_content}
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            key_status_map = json.loads(preprocess_response_string(response_text))
            for keu in all_keus:
                if keu['keu_id'] not in key_status_map: key_status_map[keu['keu_id']] = False
            return key_status_map
        except (json.JSONDecodeError, TypeError):
            print("Auditor Agent: Error parsing KEU key status response. Defaulting all to not key.")
            return {keu['keu_id']: False for keu in all_keus}

class MDTConsultation:
    """Multi-disciplinary team consultation coordinator."""

    def __init__(self,
                max_rounds: int = 1, # MedAgent原框架为单轮，此处设为1以保持一致，可调
                model_key: str = "qwen-max-latest",
                meta_model_key: str = "qwen-max-latest",
                decision_model_key: str = "qwen-max-latest",
                auditor_model_key: str = "gemini-2.5-pro",
                conflict_analysis_model_key: str = "deepseek-reasoner",
                config_path: str = "config.toml"):
        self.max_rounds = max_rounds
        self.model_key = model_key
        self.config_path = config_path
        # Initialize agents
        self.expert_gatherer = ExpertGathererAgent(agent_id="expert_gatherer", model_key=model_key, config_path=config_path)
        self.meta_agent = MetaAgent(agent_id="meta", model_key=meta_model_key, config_path=config_path)
        self.decision_agent = DecisionMakingAgent(agent_id="decision", model_key=decision_model_key, config_path=config_path)
        self.auditor_agent = AuditorAgent(agent_id="auditor", model_key=auditor_model_key, config_path=config_path)
        self.analysis_llm = AnalysisHelperLLM(model_key=conflict_analysis_model_key, config_path=config_path)

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
                            options: Optional[Dict[str, str]] = None,
                            image_path: Optional[str] = None) -> Dict[str, Any]:
            start_time = time.time()
            print(f"Starting MDT consultation for case {qid}")
            print(f"Question: {question}")
            if options: print(f"Options: {options}")

            # Step 1: Gather relevant domain experts
            specialties, gatherer_log = self.expert_gatherer.gather_question_domain_experts(question)
            print(f"Gathered specialties: {[s.value for s in specialties]}")
            self._initialize_doctor_agents(specialties)

            case_history = {
                "qid": qid,
                "question": question, "options": options, "image_path": image_path,
                "expert_gatherer_log": gatherer_log,
                "selected_specialties": [s.value for s in specialties],
                "rounds": [],
                "final_decision_log": None,
                "consensus_reached": False, "total_rounds": 0, "processing_time": 0,
                'audit_trail': {
                    "keus": {},
                    "viewpoints": {doc.agent_id: [] for doc in self.doctor_agents},
                    "collaboration_audits": {},
                    "ccps": {}
                }
            }
            audit_trail = case_history['audit_trail']
            all_unresolved_ccps = []
            ccp_counter = 0
            current_round = 0
            consensus_reached = False
            final_decision_log = None # 初始化 final_decision_log

            while current_round < self.max_rounds and not consensus_reached:
                current_round += 1
                print(f"Starting round {current_round}")
                
                round_data = {"round": current_round, "analyses": [], "synthesis": None, "reviews": []}

                # Step 2: Each doctor analyzes the case
                doctor_opinion_parsed_outputs = []
                keu_counter = 0
                for i, doctor in enumerate(self.doctor_agents):
                    print(f"Doctor {i+1} ({doctor.specialty.value}) analyzing case")
                    opinion_log = doctor.analyze_case(question, options, image_path)
                    parsed_output = opinion_log["parsed_output"]
                    explanation = parsed_output.get("explanation", "")
                    
                    contribution_audit = self.auditor_agent.audit_domain_agent_contribution(question, doctor.agent_id, doctor.specialty, explanation)
                    risk_audit = self.auditor_agent.audit_risk_and_quality(doctor.agent_id, explanation, image_path=image_path)
                    step_id = f"round_1_analysis_{doctor.agent_id}"
                    audit_trail["collaboration_audits"][step_id] = {**contribution_audit, **risk_audit}
                    doctor_opinion_parsed_outputs.append(parsed_output)
                    
                    initial_viewpoint_entry = {
                        "step": f"round_{current_round}_analysis",
                        "viewpoint": parsed_output.get("answer"),
                        "viewpoint_changed": False,
                        "justification_type": "initial_analysis",
                        "cited_references": [f"KEU-{idx+keu_counter}" for idx, content in enumerate(parsed_output.get("keus", []))]
                    }
                    audit_trail["viewpoints"][doctor.agent_id].append(initial_viewpoint_entry)
                    
                    if "keus" in parsed_output:
                        for keu_content in parsed_output["keus"]:
                            keu_id = f"KEU-{keu_counter}"
                            new_keu = KEU(keu_id, keu_content, doctor.agent_id, current_round)
                            audit_trail["keus"][keu_id] = new_keu
                            keu_counter += 1
                    
                    round_data["analyses"].append({"doctor_id": doctor.agent_id, "specialty": doctor.specialty.value, "log": opinion_log})
                    print(f"Doctor {i+1} opinion: {parsed_output.get('answer', '')}")
                
                all_keus_for_audit = [{"keu_id": k, "content": v.content} for k, v in audit_trail["keus"].items()]
                if all_keus_for_audit:
                    key_status_map = self.auditor_agent.identify_key_evidential_units(question, doctor_opinion_parsed_outputs, self.doctor_agents, all_keus_for_audit, image_path=image_path)
                    for keu_id, is_key in key_status_map.items():
                        if keu_id in audit_trail["keus"]: audit_trail["keus"][keu_id].is_key = is_key

                initial_contributions = [{'agent_id': self.doctor_agents[i].agent_id, 'specialty': self.doctor_agents[i].specialty.value, 'text': op.get('explanation','')} for i, op in enumerate(doctor_opinion_parsed_outputs)]
                initial_ccps = self.auditor_agent.identify_critical_conflicts(initial_contributions, "doctors' initial analyses")
                audit_trail["ccps"][current_round] = []
                for ccp in initial_ccps:
                    ccp.update({'ccp_id': f"CCP-{ccp_counter}", 'round_identified': current_round, 'status': 'unresolved', 'round_resolved': None})
                    audit_trail["ccps"][current_round].append(ccp)
                    ccp_counter += 1
                all_unresolved_ccps.extend(audit_trail["ccps"][current_round])

                ccp_text_for_prompt = ""
                if all_unresolved_ccps:
                    ccp_text_for_prompt += "\n\n[ATTENTION] The following Critical Conflict Points (CCPs) remain UNRESOLVED:\n"
                    for ccp in all_unresolved_ccps:
                        ccp_text_for_prompt += f"- CCP ID: {ccp['ccp_id']}: {ccp['conflict_summary']}\n"

                # Step 3: Meta agent synthesizes opinions
                print("Meta agent synthesizing opinions")
                synthesis_log = self.meta_agent.synthesize_opinions(question, doctor_opinion_parsed_outputs, self.doctor_specialties, audit_trail, ccp_text_for_prompt, options)
                round_data["synthesis"] = synthesis_log
                synthesis_parsed_output = synthesis_log["parsed_output"]
                synthesis_explanation = synthesis_parsed_output.get("explanation", "")
                
                synthesis_risk_audit = self.auditor_agent.audit_risk_and_quality(self.meta_agent.agent_id, synthesis_explanation, image_path=image_path)
                synthesis_quality_audit = self.auditor_agent.audit_single_argument_quality(question, synthesis_explanation, image_path=image_path)
                audit_trail["collaboration_audits"][f"round_{current_round}_synthesis"] = {**synthesis_risk_audit, **synthesis_quality_audit}
                for keu_id, keu in audit_trail["keus"].items():
                    if keu_id in synthesis_explanation or keu.content in synthesis_explanation:
                        keu.present_in_synthesis[current_round] = True
                        keu.cited_by.append({"agent_id": self.meta_agent.agent_id, "round": current_round, "action": "synthesis"})
                    else:
                        keu.present_in_synthesis[current_round] = False

                # Step 4: Doctors review synthesis
                doctor_review_parsed_outputs = []
                all_agree = True
                for i, doctor in enumerate(self.doctor_agents):
                    print(f"Doctor {i+1} ({doctor.specialty.value}) reviewing synthesis")
                    review_log = doctor.review_synthesis(question, synthesis_parsed_output, audit_trail, ccp_text_for_prompt, options, image_path)
                    review_parsed_output = review_log["parsed_output"]
                    review_reason = review_parsed_output.get("reason", "")
                    cited_refs = review_parsed_output.get("cited_references", [])
                    
                    contribution_audit = self.auditor_agent.audit_domain_agent_contribution(question, doctor.agent_id, doctor.specialty, review_reason)
                    risk_audit = self.auditor_agent.audit_risk_and_quality(doctor.agent_id, review_reason, image_path=image_path)
                    audit_trail["collaboration_audits"][f"round_{current_round}_review_{doctor.agent_id}"] = {**contribution_audit, **risk_audit}
                    
                    for keu_id, keu in audit_trail["keus"].items():
                        if keu_id in cited_refs or keu_id in review_reason or keu.content in review_reason:
                            keu.cited_by.append({"agent_id": doctor.agent_id, "round": current_round, "action": "review"})
                        if not review_parsed_output.get("agree", True) and (keu_id in review_reason or keu.content in review_reason):
                            keu.rebuttals.append({"agent_id": doctor.agent_id, "round": current_round, "reason": review_reason})
                    
                    doctor_review_parsed_outputs.append(review_parsed_output)
                    round_data["reviews"].append({"doctor_id": doctor.agent_id, "specialty": doctor.specialty.value, "log": review_log})
                    
                    review_viewpoint_entry = {
                        "step": f"round_{current_round}_review", "viewpoint": review_parsed_output.get("current_viewpoint"),
                        "viewpoint_changed": review_parsed_output.get("viewpoint_changed", False),
                        "justification_type": review_parsed_output.get("justification_type", "unknown"),
                        "cited_references": cited_refs
                    }
                    audit_trail["viewpoints"][doctor.agent_id].append(review_viewpoint_entry)
                    
                    agrees = review_parsed_output.get('agree', False)
                    all_agree = all_agree and agrees
                    print(f"Doctor {i+1} agrees: {'Yes' if agrees else 'No'}")

                overall_quality_audit = self.auditor_agent.audit_overall_quality_for_decision(question, doctor_review_parsed_outputs, self.doctor_specialties)
                audit_trail["collaboration_audits"][f"round_{current_round}_pre_decision_quality"] = overall_quality_audit

                round_discussion_text = synthesis_explanation + "".join([r['log']['parsed_output'].get("reason","") for r in round_data["reviews"]])
                still_unresolved = []
                for ccp in all_unresolved_ccps:
                    was_addressed, resolution_reasoning = self.analysis_llm.check_if_conflict_was_addressed(ccp, round_discussion_text)
                    if was_addressed:
                        ccp.update({'status': 'resolved', 'round_resolved': current_round, 'resolution_reasoning': resolution_reasoning})
                    else:
                        still_unresolved.append(ccp)
                all_unresolved_ccps = still_unresolved

                review_contributions = [{'agent_id': r["doctor_id"], 'specialty': r["specialty"], 'text': r["log"]["parsed_output"].get("reason","")} for r in round_data["reviews"]]
                new_ccps = self.auditor_agent.identify_critical_conflicts(review_contributions, "doctors' review reasons")
                if current_round not in audit_trail["ccps"]: audit_trail["ccps"][current_round] = []
                for ccp in new_ccps:
                    ccp.update({'ccp_id': f"CCP-{ccp_counter}", 'round_identified': current_round, 'status': 'unresolved', 'round_resolved': None})
                    audit_trail["ccps"][current_round].append(ccp)
                    ccp_counter += 1
                all_unresolved_ccps.extend(new_ccps)
                
                consensus_reached = all_agree
                case_history["rounds"].append(round_data)
                
                # --- 修正点 1: 将决策逻辑移入循环内的终止条件判断中 ---
                if consensus_reached or current_round == self.max_rounds:
                    print("Proceeding to final decision.")
                    
                    # Step 5: Decision making agent provides final answer
                    final_decision_log = self.decision_agent.make_decision(question, synthesis_parsed_output, doctor_review_parsed_outputs, self.doctor_specialties, audit_trail, options, image_path)
                    decision_explanation = final_decision_log.get("parsed_output", {}).get("explanation", "")
                    
                    # 机制三和机制一的最终审计
                    decision_quality_audit = self.auditor_agent.audit_single_argument_quality(question, decision_explanation, image_path=image_path, domain_agent_quality=overall_quality_audit)
                    decision_risk_audit = self.auditor_agent.audit_risk_and_quality(self.decision_agent.agent_id, decision_explanation, image_path=image_path)
                    audit_trail["collaboration_audits"][f"final_decision"] = {**decision_risk_audit, **decision_quality_audit}
                    for keu_id, keu in audit_trail["keus"].items():
                        if keu_id in decision_explanation or keu.content in decision_explanation:
                            keu.present_in_final_decision = True

                    # --- 修正点 2: 解决 "机制四：判断遗留的冲突是否被解决" 的 TODO ---
                    print("Performing final conflict resolution check against the decision explanation...")
                    for ccp in all_unresolved_ccps:
                        # 检查最终决策的解释是否解决了这个冲突
                        was_addressed, resolution_reasoning = self.analysis_llm.check_if_conflict_was_addressed(ccp, decision_explanation)
                        if was_addressed:
                            # 如果解决，更新其在audit_trail中的状态
                            # 注意：直接修改ccp对象会更新audit_trail中的对应项
                            ccp['status'] = 'resolved'
                            ccp['round_resolved'] = 'final_decision' # 标记为在最终决策中解决
                            ccp['resolution_reasoning'] = resolution_reasoning
                            print(f"CCP {ccp['ccp_id']} was resolved in the final decision.")
                    
                    break # 决策已做出，跳出循环
                else:
                    print("No consensus reached, continuing to next round.")

            # --- 循环外的收尾工作 ---
            if final_decision_log:
                print(f"Final answer: {final_decision_log['parsed_output'].get('answer', 'N/A')}")
            else:
                # 这种情况理论上不应发生，但作为保险
                print("Consultation ended without a final decision being generated.")

            # Finalize history
            processing_time = time.time() - start_time
            if "keus" in audit_trail and audit_trail["keus"]:
                # 将 KEU 对象序列化为字典以便保存为 JSON
                serializable_keus = {keu_id: keu.to_dict() for keu_id, keu in audit_trail["keus"].items()}
                audit_trail["keus"] = serializable_keus
            
            case_history.update({
                "final_decision_log": final_decision_log,
                "consensus_reached": consensus_reached,
                "total_rounds": current_round,
                "processing_time": processing_time
            })

            return case_history

def parse_structured_output(response_text: str) -> Dict[str, str]:
    """Fallback parser for non-JSON LLM responses."""
    try:
        return json.loads(preprocess_response_string(response_text))
    except json.JSONDecodeError:
        lines = response_text.strip().split('\n')
        result = {}
        current_key = None
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace("'", "").replace('"', '')
                if key in ['explanation', 'answer', 'reason', 'current_viewpoint', 'justification_type']:
                    current_key = key
                    result[current_key] = value.strip()
                else: # Handle nested content
                    if current_key: result[current_key] += " " + line.strip()
            else:
                if current_key: result[current_key] += " " + line.strip()
        
        # Ensure mandatory fields
        if "explanation" not in result and "reason" not in result:
            result["explanation"] = response_text
        if "answer" not in result and "current_viewpoint" not in result:
            result["answer"] = "No answer found"
        return result


def process_input(item, model_key, meta_model_key, decision_model_key, auditor_model_key, conflict_analysis_model_key, config_path):
    """Process a single input data item."""
    mdt = MDTConsultation(
        max_rounds=3,
        model_key=model_key,
        meta_model_key=meta_model_key,
        decision_model_key=decision_model_key,
        auditor_model_key=auditor_model_key,
        conflict_analysis_model_key=conflict_analysis_model_key,
        config_path=config_path
    )
    return mdt.run_consultation(
        qid=item.get("qid"),
        question=item.get("question"),
        options=item.get("options"),
        image_path=item.get("image_path"),
    )


def main():
    parser = argparse.ArgumentParser(description="Run MDT consultation on medical datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], default="mc", help="QA type")
    parser.add_argument("--model", required=True, type=str, help="Model for doctor agents") # meta model = domain model
    parser.add_argument("--meta_model", required=True, type=str, help="Model for meta agent")
    parser.add_argument("--decision_model", required=True, type=str, help="Model for decision agent")
    parser.add_argument("--auditor_model", type=str, required=True, help="Model for auditor agent")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to process")
    parser.add_argument("--config_path", type=str, default="config.toml", help="Path to config file")
    parser.add_argument("--test_mode", type=bool, required=True, help="If set, log will be saved to a test-specific directory.")
    args = parser.parse_args()

    test_mode = args.test_mode
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if test_mode:
        terminal_log_dir = os.path.join("logs", "observation", "test", "terminal_log", "MedAgent", args.dataset)
    else:
        terminal_log_dir = os.path.join("logs", "observation", "terminal_log", "MedAgent", args.dataset)
    os.makedirs(terminal_log_dir, exist_ok=True)
    terminal_log_file = os.path.join(terminal_log_dir, f"{args.dataset}_{timestamp}_full_terminal.log")
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条

    if test_mode:
        logs_dir = os.path.join("logs", "observation", "test", "MedAgent", args.dataset)
    else:
        logs_dir = os.path.join("logs", "observation", "MedAgent", args.dataset)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Logs will be saved to: {logs_dir}")

    data_path = f"./my_datasets/processed/medqa/{args.dataset}/medqa_{args.qa_type}_test.json"
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    for item in tqdm(data[:args.num_samples], desc=f"Running MDT on {args.dataset}"):
        qid = item["qid"]
        log_file_path = os.path.join(logs_dir, f"{qid}-result.json")
        if os.path.exists(log_file_path):
            print(f"Skipping {qid} - already processed")
            continue

        try:
            full_case_history = process_input(
                item,
                model_key=args.model,
                meta_model_key=args.meta_model,
                decision_model_key=args.decision_model,
                auditor_model_key=args.auditor_model,
                conflict_analysis_model_key=args.auditor_model,
                config_path=args.config_path
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
            save_json(item_result, log_file_path)

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            error_log = {"qid": qid, "error": str(e)}
            save_json(error_log, os.path.join(logs_dir, f"{qid}-error.json"))


if __name__ == "__main__":
    main()