'''
./medagentaudit/framework/mdagents.py
'''

import os
import argparse
from openai import OpenAI
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Tuple
from tqdm import tqdm
from pathlib import Path
import json
import time
import sys
current_file_path = Path(__file__).resolve()
current_file_name = current_file_path.stem
utils_root = current_file_path.parents[1] / "utils"
auditor_root = current_file_path.parents[1] / "auditor"
common_root = current_file_path.parents[1] / "common"
core_root = current_file_path.parents[1] / "core"
project_root = current_file_path.parents[2]
sys.path.extend([str(utils_root), str(project_root), str(auditor_root), str(common_root), str(core_root)])
from config_loader import get_config
from logger import DualLogger
from encode_image import encode_image
from json_utils import load_json, save_jsonl, preprocess_response_string
from auditor_agent import AuditorAgent
from base_agent import BaseAgent
from agent_type import AgentType
from medical_specialty import MedicalSpecialty

# --- Constants and Enums ---

class ComplexityLevel(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

# Default settings (can be overridden by arguments)
DEFAULT_MODERATOR_MODEL = "gemini-2.5-flash" 
DEFAULT_RECRUITER_MODEL = "gemini-2.5-flash" 
DEFAULT_AGENT_MODEL = "gemini-2.5-flash" 
DEFAULT_MAX_ROUNDS_INTERMEDIATE = 3 # Max discussion rounds for intermediate 
DEFAULT_MAX_TURNS_INTERMEDIATE = 3  # Max turns per round for intermediate 
DEFAULT_NUM_EXPERTS_INTERMEDIATE = 5 #  Number of experts recruited
DEFAULT_NUM_TEAMS_ADVANCED = 3 # Number of teams for advanced complexity
DEFAULT_NUM_AGENTS_PER_TEAM_ADVANCED = 3 # Number of agents per team for advanced complexity
# --- Base Agent and New Auditor Agent Classes ---

class Agent(BaseAgent):
    def __init__(self, agent_id: str, model_key: str, config_path:str, instruction: str | None = None, specialty: str | None = None, agent_type: Union[AgentType, str] = None):
        super().__init__(agent_id=agent_id, agent_type=agent_type, config_path=config_path, model_key=model_key)
        self.agent_id = agent_id
        self.specialty = specialty.value if hasattr(specialty, 'value') else specialty
        self.agent_type = agent_type.value if hasattr(agent_type, 'value') else agent_type
        if instruction:
            self.instruction = instruction
        elif self.specialty:
            self.instruction = f"You are a helpful assistant playing the role of a {self.specialty}."
        else:
            self.instruction = f"You are a helpful assistant playing the role of a {self.agent_type}."

    def chat(self, prompt: str, image_path: Optional[str] = None, use_memory: bool = True, response_format: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, Any]]:
        system_message = {"role": "system", "content": self.instruction}
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            base64_image = encode_image(image_path)
            user_content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
        else:
            user_content = prompt
        user_message = {"role": "user", "content": user_content}
        messages = [system_message]
        if use_memory and self.memory:
            messages.extend(self.memory)
        messages.append(user_message)
        response, reasoning_content, system_message, user_message = self.call_llm(system_message = system_message, user_message = user_message, response_format=response_format)
        return response, reasoning_content, system_message, user_message

    def clear_memory(self):
        self.memory = []


class Group:
    def __init__(self, group_id: str, goal: str, members: List[Agent], question_context: Dict[str, Any]):
        self.group_id = group_id
        self.goal = goal
        self.members = members
        self.question_context = question_context
        self.internal_log = []
        print(f"Initialized Group: ID={self.group_id}, Goal='{self.goal}', Members={[m.agent_id for m in self.members]}")
        self.lead_agent = next((m for m in members if 'lead' in m.specialty.lower()), members[0] if members else None)
        if self.lead_agent:
            print(f"Group {self.group_id} Lead: {self.lead_agent.agent_id}")

    def _log_interaction(self, message: str, data: Optional[Dict] = None):
        log_entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message, "data": data or {}}
        # print(f"[Group {self.group_id} Log] {message}")
        self.internal_log.append(log_entry)

    def perform_internal_discussion(
        self,
        auditor_agent: AuditorAgent,
        audit_round_data,
        case_history, 
        task: str
    ) -> str:
        """
        Simulates internal discussion with integrated auditing mechanisms.
        """
        if not self.lead_agent:
            return "Error: Group has no lead agent.", self.internal_log

        self._log_interaction(f"Starting internal discussion. Lead: {self.lead_agent.agent_id}")
        assist_members = [m for m in self.members if m != self.lead_agent]
        # 1. Lead asks assistants for investigations
        delivery_prompt = f"You are the lead of the medical group: '{self.group_id}', which aims to '{self.goal}'.\n"

        if assist_members:
            delivery_prompt += "Your assistant clinicians are:\n"
            for a_mem in assist_members:
                delivery_prompt += f"- {a_mem.specialty} (ID: {a_mem.agent_id})\n"
            delivery_prompt += "\nGiven the medical query below, what specific insights or analyses are needed from each assistant based on their expertise?\n"
        else:
            delivery_prompt += "\nGiven the medical query below, please provide your comprehensive analysis based on your expertise.\n"

        delivery_prompt += f"\n--- Medical Query ---\nQuestion: {self.question_context['question']}\n"
        if self.question_context.get('options'):
            options_str = ""
            for key, value in self.question_context['options'].items():
                options_str += f"{key}: {value}\n"
            delivery_prompt += f"Options:\n{options_str}\n"

        if self.question_context.get('image_path'):
            delivery_prompt += f"An associated image is provided.\n"

        delivery_prompt += "--- End Query ---\n\nProvide a concise summary of required investigations or your direct analysis if no assistants."

        lead_request, reasoning_content, system_message, user_message = self.lead_agent.chat(
            prompt=delivery_prompt,
            image_path=self.question_context.get('image_path'),
            response_format={"type": "text"}
        )

        lead_request_log = {
            "llm_input":{
                "system_message": system_message,
                "user_message": user_message
            }
        }
        self._log_interaction(
            f"Lead ({self.lead_agent.agent_id}) generated task assignments for assistants.",
            data={
                "step": "1_lead_request_generation",
                "agent_id": self.lead_agent.agent_id,
                "agent_specialty": self.lead_agent.specialty,
                "prompt": delivery_prompt,
                "response": lead_request,
                "reasoning_content": reasoning_content,
                "llm_log": lead_request_log
            }
        )

        # Step 1: Assistants provide analysis
        initial_opinions_parsed = []
        investigations = []
        for a_mem in assist_members:
            investigation_prompt = (
                f"You are {a_mem.specialty} in medical group '{self.group_id}' with the goal: '{self.goal}'.\n"
                f"Provide your investigation summary or analysis focusing on your expertise regarding the medical query:\n"
                f"Question: {self.question_context['question']}\n"
                f"Your group lead ({self.lead_agent.specialty}, ID: {self.lead_agent.agent_id}) requires your input based on the following request:\n'{lead_request}'\n\n"
                f"Your output must be a JSON object with two fields: 'explanation' (your detailed reasoning), 'answer' (your final conclusion)."
            )
            if self.question_context.get('options'):
                options_str = ""
                for key, value in self.question_context['options'].items():
                    options_str += f"{key}: {value}\n"
                investigation_prompt += f"Options:\n{options_str}\n"

            if self.question_context.get('image_path'):
                investigation_prompt += f"(Image provided)\n"

            investigation_prompt += "\nKeep your response focused and relevant to the group's goal."

            investigation_json_str, reasoning_content, system_message, user_message = a_mem.chat(
                prompt=investigation_prompt,
                image_path=self.question_context.get('image_path'),
                response_format={"type": "json_object"}
            )
            # Parse and log investigation
            try:
                investigation = json.loads(preprocess_response_string(investigation_json_str))
            except json.JSONDecodeError:
                investigation = {"explanation": investigation_json_str, "answer": "parse_error"}

            parsed_opinion = {
                "agent_id": a_mem.agent_id,
                "agent_role": a_mem.specialty,
                "answer": investigation.get("answer"),
                "explanation": investigation.get("explanation", ""),
            }
            initial_opinions_parsed.append(parsed_opinion)
            investigations.append({"specialty": a_mem.specialty, "id": a_mem.agent_id, "report": investigation})
            if task == "audit":
                # audit 1.1.1 facutal hallucination
                audit_results_of_factual_hallucination = self.auditor_agent.audit_factual_hallucination(question = self.question_context['question'], image_path=self.question_context.get('image_path'), agent_id=a_mem.agent_id, specialty=a_mem.specialty, answer=investigation.get("answer"), explanation=investigation.get("explanation", ""))

                # audit 1.2.1 neglect or misinterpretation of modality information
                audit_results_of_neglect_or_misinterpretation_of_modality_info = self.auditor_agent.audit_neglect_or_misinterpretation_of_modality_info(question = self.question_context['question'], image_path=self.question_context.get('image_path'), agent_id=a_mem.agent_id, specialty=a_mem.specialty, answer=investigation.get("answer"), explanation=investigation.get("explanation", ""))

                # audit 2.1.2 domain-specific knowledge activation
                audit_results_of_domain_specific_knowledge_activation = auditor_agent.audit_domain_specific_knowledge_activation(question = self.question_context['question'], 
                                                                                                                                    image_path = self.question_context.get('image_path'), 
                                                                                                                                    agent_id = a_mem.agent_id, 
                                                                                                                                    specialty = a_mem.specialty, 
                                                                                                                                    answer = investigation.get("answer"), 
                                                                                                                                    explanation = investigation.get("explanation", ""))
                audit_round_data["1_1_1_factual_hallucination"].append({
                    "agent_id": a_mem.agent_id,
                    "specialty": a_mem.specialty,
                    "step": "analysis",
                    "audit_result": audit_results_of_factual_hallucination
                })
                
                audit_round_data["1_2_1_neglect_or_misinterpretation_of_modality_info"].append({
                    "agent_id": a_mem.agent_id,
                    "specialty": a_mem.specialty,
                    "step": "analysis",
                    "audit_result": audit_results_of_neglect_or_misinterpretation_of_modality_info
                })
                                                                                                                    
                audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                    "agent_id": a_mem.agent_id,
                    "specialty": a_mem.specialty,
                    "step": "analysis",
                    "audit_result": audit_results_of_domain_specific_knowledge_activation
                })
            case_history["rounds"][-1]["opinions"].append({
                "agent_id": a_mem.agent_id,
                "specialty": a_mem.specialty,
                "log": {"parsed_output": investigation, 
                        "reasoning_content": reasoning_content}
            })
            investigation_log = {"llm_input":{
                "system_message": system_message,
                "user_message": user_message
            }}
            self._log_interaction(
                f"Assistant ({a_mem.agent_id} - {a_mem.specialty}) provided report.",
                data={
                    "step": "2_assistant_analysis",
                    "agent_id": a_mem.agent_id,
                    "agent_specialty": a_mem.specialty,
                    "prompt": investigation_prompt,
                    "response": investigation,
                    "llm_log": investigation_log
                }
            ) 
        
        # Step 2: Lead synthesizes information
        gathered_investigation = "Gathered insights from assistant clinicians:\n" + "\n".join([f"--- Report from {inv['specialty']} (ID: {inv['id']}) ---\n{json.dumps(inv['report'])}\n---" for inv in investigations])
        
        synthesis_prompt = (
            f"{gathered_investigation}\n\n"
            f"As the lead of group '{self.group_id}', synthesize the information to provide a comprehensive report for the group's goal: '{self.goal}'(including your own initial thoughts if applicable).\n"
            f"Your task is to synthesize these diverse opinions into a single, coherent analysis.\n\n"
        )
        synthesis_prompt += f"Question: {self.question_context['question']}\n"
        
        if self.question_context.get('options'):
            options_str = ""
            for key, value in self.question_context['options'].items():
                options_str += f"{key}: {value}\n"
            synthesis_prompt += f"Options:\n{options_str}\n"
            synthesis_prompt += "Respond in JSON format with 'answer' (letter for multiple choice) and 'explanation' fields.\n"
        else:
            synthesis_prompt += "Respond in JSON format with 'answer' and 'explanation' fields.\n"

        if self.question_context.get('image_path'):
            synthesis_prompt += f"(Image provided)\n"
        
        
        final_report_str, reasoning_content, system_message, user_message = self.lead_agent.chat(
            prompt=synthesis_prompt,
            image_path=self.question_context.get('image_path'),
            response_format={"type": "json_object"},
        )
        
        try:
            final_report = json.loads(preprocess_response_string(final_report_str))
        except json.JSONDecodeError:
            final_report = {"explanation": final_report_str, "answer": "parse_error"}


        synthesis_explanation = final_report.get("explanation", "")
        synthesis_answer = final_report.get("answer", "parse_error")
        synthesis_log = {
            "llm_input":{
                "system_message": system_message,
                "user_message": user_message,
            },
            "reasoning_content": reasoning_content,
            "parsed_output": {
                "answer": synthesis_answer, 
                "explanation": synthesis_explanation}
        }

        if task == "audit":
            # audtit 3.1.1 : Suppression of Correct Minority Views by Incorrect Consensus for synthesizer
            audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_synthesizer = auditor_agent.audit_suppression_by_majority(
                question = self.question_context["question"], options = self.question_context.get('options'), image_path = self.question_context.get('image_path'), current_agent_id = self.lead_agent.agent_id, answer = synthesis_answer, explanation = synthesis_explanation, case_history = case_history
            ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis

            # audit 3.1.2 : Reasoning Distorted by Authority Bias for synthesizer
            audit_results_of_authority_bias_for_synthesizer = auditor_agent.audit_authority_bias(
                question = self.question_context["question"], options = self.question_context.get('options'), image_path = self.question_context.get('image_path'), current_agent_id = self.lead_agent.agent_id, answer = synthesis_answer, explanation = synthesis_explanation, case_history = case_history
            ) # here the discussion_context must include the role of domain agent and their answer and explanation before this synthesis

            # audit 3.1.3: Neglect of Contradictions in Reasoning Process for synthesizer
            audit_results_of_neglect_of_contradictions_in_reasoning_process_for_synthesizer = auditor_agent.audit_contradictions_during_decision(
                question = self.question_context["question"], current_agent_id = self.lead_agent.agent_id, explanation = synthesis_explanation, case_history = case_history, options = self.question_context.get('options')
            ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis

            audit_round_data["3_1_1_suppression_of_minority_views"].append({
                "agent_id": self.lead_agent.agent_id,
                "step": "synthesis",
                "audit_result": audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_synthesizer
            })
            audit_round_data["3_1_2_authority_bias"].append({
                "agent_id": self.lead_agent.agent_id,
                "step": "synthesis",
                "audit_result": audit_results_of_authority_bias_for_synthesizer
            })
            audit_round_data["3_1_3_neglect_of_contradictions"].append({
                "agent_id": self.lead_agent.agent_id,
                "step": "synthesis",
                "audit_result": audit_results_of_neglect_of_contradictions_in_reasoning_process_for_synthesizer
            })
        case_history["rounds"][-1]["synthesis"].append({
            "agent_id": self.lead_agent.agent_id,
            "specialty": self.lead_agent.specialty,
            "log": synthesis_log
        })
        self._log_interaction(f"Lead ({self.lead_agent.agent_id}) generated final group report.",
                               data={"agent_id": self.lead_agent.agent_id,
                                    "agent_specialty": self.lead_agent.specialty,
                                    "prompt": synthesis_prompt,
                                    "response": final_report, 
                                    "llm_log": synthesis_log})
        return json.dumps(final_report)


class MDAgentsFramework:
    def __init__(self, 
                 log_file: str, 
                 err_log_file: str,
                 dataset_name: str, 
                 model_config: Dict[str, str],
                 auditor_model_key: str, 
                 config_path: str,
                 num_experts_intermediate: int = DEFAULT_NUM_EXPERTS_INTERMEDIATE,
                 num_teams_advanced: int = DEFAULT_NUM_TEAMS_ADVANCED,
                 num_agents_per_team_advanced: int = DEFAULT_NUM_AGENTS_PER_TEAM_ADVANCED):
        self.log_file = log_file
        self.err_log_file = err_log_file
        self.dataset_name = dataset_name
        self.model_config = model_config
        self.num_experts_intermediate = num_experts_intermediate
        self.num_teams_advanced = num_teams_advanced
        self.num_agents_per_team_advanced = num_agents_per_team_advanced
        self.config_path = config_path
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.moderator_agent = Agent(agent_id="moderator",
                                        agent_type=AgentType.MODERATOR, 
                                        model_key=model_config.get('moderator', DEFAULT_MODERATOR_MODEL),
                                        config_path=config_path,
                                        instruction="You are a medical expert who conducts initial assessment. Your job is to decide the difficulty/complexity of the medical query based on the provided definitions. Respond in JSON format."
        )
        self.recruiter_agent = Agent(agent_id="recruiter",
                                        agent_type=AgentType.RECRUITER, 
                                        model_key=model_config.get('recruiter', DEFAULT_RECRUITER_MODEL),
                                        config_path=config_path,
                                        instruction="You are an experienced medical expert who recruits appropriate specialists based on the medical query and its complexity level. Respond in JSON format."
        )
        self.decision_maker_agent = Agent(agent_id="final_decision_maker",
                                            agent_type=AgentType.DECISION_MAKER,
                                            model_key=model_config.get('moderator', DEFAULT_MODERATOR_MODEL),
                                            config_path=config_path,
                                            instruction="You are a final medical decision maker. Review all provided information (opinions, reports, discussions) and make the final, consolidated answer to the original medical query. Respond in JSON format."
        )
        
        self.auditor_agent = AuditorAgent(agent_id="auditor", model_key=auditor_model_key,config_path = config_path, agent_type = AgentType.AUDITOR)
        
    def _determine_complexity(self, question: str, options: Optional[Dict] = None, image_path: Optional[str] = None) -> Tuple[ComplexityLevel, Dict[str, Any]]:
        print("\n--- Determining Complexity ---")
        self.moderator_agent.clear_memory()
        query_context = f"Medical Query:\n{question}\n"
        if options:
            options_str = ""
            for k, v in options.items():
                options_str += f"{k}: {v}\n"
            query_context += f"Options:\n{options_str}\n"
        if image_path:
            query_context += "This query includes a medical image.\n"

        prompt = (
            f"Given the medical query below, decide its difficulty/complexity.\n\n"
            f"{query_context}\n"
            f"Complexity Guidelines:\n"
            f"1) basic: A single medical agent (like a PCP or general physician) can likely answer this knowledge question or simple case directly.\n"
            f"2) intermediate: Requires discussion among a team of medical experts with different specialties to reach a consensus.\n"
            f"3) advanced: A complex case requiring multiple teams (e.g., initial assessment, diagnostics, final review) collaborating across departments.\n\n"
            f"Respond with a JSON object containing a 'complexity' field with one of these values: 'basic', 'intermediate', or 'advanced'."
        )        
        response, _, system_message, user_message = self.moderator_agent.chat(
            prompt=prompt,
            image_path=None,
            response_format={"type": "json_object"},
        )
        step_log = {"step_name": "determine_complexity", 
                    "prompt": prompt, 
                    "llm_log": {"llm_input":{
                        "system_message": system_message,
                        "user_message": user_message
                    }}}
        try:
            response_clean = preprocess_response_string(response)
            response_json = json.loads(response_clean)
            complexity_str = response_json.get("complexity", "").lower()
            
            step_log["parsed_response"] = response_json

            if complexity_str == "basic":
                print("Complexity: BASIC")
                return ComplexityLevel.BASIC, step_log
            elif complexity_str == "intermediate":
                print("Complexity: INTERMEDIATE")
                return ComplexityLevel.INTERMEDIATE, step_log
            elif complexity_str == "advanced":
                print("Complexity: ADVANCED")
                return ComplexityLevel.ADVANCED, step_log
            else:
                print(f"Warning: Invalid complexity value '{complexity_str}'. Defaulting to INTERMEDIATE.")
                step_log["warning"] = f"Invalid complexity value '{complexity_str}', defaulted to INTERMEDIATE."
                return ComplexityLevel.INTERMEDIATE, step_log
 
        except Exception as e:
            print(f"Error parsing complexity response: {e}. Raw response: {response}. Defaulting to INTERMEDIATE.")
            step_log["error"] = f"Error parsing response: {e}. Raw response: {response}"
            step_log["warning"] = "Defaulted to INTERMEDIATE due to parsing error."
            return ComplexityLevel.INTERMEDIATE, step_log

    def _recruit_experts(self,
                         question: str,
                         options: Optional[Dict], 
                         complexity: ComplexityLevel, 
                         image_path: Optional[str] = None) -> Tuple[Union[List, Dict], Dict[str, Any]]:
        print("\n--- Recruiting Experts/Teams ---")
        self.recruiter_agent.clear_memory()

        step_log = {"step_name": "recruit_experts", "complexity": complexity.value}
        if complexity == ComplexityLevel.BASIC:
            print("Basic complexity - no recruitment needed.")
            step_log["message"] = "No recruitment needed for BASIC complexity."
            return [], step_log
        query_context = f"Medical Query:\n{question}\n"
        if options:
            options_str = ""
            for k, v in options.items():
                options_str += f"{k}: {v}\n"
            query_context += f"Options:\n{options_str}\n"
        if image_path:
            query_context += "This query includes a medical image.\n"
        prompt = ""
        if complexity == ComplexityLevel.INTERMEDIATE:
            recruitment_instruction = (
                f"You are an experienced medical expert. Your task is to recruit a team of {self.num_experts_intermediate} experts "
                f"with diverse specialties and expertise to discuss and solve the given medical query. "
                f"Specify their specialty, a brief expertise description, and optionally a communication hierarchy (e.g., 'Cardiologist > Nurse', 'Independent'). "
                f"Respond in JSON format."
            )
            self.recruiter_agent.instruction = recruitment_instruction
            prompt = (
                f"{query_context}\n"
                f"Recruit {self.num_experts_intermediate} experts for this moderately complex query.\n"
                f"Respond with a JSON array 'experts' where each expert object has fields: 'specialty', 'expertise', and 'hierarchy'.\n"
                f"Example response structure:\n"
                f"{{\"experts\": [\n"
                f"  {{\"specialty\": \"Pediatrician\", \"expertise\": \"Specializes in child healthcare\", \"hierarchy\": \"Independent\"}},\n"
                f"  {{\"specialty\": \"Cardiologist\", \"expertise\": \"Focuses on heart conditions\", \"hierarchy\": \"Pediatrician > Cardiologist\"}},\n"
                f"  {{\"specialty\": \"Pulmonologist\", \"expertise\": \"Specializes in respiratory disorders\", \"hierarchy\": \"Independent\"}}\n"
                f"]}}"
            )
        elif complexity == ComplexityLevel.ADVANCED:
            recruitment_instruction = (
                f"You are an experienced medical director. Your task is to organize {self.num_teams_advanced} Multidisciplinary Teams (MDTs) "
                f"for a complex medical query. Each MDT should have around {self.num_agents_per_team_advanced} clinicians. Define the purpose (goal) of each team "
                f"and list its members with their specialties and expertise. Ensure you include an 'Initial Assessment Team (IAT)' and a 'Final Review and Decision Team (FRDT)'. "
                f"Respond in JSON format."
            )
            self.recruiter_agent.instruction = recruitment_instruction
            prompt = (
                f"{query_context}\n"
                f"Organize {self.num_teams_advanced} MDTs, each with ~{self.num_agents_per_team_advanced} members, for this complex query.\n"
                f"Include an IAT and an FRDT.\n"
                f"Respond with a JSON object containing a 'teams' array where each team has: 'group_id', 'goal', and 'members' array.\n"
                f"Each member should have 'specialty', 'expertise', and optionally 'is_lead' (boolean) fields.\n"
                f"Example structure:\n"
                f"{{\"teams\": [\n"
                f"  {{\"group_id\": \"Group 1\", \"goal\": \"Initial Assessment Team (IAT)\", \"members\": [\n"
                f"    {{\"specialty\": \"Emergency Physician\", \"expertise\": \"Handles acute assessment\", \"is_lead\": true}},\n"
                f"    {{\"specialty\": \"Triage Nurse\", \"expertise\": \"Gathers initial patient data\"}}\n"
                f"  ]}},\n"
                f"  {{\"group_id\": \"Group 2\", \"goal\": \"Final Review and Decision Team (FRDT)\", \"members\": [\n"
                f"    {{\"specialty\": \"Senior Consultant\", \"expertise\": \"Oversees final decision\", \"is_lead\": true}},\n"
                f"    {{\"specialty\": \"Clinical Pharmacist\", \"expertise\": \"Medication review\"}}\n"
                f"  ]}}\n"
                f"]}}"
            )

        recruitment_response, reasoning_content, system_message, user_message = self.recruiter_agent.chat(
            prompt=prompt,
            image_path=None,
            response_format={"type": "json_object"},
        )
        print(f"Recruiter Response ({complexity.value}):\n{recruitment_response}")

        llm_log = {"llm_input":{
            "system_message": system_message,
            "user_message": user_message
        }}

        step_log["prompt"] = prompt
        step_log["llm_log"] = llm_log

        if complexity == ComplexityLevel.INTERMEDIATE:
            try:
                response_clean = preprocess_response_string(recruitment_response)
                response_json = json.loads(response_clean)
                experts = response_json.get("experts", [])
                step_log["parsed_response"] = response_json

                validated_experts = []
                for expert in experts:
                    if isinstance(expert, dict) and 'specialty' in expert:
                        validated_expert = {
                            "specialty": expert.get("specialty", "Unknown Specialty"),
                            "expertise": expert.get("expertise", "General expertise related to the specialty."),
                            "hierarchy": expert.get("hierarchy", "Independent")
                        }
                        validated_experts.append(validated_expert)
                experts = validated_experts

            except Exception as e:
                print(f"Error parsing expert recruitment response: {e}. Raw response: {recruitment_response}")
                print("Warning: Failed to parse experts. Using default specialties.")
                step_log["error"] = f"Error parsing response: {e}"
                step_log["warning"] = "Using default specialties due to parsing error."
                default_specialties = ["Internal Medicine Specialist", "Radiologist", "Surgeon", "Pathologist", "Pharmacist"]
                experts = [{"specialty": r, "expertise": f"Expertise in {r}", "hierarchy": "Independent"} for r in default_specialties[:self.num_experts_intermediate]]

            specialties = [e['specialty'] for e in experts]
            print(f"Recruited Experts: {[e['specialty'] for e in experts]}")
            step_log["specialties"] = specialties
            step_log["recruited_personnel"] = experts
            return experts, step_log

        elif complexity == ComplexityLevel.ADVANCED:
            try:
                response_clean = preprocess_response_string(recruitment_response)
                response_json = json.loads(response_clean)
                teams = response_json.get("teams", [])

                step_log["parsed_response"] = response_json

                validated_teams = []
                for team in teams:
                    if isinstance(team, dict) and 'group_id' in team and 'members' in team:
                        validated_members = []
                        for member in team.get('members', []):
                            if isinstance(member, dict) and 'specialty' in member:
                                validated_member = {
                                    "specialty": member.get("specialty", "Unknown Specialty"),
                                    "expertise": member.get("expertise", "General expertise for the specialty.")
                                }
                                if member.get("is_lead") is True:
                                    validated_member["is_lead"] = True
                                validated_members.append(validated_member)

                        validated_team = {
                            "group_id": team.get("group_id", f"Group {len(validated_teams)+1}"),
                            "goal": team.get("goal", f"Goal for Group {len(validated_teams)+1}"),
                            "members": validated_members
                        }
                        validated_teams.append(validated_team)
                teams = validated_teams
                
            except Exception as e:
                print(f"Error parsing team recruitment response: {e}. Raw response: {recruitment_response}")
                print("Warning: Failed to parse any teams. Using default structure.")
                step_log["error"] = f"Error parsing response: {e}"
                step_log["warning"] = "Using default team structure due to parsing error."
                teams = [
                    {"group_id": "Group 1", "goal": "Initial Assessment Team (IAT)", "members": [
                        {"specialty": "Emergency Physician", "expertise": "Acute assessment", "is_lead": True},
                        {"specialty": "Radiologist", "expertise": "Initial imaging"}
                    ]},
                    {"group_id": "Group 2", "goal": "Diagnostic Team", "members": [
                        {"specialty": "Cardiologist", "expertise": "Heart conditions", "is_lead": True},
                        {"specialty": "Neurologist", "expertise": "Nervous system"}
                    ]},
                    {"group_id": "Group 3", "goal": "Final Review and Decision Team (FRDT)", "members": [
                        {"specialty": "Senior Consultant", "expertise": "Oversees decision", "is_lead": True},
                        {"specialty": "Clinical Pharmacist", "expertise": "Medication review"}
                    ]}
                ]
                teams = teams[:self.num_teams_advanced]
            specialties = []
            for team in teams:
                for member in team['members']:
                    specialties.append(member['specialty'])
            print(f"Recruited Teams: {[t['goal'] for t in teams]}")
            step_log["recruited_personnel"] = teams
            step_log["specialties"] = [s.value for s in specialties]
            return teams, step_log
        return [], step_log

    def _process_basic_query(self, data_item: Dict) -> Dict:
        print("\n--- Processing Basic Query ---")
        case_history = {"rounds": []}
        round_data = {"round": 1, "opinions": [], "decision": None}
        case_history["rounds"].append(round_data)
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)
        agent = Agent(
            agent_id="basic_solver",
            specialty=MedicalSpecialty.GENERAL_MEDICINE,
            config_path=self.config_path,
            model_key=agent_model_key,
            instruction="You are a helpful medical assistant. Answer the following medical question accurately. Respond in JSON format."
        )                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        main_prompt = f"Question: {data_item['question']}\n"
        options = data_item.get('options')
        if options:
            options_str = "Options:\n"
            for k, v in options.items():
                options_str += f"({k}) {v}\n"
            main_prompt += options_str
            main_prompt += "\nProvide your answer as a JSON object with 'answer' (letter for multiple-choice), 'explanation', "
        else:
            main_prompt += "\nProvide your answer as a JSON object with 'answer', 'explanation', "

        response, reasoning_content, _, _ = agent.chat(prompt=main_prompt, 
                                       image_path=data_item.get('image_path'), 
                                       response_format={"type": "json_object"})
    

        try:
            result = json.loads(preprocess_response_string(response))
            predicted_answer = result.get("answer", "parse_error")
            explanation = result.get("explanation", "No explanation provided.")
            if options and isinstance(predicted_answer, str):
                if predicted_answer.startswith('(') and predicted_answer.endswith(')'):
                    predicted_answer = predicted_answer[1:-1].strip()
                elif predicted_answer.startswith('(') and len(predicted_answer) > 2 and predicted_answer[1].isalpha() and predicted_answer[2] == ')':
                    predicted_answer = predicted_answer[1]
                elif len(predicted_answer) > 1 and predicted_answer[0].isalpha() and (predicted_answer[1] == '.' or predicted_answer[1] == ')'):
                    predicted_answer = predicted_answer[0]

            
            case_history["rounds"][-1]["decision"] = {"parsed_output": result, "reasoning_content": reasoning_content}
        except Exception as e:
            print(f"Error parsing basic response: {e}. Raw response: {response}")
            predicted_answer = "Could not parse answer."
            explanation = "Error parsing model response."
        print(f"Basic Query Result: Answer='{predicted_answer}', Explanation='{explanation[:100]}...'")

        
        
        return {
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "complexity": ComplexityLevel.BASIC.value,
            "case_history": case_history
        }


    def _process_intermediate_query(self, data_item: Dict, expert_configs: List[Dict], audit_results_of_role_assignment: Dict, specialties: List, task: str) -> Dict:
        case_history = {"rounds": []}
        round_data = {"round": 1, "opinions": [], "synthesis": None, "reviews": [], "decision": None}
        case_history["rounds"].append(round_data)
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
            audit_round_data["2_1_1_role_assignment"].append({
                "specialties": specialties,
                "step": "role_assignment",
                "audit_result": audit_results_of_role_assignment
            })

        print("\n--- Processing Intermediate Query ---")
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)
        agents = []
        for i, config in enumerate(expert_configs):
            agent_id = f"expert_{i+1}_{config['specialty'].replace(' ','_').lower()}"
            agent = Agent(
                agent_id=agent_id,
                specialty=config['specialty'],
                config_path=self.config_path,
                model_key=agent_model_key,
                instruction=f"You are a {config['specialty']} with expertise in {config['expertise']}. Collaborate with other medical experts to answer the medical query. Maintain your persona and provide insights based on your specialty. Respond in JSON format."
            )
            agents.append(agent)
        if not agents:
            print("Error: No expert agents created for intermediate query. Aborting.")
            return {"predicted_answer": "Error", "explanation": "Failed to create expert agents."}

        print(f"Created {len(agents)} expert agents: {[a.agent_id for a in agents]}")
        
        question = data_item['question']
        options = data_item.get('options')
        image_path = data_item.get('image_path')

        question_context = f"Question: {question}\n"
        if options:
            options_str = "Options:\n"
            for k, v in options.items():
                options_str += f"({k}) {v}\n"
            question_context += options_str
        if image_path:
            question_context += "(Image provided separately)\n"

        print("\n-- Round 1: Initial Opinions --")

        initial_report_parts = []
        initial_opinions_parsed = []
        for agent in agents:
            agent.clear_memory()
            prompt = (
                f"{question_context}\n"
                f"Based on your expertise as a {agent.specialty}, provide your initial analysis and answer.\n"
                f"Respond with a JSON object containing 'answer' and 'explanation' fields."
            )
            response, reasoning_content, system_message, user_message = agent.chat(
                prompt=prompt,
                image_path=image_path,
                response_format={"type": "json_object"},
            )

            opinion_log = {
                "reasoning_content": reasoning_content,
                "llm_input": {
                    "system_message": system_message,
                    "user_message": user_message
                }
            }

            try:
                response_clean = preprocess_response_string(response)
                response_json = json.loads(response_clean)
                ans = response_json.get("answer", "")
                expl = response_json.get("explanation", "No explanation provided.")
                opinion_log["parsed_output"] = response_json
                
                if options and isinstance(ans, str):
                    if ans.startswith('(') and ans.endswith(')'):
                        ans = ans[1:-1].strip()
                    elif ans.startswith('(') and len(ans) > 2 and ans[1].isalpha() and ans[2] == ')':
                        ans = ans[1]
                    elif len(ans) > 1 and ans[0].isalpha() and (ans[1] == '.' or ans[1] == ')'):
                        ans = ans[0]
                if task == "audit":
                    # audit 1.1.1 facutal hallucination
                    audit_results_of_factual_hallucination = self.auditor_agent.audit_factual_hallucination(question = question, image_path=image_path, agent_id=agent.agent_id, specialty=agent.specialty, answer=ans, explanation=expl)

                    # audit 1.2.1 neglect or misinterpretation of modality information
                    audit_results_of_neglect_or_misinterpretation_of_modality_info = self.auditor_agent.audit_neglect_or_misinterpretation_of_modality_info(question = question, image_path=image_path, agent_id=agent.agent_id, specialty=agent.specialty, answer=ans, explanation=expl)

                    # audit 2.1.2 domain-specific knowledge activation
                    audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question = question, image_path=image_path, agent_id=agent.agent_id, specialty=agent.specialty, answer=ans, explanation=expl)
                    audit_round_data["1_1_1_factual_hallucination"].append({
                        "agent_id": agent.agent_id,
                        "specialty": agent.specialty,
                        "step": "analysis",
                        "audit_result": audit_results_of_factual_hallucination
                    })

                    audit_round_data["1_2_1_neglect_or_misinterpretation_of_modality_info"].append({
                        "agent_id": agent.agent_id,
                        "specialty": agent.specialty,
                        "step": "analysis",
                        "audit_result": audit_results_of_neglect_or_misinterpretation_of_modality_info
                    })
                    audit_round_data["2_1_2_domain_specific_knowledge_activation"].append({
                        "agent_id": agent.agent_id,
                        "specialty": agent.specialty,
                        "step": "analysis",
                        "audit_result": audit_results_of_domain_specific_knowledge_activation
                    })

                case_history["rounds"][-1]["opinions"].append({
                    "agent_id": agent.agent_id,
                    "specialty": agent.specialty,
                    "log": opinion_log
                })

            except Exception as e:
                print(f"Error parsing initial opinion from {agent.agent_id}: {e}. Raw response: {response}")
                ans = "Could not parse answer."
                expl = "Error parsing model response."
                opinion_log["error"] = f"Error parsing response: {e}"
            
            parsed_opinion = {
                "agent_id": agent.agent_id,
                "agent_specialty": agent.specialty,
                "answer": ans,
                "explanation": expl,
            }

            initial_opinions_parsed.append(parsed_opinion)
            
            initial_report_parts.append(f"Expert {agent.specialty} ({agent.agent_id}):\nAnswer: {ans}\nExplanation: {expl}\n---")
            print(f"Agent {agent.agent_id} ({agent.specialty}) Initial Answer: {ans}")

        print("\n-- Synthesizing Final Decision --")
        self.decision_maker_agent.clear_memory()

        synthesis_prompt = (
            f"You need to make a final decision for the following medical query based on initial opinions from a team of experts:\n\n"
            f"{question_context}\n\n"
            f"--- Expert Opinions (Round 1) ---\n"
            f"{''.join(initial_report_parts)}\n"
            f"--- End Opinions ---\n\n"
            f"Review these opinions carefully. Consider the different expert perspectives and their specific expertise.\n"
            f"Respond with a JSON object containing 'answer' (letter for multiple-choice) and 'explanation' fields."
        )
        
        final_response, reasoning_content, final_system_message, final_user_message = self.decision_maker_agent.chat(
            prompt=synthesis_prompt,
            response_format={"type": "json_object"},
            image_path=image_path
        )
        decision_log = {
            "reasoning_content": reasoning_content,
            "llm_input": {"system_message": final_system_message, "user_message": final_user_message}
        }

        try:
            response_clean = preprocess_response_string(final_response)
            response_json = json.loads(response_clean)
            final_answer = response_json.get("answer", "")
            final_explanation = response_json.get("explanation", "No explanation provided.")
            decision_log["parsed_output"] = response_json

            if options and isinstance(final_answer, str):
                if final_answer.startswith('(') and final_answer.endswith(')'):
                    final_answer = final_answer[1:-1].strip()
                elif final_answer.startswith('(') and len(final_answer) > 2 and final_answer[1].isalpha() and final_answer[2] == ')':
                    final_answer = final_answer[1]
                elif len(final_answer) > 1 and final_answer[0].isalpha() and (final_answer[1] == '.' or final_answer[1] == ')'):
                    final_answer = final_answer[0]
            if task == "audit":
                # audit 3.1.1 : Suppression of Correct Minority Views by Incorrect Consensus for decision-maker
                audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker = self.auditor_agent.audit_suppression_by_majority(
                    question = question, options = options, image_path = image_path, current_agent_id = self.decision_maker_agent.agent_id, answer = final_answer, explanation = final_explanation, case_history = case_history
                )

                # audit 3.1.2 : Reasoning Distorted by Authority Bias for synthesizer
                audit_results_of_authority_bias_for_decision_maker = self.auditor_agent.audit_authority_bias(
                    question = question, options = options, image_path = image_path, current_agent_id = self.decision_maker_agent.agent_id, answer = final_answer, explanation = final_explanation, case_history = case_history
                ) # here the discussion_context must include the role of domain agent and their answer and explanation before this synthesis

                # audit 3.1.3: Neglect of Contradictions in Reasoning Process for synthesizer
                audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker = self.auditor_agent.audit_contradictions_during_decision(
                    question = question, current_agent_id = self.decision_maker_agent.agent_id, explanation = final_explanation, case_history = case_history, options = options
                ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis
                
                audit_round_data["3_1_1_suppression_of_minority_views"].append({
                    "agent_id": self.decision_maker_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker
                })
                audit_round_data["3_1_2_authority_bias"].append({
                    "agent_id": self.decision_maker_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_authority_bias_for_decision_maker
                })
                audit_round_data["3_1_3_neglect_of_contradictions"].append({
                    "agent_id": self.decision_maker_agent.agent_id,
                    "step": "decision",
                    "audit_result": audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker
                })
                audit["rounds"].append(audit_round_data)
            case_history["rounds"][-1]["decision"] = decision_log # after synthesizer then log, in case repetition
            if task == "audit":
                case_history["audit"] = audit

        except Exception as e:
            print(f"Error parsing final decision: {e}. Raw response: {final_response}")
            final_answer = "Could not parse answer."
            final_explanation = "Error parsing model response."
        print(f"Intermediate Query Final Result: Answer='{final_answer}', Explanation='{final_explanation[:100]}...'")

        return {
            "predicted_answer": final_answer,
            "explanation": final_explanation,
            "complexity": ComplexityLevel.INTERMEDIATE.value,
            "case_history": case_history
        }
    
    def _process_advanced_query(self, data_item: Dict, team_configs: List[Dict], audit_results_of_role_assignment: Dict, specialties: List, task: str) -> Dict:
        print("\n--- Processing Advanced Query ---")
        options = data_item.get('options')
        case_history = {"rounds": []}
        round_data = {"round": 1, "opinions": [], "synthesis": [], "reviews": [], "decision": None}
        case_history["rounds"].append(round_data)
        audit_round_data = {}
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

            audit_round_data["2_1_1_role_assignment"].append({
                "specialties": specialties,
                "step": "role_assignment",
                "audit_result": audit_results_of_role_assignment
            })

        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)
        question_context = {
            "question": data_item["question"],
            "options": data_item.get("options"),
            "image_path": data_item.get("image_path"),
        }
        groups = []

        for i, config in enumerate(team_configs):
            members = []
            for j, member_config in enumerate(config['members']):
                agent_id = f"{config['group_id'].replace(' ','_').lower()}_member_{j+1}_{member_config['specialty'].replace(' ','_').lower()}"
                is_lead = member_config.get('is_lead', False)
                instruction_prefix = f"You are {member_config['specialty']} ({member_config['expertise']}) in team '{config['goal']}'."
                if is_lead:
                    instruction_prefix += " You are the LEAD of this team."
                agent = Agent(
                    agent_id=agent_id,
                    specialty=f"{member_config['specialty']}{' (Lead)' if is_lead else ''}",
                    model_key=agent_model_key,
                    instruction=f"{instruction_prefix} Collaborate within your team to achieve the goal: '{config['goal']}'. Respond in JSON format."
                )
                members.append(agent)

            group = Group(
                group_id=config['group_id'],
                goal=config['goal'],
                members=members,
                question_context=question_context
            )
            groups.append(group)

        print(f"Created {len(groups)} teams: {[g.group_id for g in groups]}")

        team_reports = {}
        all_reports_for_synthesis = ""
        ordered_groups = sorted(groups, key=lambda g: 0 if "initial" in g.goal.lower() else (2 if "final" in g.goal.lower() else 1))


        for group in ordered_groups:
            print(f"\n-- Processing Team: {group.group_id} ({group.goal}) --")
            raw_report = group.perform_internal_discussion( # not return audit_round_data and case_history
                auditor_agent=self.auditor_agent,
                audit_round_data = audit_round_data,
                case_history = case_history, 
                task = task
            )
            
            try:
                response_clean = preprocess_response_string(raw_report)
                report_json = json.loads(response_clean)
                team_reports[group.group_id] = report_json
                report_for_synthesis = f"--- Report from {group.group_id} ({group.goal}) ---\n"
                report_for_synthesis += f"Answer: {report_json.get('answer', 'N/A')}\n"
                report_for_synthesis += f"Explanation: {report_json.get('explanation', 'No explanation provided.')}\n---\n"
            except Exception as e:
                print(f"Error parsing team report: {e}. Raw report: {raw_report}")
                team_reports[group.group_id] = {"raw_report": raw_report, "error": str(e)}
                report_for_synthesis = f"--- Report from {group.group_id} ({group.goal}) ---\n{raw_report}\n---\n"
            
            all_reports_for_synthesis += report_for_synthesis

        print("\n-- Synthesizing Final Decision from Team Reports --")
        self.decision_maker_agent.clear_memory()
        if not all_reports_for_synthesis:
            all_reports_for_synthesis = "No team reports were generated."
        options_str = ""
        if data_item.get('options'):
            options_str += "Options:\n"
            for k,v in data_item['options'].items():
                options_str += f"({k}) {v}\n"

        synthesis_prompt = (
            f"You need to make the ultimate final decision for the following complex medical query based on reports from multiple specialized teams:\n\n"
            f"Original Query Context:\nQuestion: {data_item['question']}\n{options_str}"
            f"{'(Image associated)' if data_item.get('image_path') else ''}\n\n"
            f"--- Compiled Team Reports ---\n"
            f"{all_reports_for_synthesis}\n"
            f"--- End Reports ---\n\n"
            f"Your task is to synthesize all available information into a single, definitive analysis.\n\n"
            f"Synthesize all this information into one final, definitive answer and explanation.\n"
            f"Respond with a JSON object containing 'answer' (letter for multiple-choice) and 'explanation' fields."
        )
        final_response, reasoning_content, final_system_message, final_user_message = self.decision_maker_agent.chat(
            prompt=synthesis_prompt,
            response_format={"type": "json_object"},
            image_path=data_item.get("image_path")
        )
        
        try:
            response_clean = preprocess_response_string(final_response)
            response_json = json.loads(response_clean)
            final_answer = response_json.get("answer", "")
            final_explanation = response_json.get("explanation", "No explanation provided.")

            if options and isinstance(final_answer, str):
                if final_answer.startswith('(') and final_answer.endswith(')'):
                    final_answer = final_answer[1:-1].strip()
                elif final_answer.startswith('(') and len(final_answer) > 2 and final_answer[1].isalpha() and final_answer[2] == ')':
                    final_answer = final_answer[1]
                elif len(final_answer) > 1 and final_answer[0].isalpha() and (final_answer[1] == '.' or final_answer[1] == ')'):
                    final_answer = final_answer[0]
        except Exception as e:
            print(f"Error parsing final decision (advanced): {e}. Raw response: {final_response}")
            final_answer = "Could not parse answer."
            final_explanation = "Error parsing model response."
        print(f"Advanced Query Final Result: Answer='{final_answer}', Explanation='{final_explanation[:100]}...'")
        if task == "audit":
            # audtit 3.1.1 : Suppression of Correct Minority Views by Incorrect Consensus for decision-maker
            audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker = self.auditor_agent.audit_suppression_by_majority(
                question = data_item["question"], options = options, image_path = data_item.get("image_path"), current_agent_id = self.decision_maker_agent.agent_id, answer = final_answer, explanation = final_explanation, case_history = case_history
            ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis

            # audit 3.1.2 : Reasoning Distorted by Authority Bias for decision-maker
            audit_results_of_authority_bias_for_decision_maker = self.auditor_agent.audit_authority_bias(
                question = data_item["question"], options = options, image_path = data_item.get("image_path"), current_agent_id = self.decision_maker_agent.agent_id, answer = final_answer, explanation = final_explanation, case_history = case_history
            ) # here the discussion_context must include the role of domain agent and their answer and explanation before this synthesis

            # audit 3.1.3: Neglect of Contradictions in Reasoning Process for decision-maker
            audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker = self.auditor_agent.audit_contradictions_during_decision(
                question = data_item["question"], current_agent_id = self.decision_maker_agent.agent_id, explanation = final_explanation, case_history = case_history, options = options
            ) # here the discussion_context includes all the domain agents' answers and explanations before this synthesis

            audit_round_data["3_1_1_suppression_of_minority_views"].append({
                "agent_id": self.decision_maker_agent.agent_id,
                "step": "decision",
                "audit_result": audit_results_of_suppression_of_correct_minority_views_by_incorrect_consensus_for_decision_maker
            })
            audit_round_data["3_1_2_authority_bias"].append({
                "agent_id": self.decision_maker_agent.agent_id,
                "step": "decision",
                "audit_result": audit_results_of_authority_bias_for_decision_maker
            })
            audit_round_data["3_1_3_neglect_of_contradictions"].append({
                "agent_id": self.decision_maker_agent.agent_id,
                "step": "decision",
                "audit_result": audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker
            })

            audit["rounds"].append(audit_round_data)
        case_history['rounds'][-1]['decision'] = {'parsed_output': response_json, "reasoning_content": reasoning_content}
        if task == "audit":
            case_history["audit"] = audit
        return {
            "predicted_answer": final_answer,
            "explanation": final_explanation,
            "complexity": ComplexityLevel.ADVANCED.value,
            "case_history": case_history
        }
    def run_query(self, data_item: Dict, task: str) -> Dict:
        qid = data_item["qid"]
        print(f"\n{'='*20} Processing QID: {qid} {'='*20}")
        start_time = time.time()
        process_log = []
        result_data = {}

        complexity, complexity_log = self._determine_complexity(data_item["question"], data_item.get("options"), data_item.get("image_path"))
        process_log.append(complexity_log)

        if complexity == ComplexityLevel.BASIC:
            result_data = self._process_basic_query(data_item=data_item)
        else:
            recruited, recruitment_log = self._recruit_experts(data_item["question"], data_item.get("options"), complexity, data_item.get("image_path"))
            process_log.append(recruitment_log)
            specialties = recruitment_log.get("specialties", [])
            audit_results_of_role_assignment = {}
            if task == "audit":
                # audit 2.1.1 role assignment
                audit_results_of_role_assignment = self.auditor_agent.audit_role_assignment(question=data_item["question"], image_path=data_item.get("image_path"), specialties=specialties)
            if complexity == ComplexityLevel.INTERMEDIATE:
                result_data = self._process_intermediate_query(data_item, recruited, audit_results_of_role_assignment, specialties, task)
            elif complexity == ComplexityLevel.ADVANCED:
                result_data = self._process_advanced_query(data_item, recruited, audit_results_of_role_assignment, specialties, task)

        processing_time = time.time() - start_time
        

        return {
            "qid": qid, "question": data_item["question"], "options": data_item.get("options"),
            "ground_truth": data_item.get("answer"), "complexity_level": result_data.get("complexity"),
            "predicted_answer": result_data.get("predicted_answer"), "explanation": result_data.get("explanation"),
            "processing_time_seconds": processing_time,
            "process_log": process_log, 
            "case_history": result_data.get("case_history", {}),
        }

    def run_dataset(self, data: List[Dict], task: str):
        for item in tqdm(data, desc=f"Running MDAgents on {self.dataset_name}"):
            qid = item.get("qid", "unknown_qid")

            existing_qids = set()
            if self.log_file.exists():
                print(f"Output file {self.log_file} already exists. Appending new results.")
                with open(self.log_file, 'r', encoding='utf-8') as f:
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
                result = self.run_query(data_item=item, task = task)
                save_jsonl(result, self.log_file)
            except Exception as e:
                print(f"FATAL ERROR for QID {qid}: {e}")
                save_jsonl({"qid": qid, "error": str(e)}, self.err_log_file)

def main():
    parser = argparse.ArgumentParser(description="Run MDAgents Framework on medical datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--moderator_model", type=str, required=True,help = "deepseek-reasoner/gpt-5.1/gemini-2.5-flash")
    parser.add_argument("--recruiter_model", type=str, required=True, help = "deepseek-reasoner/gpt-5.1/gemini-2.5-flash")
    parser.add_argument("--agent_model", type=str, required=True, help = "qa = deepseek-reasoner/gpt-5.1/gemini-2.5-flash ,vqa = qwen-3-vl/gpt-5.1/gemini-2.5-flash") # this agent need to read the figure if we execute the vqa task
    parser.add_argument("--auditor_model", type=str, required=True, help="gemini-3-pro-preview")
    parser.add_argument("--config_path", type=str, required=True,help="Path to the config.toml file,default = utils/config.toml")
    parser.add_argument("--num_samples",type = int, required = True, help = "number of samples to run")
    parser.add_argument("--num_experts", type=int, default=DEFAULT_NUM_EXPERTS_INTERMEDIATE)
    parser.add_argument("--num_teams", type=int, default=DEFAULT_NUM_TEAMS_ADVANCED)
    parser.add_argument("--time_stamp", type=str, required=True, help="Timestamp for logging purposes")
    parser.add_argument("--task", type = str, required=True, help="audit or open_coding?")
    
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")

    timestamp = args.time_stamp
    current_mas_name = current_file_name
    main_llm = args.agent_model
    task = args.task
    print(f"Task: {task}")

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

    model_config = {
        "moderator": args.moderator_model,
        "recruiter": args.recruiter_model,
        "default_agent": args.agent_model
    }

    framework = MDAgentsFramework(
        log_file=output_file, 
        err_log_file = error_output_file,
        dataset_name=args.dataset, 
        model_config=model_config,
        auditor_model_key=args.auditor_model,
        num_experts_intermediate=args.num_experts, 
        num_teams_advanced=args.num_teams,
        config_path = args.config_path
    )

    framework.run_dataset(data=data[:args.num_samples], task = task)

if __name__ == "__main__":
    main()