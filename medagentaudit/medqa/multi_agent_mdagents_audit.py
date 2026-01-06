# multi_agent_mdagents_full_log.py

import os
import time
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
project_root = current_file_path.parents[2]
sys.path.extend([str(utils_root), str(project_root)])
from config import get_config
from dual_logger import DualLogger
from encode_image import encode_image
from json_utils import load_json, save_json, preprocess_response_string
from auditor_agent import AuditorAgent
from BaseAgent import BaseAgent
from agent_type import AgentType
from medical_specialty import MedicalSpecialty
from parse_structured_output import parse_structured_output

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
    def __init__(self, agent_id: str, role: Union[AgentType, str], model_key: str, config_path:str, instruction: Optional[str] = None):
        super().__init__(agent_id=agent_id, agent_type=role, config_path=config_path, model_key=model_key)
        self.agent_id = agent_id
        self.role = role if isinstance(role, str) else role.value
        self.instruction = instruction or f"You are a helpful assistant playing the role of a {self.role}."

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
        response, system_message, user_message = self.call_llm(messages, response_format=response_format)
        return response, system_message, user_message

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
        self.lead_agent = next((m for m in members if 'lead' in m.role.lower()), members[0] if members else None)
        if self.lead_agent:
            print(f"Group {self.group_id} Lead: {self.lead_agent.agent_id}")

    def _log_interaction(self, message: str, data: Optional[Dict] = None):
        log_entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message, "data": data or {}}
        # print(f"[Group {self.group_id} Log] {message}")
        self.internal_log.append(log_entry)

    def perform_internal_discussion(
        self,
        auditor_agent: AuditorAgent,
        audit_trail: Dict[str, Any],
        keu_counter: int,
        ccp_counter: int
    ) -> Tuple[str, List[Dict], int, int]:
        """
        [MODIFIED] Simulates internal discussion with integrated auditing mechanisms.
        """
        if not self.lead_agent:
            return "Error: Group has no lead agent.", self.internal_log, keu_counter, ccp_counter

        self._log_interaction(f"Starting internal discussion. Lead: {self.lead_agent.agent_id}")
        assist_members = [m for m in self.members if m != self.lead_agent]
        # 1. Lead asks assistants for investigations
        delivery_prompt = f"You are the lead of the medical group: '{self.group_id}', which aims to '{self.goal}'.\n"

        if assist_members:
            delivery_prompt += "Your assistant clinicians are:\n"
            for a_mem in assist_members:
                delivery_prompt += f"- {a_mem.role} (ID: {a_mem.agent_id})\n"
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

        lead_request, system_message, user_message = self.lead_agent.chat(
            prompt=delivery_prompt,
            image_path=self.question_context.get('image_path'),
        )
 
        self._log_interaction(
            f"Lead ({self.lead_agent.agent_id}) generated task assignments for assistants.",
            data={
                "step": "1_lead_request_generation",
                "agent_id": self.lead_agent.agent_id,
                "agent_role": self.lead_agent.role,
                "prompt": delivery_prompt,
                "response": lead_request,
                "llm_log": lead_request_log
            }
        )

        # Step 1: Assistants provide analysis and extract KEUs
        initial_opinions_parsed = []
        investigations = []
        assistant_contributions_for_ccp = []
        for a_mem in assist_members:
            investigation_prompt = (
                f"You are {a_mem.role} in medical group '{self.group_id}' with the goal: '{self.goal}'.\n"
                f"Provide your investigation summary or analysis focusing on your expertise regarding the medical query:\n"
                f"Question: {self.question_context['question']}\n"
                f"Your group lead ({self.lead_agent.role}, ID: {self.lead_agent.agent_id}) requires your input based on the following request:\n'{lead_request}'\n\n"
                f"Your output must be a JSON object with three fields: 'explanation' (your detailed reasoning), 'answer' (your final conclusion), "
                f"and 'keus' (a list of key evidential units). Each KEU in the list should be a string representing a single, verifiable piece of evidence "
                f"from the case (e.g., 'A 2cm nodule is visible in the upper left lung lobe.', 'The patient's white blood cell count is 15,000/µL.')."
            )
            if self.question_context.get('options'):
                options_str = ""
                for key, value in self.question_context['options'].items():
                    options_str += f"{key}: {value}\n"
                investigation_prompt += f"Options:\n{options_str}\n"

            if self.question_context.get('image_path'):
                investigation_prompt += f"(Image provided)\n"

            investigation_prompt += "\nKeep your response focused and relevant to the group's goal."

            investigation_json_str, system_message, user_message = a_mem.chat(
                prompt=investigation_prompt,
                image_path=self.question_context.get('image_path'),
                response_format={"type": "json_object"}
            )
            # Parse and log investigation
            try:
                investigation = json.loads(preprocess_response_string(investigation_json_str))
            except json.JSONDecodeError:
                investigation = {"explanation": investigation_json_str, "answer": "parse_error", "keus": []}

            parsed_opinion = {
                "agent_id": a_mem.agent_id,
                "agent_role": a_mem.role,
                "answer": investigation.get("answer"),
                "explanation": investigation.get("explanation", ""),
                "keus": investigation.get("keus", [])
            }
            initial_opinions_parsed.append(parsed_opinion)
            investigations.append({"role": a_mem.role, "id": a_mem.agent_id, "report": investigation})
            self._log_interaction(
                f"Assistant ({a_mem.agent_id} - {a_mem.role}) provided report.",
                data={
                    "step": "2_assistant_analysis",
                    "agent_id": a_mem.agent_id,
                    "agent_role": a_mem.role,
                    "prompt": investigation_prompt,
                    "response": investigation,
                    "llm_log": investigation_log
                }
            ) 
            # 机制三: 审计特定领域知识、风险规避
            explanation = investigation.get("explanation", "")
            contribution_audit = auditor_agent.audit_domain_agent_contribution(self.question_context['question'], a_mem.agent_id, a_mem.role, explanation)
            risk_audit = auditor_agent.audit_risk_and_quality(a_mem.agent_id, explanation, image_path=self.question_context.get('image_path'))
            step_id = f"advanced_group_{self.group_id}_assist_{a_mem.agent_id}"
            audit_trail["collaboration_audits"][step_id] = {**contribution_audit, **risk_audit}

            # 机制一：定义证据单元
            if "keus" in investigation:
                for keu_content in investigation["keus"]:
                    keu_id = f"KEU-{keu_counter}"
                    new_keu = KEU(keu_id, keu_content, a_mem.agent_id, 1, self.group_id) 
                    audit_trail["keus"][keu_id] = new_keu
                    keu_counter += 1
            
            # 机制二：记录观点 [FIXED]
            if audit_trail.get("viewpoints") is None: audit_trail["viewpoints"] = {}
            if a_mem.agent_id not in audit_trail["viewpoints"]: audit_trail["viewpoints"][a_mem.agent_id] = []
            
            keus_from_this_agent = [f"KEU-{i}" for i, keu_content in enumerate(investigation.get("keus", []), start=keu_counter - len(investigation.get("keus", [])))]
            audit_trail["viewpoints"][a_mem.agent_id].append({
                "step": f"advanced_group_{self.group_id}_initial", 
                "viewpoint": investigation.get("answer"),
                "viewpoint_changed": False,
                "justification_type": "initial_analysis",
                "cited_references": keus_from_this_agent
            })
            assistant_contributions_for_ccp.append({'agent_id': a_mem.agent_id, 'role': a_mem.role, 'text': explanation})
            
        # 机制一：调用审计员代理来识别哪些KEU是关键的
        all_keus_for_audit = [{"keu_id": k, "content": v.content} for k, v in audit_trail["keus"].items()]
        if all_keus_for_audit:
            key_status_map = auditor_agent.identify_key_evidential_units(
                self.question_context['question'], 
                initial_opinions_parsed, 
                all_keus_for_audit,
                image_path=self.question_context.get('image_path')
            )
            for keu_id, is_key in key_status_map.items():
                if keu_id in audit_trail["keus"]:
                    audit_trail["keus"][keu_id].is_key = is_key
                
        # 机制四：检测其中的矛盾。
        initial_ccps = auditor_agent.identify_critical_conflicts(assistant_contributions_for_ccp, f"initial analyses in group {self.group_id}")
        if audit_trail["ccps"].get(self.group_id) is None: audit_trail["ccps"][self.group_id] = []
        for ccp in initial_ccps:
            ccp['ccp_id'] = f"CCP-{ccp_counter}"
            ccp['status'] = 'unresolved'
            audit_trail["ccps"][self.group_id].append(ccp)
            ccp_counter += 1
        
        ccp_text_for_prompt = ""
        if initial_ccps:
            ccp_text_for_prompt += "\n\n[ATTENTION] The following Critical Conflict Points (CCPs) have been identified within your team and MUST be addressed:\n"
            for ccp in initial_ccps:
                ccp_text_for_prompt += f"- CCP ID: {ccp['ccp_id']}, Conflict: {ccp['conflict_summary']} (Involved: {', '.join(ccp['conflicting_agents'])})\n"

        # Step 2: Lead synthesizes information
        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed so far:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"
            
        gathered_investigation = "Gathered insights from assistant clinicians:\n" + "\n".join([f"--- Report from {inv['role']} (ID: {inv['id']}) ---\n{json.dumps(inv['report'])}\n---" for inv in investigations])
        
        synthesis_prompt = (
            f"{gathered_investigation}\n\n"
            f"As the lead of group '{self.group_id}', synthesize the information to provide a comprehensive report for the group's goal: '{self.goal}'(including your own initial thoughts if applicable).\n"
            f"Available KEUs (Key Evidential Units):\n{keu_list_text}\n"
            f"Available CCPs (Critical Consensus Points):{ccp_text_for_prompt}\n"
            f"Note that potential conflicts (CCPs) have been identified; a robust synthesis must acknowledge or resolve these points.\n\n"
            f"**CRITICAL INSTRUCTION:** Your task is to synthesize these diverse opinions into a single, coherent analysis. In your 'explanation', you **MUST selectively cite only the most important KEU-IDs** that support your synthesized view. **DO NOT simply list all available KEUs.** Your goal is to demonstrate a deep understanding by building a new, consolidated argument from the strongest evidence (e.g., 'Synthesizing the specialists' views, the consensus leans towards X, primarily supported by the crucial findings in KEU-2 and KEU-5...').\n\n"
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
        
        # 机制三: 在团队领导决策前，审计论据综合质量得分 [FIXED]
        arguments_for_quality_audit = [
            {
                'agent_id': opinion['agent_id'],
                'agent_role': opinion['agent_role'],
                'answer': opinion.get('answer'),
                'explanation': opinion.get('explanation')
            } for opinion in initial_opinions_parsed
        ]
        quality_audit_before_synthesis = auditor_agent.audit_overall_quality_for_decision(self.question_context['question'], arguments_for_quality_audit)
        audit_trail["collaboration_audits"][f"advanced_group_{self.group_id}_pre_synthesis_quality"] = quality_audit_before_synthesis
        
        final_report_str, system_message, user_message = self.lead_agent.chat(
            prompt=synthesis_prompt,
            image_path=self.question_context.get('image_path'),
            response_format={"type": "json_object"},
        )
        
        try:
            final_report = json.loads(preprocess_response_string(final_report_str))
        except json.JSONDecodeError:
            final_report = {"explanation": final_report_str, "answer": "parse_error"}

        # 机制三：审计元智能体的风险规避和角色扮演质量
        synthesis_explanation = final_report.get("explanation", "")
        synthesis_risk_audit = auditor_agent.audit_risk_and_quality(self.lead_agent.agent_id, synthesis_explanation, image_path=self.question_context.get('image_path'))
        synthesis_quality_audit = auditor_agent.audit_single_argument_quality(self.question_context['question'], synthesis_explanation, image_path =self.question_context.get('image_path'),domain_agent_quality_domain= quality_audit_before_synthesis)
        step_id = f"advanced_group_{self.group_id}_synthesis_{self.lead_agent.agent_id}"
        audit_trail["collaboration_audits"][step_id] = {**synthesis_risk_audit, **synthesis_quality_audit}

        # 机制一：检查元智能体是否正确引用了KEU
        for keu_id, keu in audit_trail["keus"].items():
            if keu.group_id == self.group_id and (keu_id in synthesis_explanation or keu.content in synthesis_explanation):
                keu.cited_by.append({"agent_id": self.lead_agent.agent_id, "round": 1, "action": "group_synthesis"})
        
        # 机制四：检查元智能体是否解决了CCP
        for ccp in audit_trail["ccps"].get(self.group_id, []):
            if ccp['status'] == 'unresolved':
                was_addressed, _ = analysis_llm.check_if_conflict_was_addressed(ccp, synthesis_explanation)
                if was_addressed:
                    ccp['status'] = 'resolved'

        self._log_interaction(f"Lead ({self.lead_agent.agent_id}) generated final group report.",
                               data={"agent_id": self.lead_agent.agent_id,
                                    "agent_role": self.lead_agent.role,
                                    "prompt": synthesis_prompt,
                                    "response": final_report, 
                                    "llm_log": final_report_log})
        return json.dumps(final_report), self.internal_log, keu_counter, ccp_counter 

# --- MDAgents Framework Class (Heavily Modified for Auditing) ---

class MDAgentsFramework:
    def __init__(self, 
                 log_dir: str, 
                 dataset_name: str, 
                 model_config: Dict[str, str],
                 auditor_model_key: str, 
                 conflict_model_key: str, # New args
                 config_path: str,
                 num_experts_intermediate: int = DEFAULT_NUM_EXPERTS_INTERMEDIATE,
                 num_teams_advanced: int = DEFAULT_NUM_TEAMS_ADVANCED,
                 num_agents_per_team_advanced: int = DEFAULT_NUM_AGENTS_PER_TEAM_ADVANCED):
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.model_config = model_config
        self.num_experts_intermediate = num_experts_intermediate
        self.num_teams_advanced = num_teams_advanced
        self.num_agents_per_team_advanced = num_agents_per_team_advanced
        self.config_path = config_path
        os.makedirs(self.log_dir, exist_ok=True) 

        self.moderator_agent = Agent("moderator", 
                                        AgentType.MODERATOR, 
                                        model_config.get('moderator', DEFAULT_MODERATOR_MODEL),
                                        config_path=config_path,
                                        instruction="You are a medical expert who conducts initial assessment. Your job is to decide the difficulty/complexity of the medical query based on the provided definitions. Respond in JSON format."
        )
        self.recruiter_agent = Agent("recruiter", 
                                        AgentType.RECRUITER, 
                                        model_config.get('recruiter', DEFAULT_RECRUITER_MODEL),
                                        config_path=config_path,
                                        instruction="You are an experienced medical expert who recruits appropriate specialists based on the medical query and its complexity level. Respond in JSON format."
        )
        self.decision_maker_agent = Agent("final_decision_maker", 
                                            AgentType.DECISION_MAKER, 
                                            model_config.get('moderator', DEFAULT_MODERATOR_MODEL),
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
        response, system_message, user_message = self.moderator_agent.chat(
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
                f"Specify their role, a brief expertise description, and optionally a communication hierarchy (e.g., 'Cardiologist > Nurse', 'Independent'). "
                f"Respond in JSON format."
            )
            self.recruiter_agent.instruction = recruitment_instruction
            prompt = (
                f"{query_context}\n"
                f"Recruit {self.num_experts_intermediate} experts for this moderately complex query.\n"
                f"Respond with a JSON array 'experts' where each expert object has fields: 'role', 'expertise', and 'hierarchy'.\n"
                f"Example response structure:\n"
                f"{{\"experts\": [\n"
                f"  {{\"role\": \"Pediatrician\", \"expertise\": \"Specializes in child healthcare\", \"hierarchy\": \"Independent\"}},\n"
                f"  {{\"role\": \"Cardiologist\", \"expertise\": \"Focuses on heart conditions\", \"hierarchy\": \"Pediatrician > Cardiologist\"}},\n"
                f"  {{\"role\": \"Pulmonologist\", \"expertise\": \"Specializes in respiratory disorders\", \"hierarchy\": \"Independent\"}}\n"
                f"]}}"
            )
        elif complexity == ComplexityLevel.ADVANCED:
            recruitment_instruction = (
                f"You are an experienced medical director. Your task is to organize {self.num_teams_advanced} Multidisciplinary Teams (MDTs) "
                f"for a complex medical query. Each MDT should have around {self.num_agents_per_team_advanced} clinicians. Define the purpose (goal) of each team "
                f"and list its members with their roles and expertise. Ensure you include an 'Initial Assessment Team (IAT)' and a 'Final Review and Decision Team (FRDT)'. "
                f"Respond in JSON format."
            )
            self.recruiter_agent.instruction = recruitment_instruction
            prompt = (
                f"{query_context}\n"
                f"Organize {self.num_teams_advanced} MDTs, each with ~{self.num_agents_per_team_advanced} members, for this complex query.\n"
                f"Include an IAT and an FRDT.\n"
                f"Respond with a JSON object containing a 'teams' array where each team has: 'group_id', 'goal', and 'members' array.\n"
                f"Each member should have 'role', 'expertise', and optionally 'is_lead' (boolean) fields.\n"
                f"Example structure:\n"
                f"{{\"teams\": [\n"
                f"  {{\"group_id\": \"Group 1\", \"goal\": \"Initial Assessment Team (IAT)\", \"members\": [\n"
                f"    {{\"role\": \"Emergency Physician\", \"expertise\": \"Handles acute assessment\", \"is_lead\": true}},\n"
                f"    {{\"role\": \"Triage Nurse\", \"expertise\": \"Gathers initial patient data\"}}\n"
                f"  ]}},\n"
                f"  {{\"group_id\": \"Group 2\", \"goal\": \"Final Review and Decision Team (FRDT)\", \"members\": [\n"
                f"    {{\"role\": \"Senior Consultant\", \"expertise\": \"Oversees final decision\", \"is_lead\": true}},\n"
                f"    {{\"role\": \"Clinical Pharmacist\", \"expertise\": \"Medication review\"}}\n"
                f"  ]}}\n"
                f"]}}"
            )

        recruitment_response, system_message, user_message = self.recruiter_agent.chat(
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
                    if isinstance(expert, dict) and 'role' in expert:
                        validated_expert = {
                            "role": expert.get("role", "Unknown Role"),
                            "expertise": expert.get("expertise", "General expertise related to the role."),
                            "hierarchy": expert.get("hierarchy", "Independent")
                        }
                        validated_experts.append(validated_expert)
                experts = validated_experts
                specialties = [e['role'] for e in experts]
            except Exception as e:
                print(f"Error parsing expert recruitment response: {e}. Raw response: {recruitment_response}")
                print("Warning: Failed to parse experts. Using default roles.")
                step_log["error"] = f"Error parsing response: {e}"
                step_log["warning"] = "Using default roles due to parsing error."
                default_roles = ["Internal Medicine Specialist", "Radiologist", "Surgeon", "Pathologist", "Pharmacist"]
                experts = [{"role": r, "expertise": f"Expertise in {r}", "hierarchy": "Independent"} for r in default_roles[:self.num_experts_intermediate]]

            print(f"Recruited Experts: {[e['role'] for e in experts]}")
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
                            if isinstance(member, dict) and 'role' in member:
                                validated_member = {
                                    "role": member.get("role", "Unknown Role"),
                                    "expertise": member.get("expertise", "General expertise for the role.")
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
                        {"role": "Emergency Physician", "expertise": "Acute assessment", "is_lead": True},
                        {"role": "Radiologist", "expertise": "Initial imaging"}
                    ]},
                    {"group_id": "Group 2", "goal": "Diagnostic Team", "members": [
                        {"role": "Cardiologist", "expertise": "Heart conditions", "is_lead": True},
                        {"role": "Neurologist", "expertise": "Nervous system"}
                    ]},
                    {"group_id": "Group 3", "goal": "Final Review and Decision Team (FRDT)", "members": [
                        {"role": "Senior Consultant", "expertise": "Oversees decision", "is_lead": True},
                        {"role": "Clinical Pharmacist", "expertise": "Medication review"}
                    ]}
                ]
                teams = teams[:self.num_teams_advanced]
                specialties = []
                for team in teams:
                    for member in team['members']:
                        specialties.append(member['role'])
            print(f"Recruited Teams: {[t['goal'] for t in teams]}")
            step_log["recruited_personnel"] = teams
            step_log["specialties"] = specialties
            return teams, step_log
        return [], step_log

    def _process_basic_query(self, data_item: Dict) -> Dict:
        print("\n--- Processing Basic Query ---")
        case_history = {"rounds": []}
        audit = {"rounds": []}
        audit_round_data = {
            "round": 1,
            "2_1_1_role_assignment": [], 
            "2_1_2_domain_specific_knowledge_activation": [], 
            
            "2_2_1_repetition_of_initial_views": [], 
            "2_2_2_unresolved_conflicts": [],
            
            "3_1_1_suppression_of_minority_views": [],
            "3_1_2_authority_bias": [],
            "3_1_3_neglect_of_contradictions": [],
            "3_2_1_self_contradiction_when_decision": []
        }
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)
        agent = Agent(
            agent_id="basic_solver",
            role=MedicalSpecialty.GENERAL_MEDICINE,
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

        response, system_message, user_message = agent.chat(prompt=main_prompt, 
                                       image_path=data_item.get('image_path'), 
                                       response_format={"type": "json_object"})
    
        llm_log = {
            "llm_input": {
                "system_message": system_message,
                "user_message": user_message
        }
        }

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

            
            # audit 3.1.3: Neglect of Contradictions in Reasoning Process for decision-maker
            audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker = self.auditor_agent.audit_contradictions_during_decision(
                question = data_item['question'], current_agent_id = agent.agent_id, explanation = explanation, case_history = case_history
            ) 

            audit_round_data["3_1_3_neglect_of_contradictions"].append({
                "agent_id": agent.agent_id,
                "step": "decision",
                "audit_result": audit_results_of_neglect_of_contradictions_in_reasoning_process_for_decision_maker
            })
            audit.append(audit_round_data)
            case_history["rounds"][-1]["decision"] = {"parsed_output": result}
            case_history["audit"] = audit
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


    def _process_intermediate_query(self, data_item: Dict, expert_configs: List[Dict], audit_results_of_role_assignment, specialties) -> Dict:
        case_history = {"rounds": []}
        audit = {"rounds": []}
        audit_round_data = {
        "round": 1,
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
        detailed_log = {
            "step_name": "process_intermediate_query",
            "expert_configs": expert_configs,
            "initial_opinions": [],
            "final_synthesis": {}
        }
        agents = []
        for i, config in enumerate(expert_configs):
            agent_id = f"expert_{i+1}_{config['role'].replace(' ','_').lower()}"
            agent = Agent(
                agent_id=agent_id,
                role=config['role'],
                model_key=agent_model_key,
                instruction=f"You are a {config['role']} with expertise in {config['expertise']}. Collaborate with other medical experts to answer the medical query. Maintain your persona and provide insights based on your specialty. Respond in JSON format."
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
        initial_contributions_for_ccp = []
        initial_opinions_parsed = [] 
        for agent in agents:
            agent.clear_memory()
            prompt = (
                f"{question_context}\n"
                f"Based on your expertise as a {agent.role}, provide your initial analysis and answer.\n"
                f"Respond with a JSON object containing 'answer' and 'explanation' fields."
            )
            response, system_message, user_message = agent.chat(
                prompt=prompt,
                image_path=image_path,
                response_format={"type": "json_object"},
            )

            llm_log = {
                "llm_input": {
                    "system_message": system_message,
                    "user_message": user_message
                }
            }

            opinion_log = {
                "agent_id": agent.agent_id,
                "agent_role": agent.role,
                "prompt": prompt,
                "llm_log": llm_log,
            }

            try:
                response_clean = preprocess_response_string(response)
                response_json = json.loads(response_clean)
                ans = response_json.get("answer", "")
                expl = response_json.get("explanation", "No explanation provided.")
                opinion_log["parsed_response"] = response_json
                
                if options and isinstance(ans, str):
                    if ans.startswith('(') and ans.endswith(')'):
                        ans = ans[1:-1].strip()
                    elif ans.startswith('(') and len(ans) > 2 and ans[1].isalpha() and ans[2] == ')':
                        ans = ans[1]
                    elif len(ans) > 1 and ans[0].isalpha() and (ans[1] == '.' or ans[1] == ')'):
                        ans = ans[0]

                # audit 2.1.2 domain-specific knowledge activation
                audit_results_of_domain_specific_knowledge_activation = self.auditor_agent.audit_domain_specific_knowledge_activation(question, image_path, agent.agent_id, agent.role, ans, explanation)


            except Exception as e:
                print(f"Error parsing initial opinion from {agent.agent_id}: {e}. Raw response: {response}")
                ans = "Could not parse answer."
                expl = "Error parsing model response."
                opinion_log["error"] = f"Error parsing response: {e}"
            
            parsed_opinion = {
                "agent_id": agent.agent_id,
                "agent_role": agent.role,
                "answer": ans,
                "explanation": expl,
            }
            initial_opinions_parsed.append(parsed_opinion)
            
            initial_report_parts.append(f"Expert {agent.role} ({agent.agent_id}):\nAnswer: {ans}\nExplanation: {expl}\n---")
            print(f"Agent {agent.agent_id} ({agent.role}) Initial Answer: {ans}")
            detailed_log["initial_opinions"].append(opinion_log)

            
            
            
            explanation = response_json.get("explanation", "")
            initial_contributions_for_ccp.append({'agent_id': agent.agent_id, 'role': agent.role, 'text': explanation})
        


        print("\n-- Synthesizing Final Decision --")
        self.decision_maker_agent.clear_memory()

                

        initial_reports_text = "\n".join(initial_report_parts)
        
        synthesis_prompt = (
            f"You need to make a final decision for the following medical query based on initial opinions from a team of experts:\n\n"
            f"{question_context}\n\n"
            f"--- Expert Opinions (Round 1) ---\n"
            f"{''.join(initial_report_parts)}\n"
            f"--- End Opinions ---\n\n"
            f"Review these opinions carefully. Consider the different expert perspectives and their specific expertise.\n"
            f"Respond with a JSON object containing 'answer' (letter for multiple-choice) and 'explanation' fields."
        )
             
        arguments_for_audit = initial_opinions_parsed 
        quality_audit = self.auditor_agent.audit_overall_quality_for_decision(data_item['question'], arguments_for_audit)
        audit_trail["collaboration_audits"]["intermediate_pre_decision_quality"] = quality_audit
        
        final_response, final_system_message, final_user_message = self.decision_maker_agent.chat(
            prompt=synthesis_prompt,
            response_format={"type": "json_object"},
            image_path=image_path
        )
        detailed_log["final_synthesis"] = {
            "agent_id": self.decision_maker_agent.agent_id,
            "agent_role": self.decision_maker_agent.role,
            "prompt": synthesis_prompt,
            "llm_log": final_llm_log
        }
        try:
            response_clean = preprocess_response_string(final_response)
            response_json = json.loads(response_clean)
            final_answer = response_json.get("answer", "")
            final_explanation = response_json.get("explanation", "No explanation provided.")
            detailed_log["final_synthesis"]["parsed_response"] = response_json

            if options and isinstance(final_answer, str):
                if final_answer.startswith('(') and final_answer.endswith(')'):
                    final_answer = final_answer[1:-1].strip()
                elif final_answer.startswith('(') and len(final_answer) > 2 and final_answer[1].isalpha() and final_answer[2] == ')':
                    final_answer = final_answer[1]
                elif len(final_answer) > 1 and final_answer[0].isalpha() and (final_answer[1] == '.' or final_answer[1] == ')'):
                    final_answer = final_answer[0]
        except Exception as e:
            print(f"Error parsing final decision: {e}. Raw response: {final_response}")
            final_answer = "Could not parse answer."
            final_explanation = "Error parsing model response."
            detailed_log["final_synthesis"]["error"] = f"Error parsing response: {e}"
        print(f"Intermediate Query Final Result: Answer='{final_answer}', Explanation='{final_explanation[:100]}...'")

        risk_audit = self.auditor_agent.audit_risk_and_quality(self.decision_maker_agent.agent_id, final_explanation, image_path=image_path)
        quality_audit = self.auditor_agent.audit_single_argument_quality(data_item['question'], final_explanation, image_path=image_path, domain_agent_quality_domain=quality_audit)
        audit_trail["collaboration_audits"]["intermediate_final_decision"] = {**risk_audit, **quality_audit}
        
        for keu_id, keu in audit_trail["keus"].items():
            if keu_id in final_explanation or keu.content in final_explanation:
                keu.present_in_final_decision = True

        resolved_this_round_log = []
        still_unresolved = []
        for ccp in all_unresolved_ccps:
            was_addressed, resolution_reasoning = self.analysis_llm.check_if_conflict_was_addressed(ccp, final_explanation)
            if was_addressed:
                ccp['status'] = 'resolved'
                ccp['round_resolved'] = 1
                ccp['resolution_reasoning'] = resolution_reasoning
                resolved_this_round_log.append(ccp)
            else:
                still_unresolved.append(ccp)
        all_unresolved_ccps = still_unresolved
        print(f"Round 1: {len(resolved_this_round_log)} CCP(s) were resolved.")

        return {
            "predicted_answer": final_answer,
            "explanation": final_explanation,
            "detailed_log": detailed_log,
            "complexity": ComplexityLevel.INTERMEDIATE.value,
        }
    def _process_advanced_query(self, data_item: Dict, team_configs: List[Dict], audit_trail: Dict) -> Dict:
        print("\n--- Processing Advanced Query ---")
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)
        detailed_log = {
            "step_name": "process_advanced_query",
            "team_configs": team_configs,
            "team_discussions": [],
            "final_decision_synthesis": {}
        }
        keu_counter = len(audit_trail["keus"])
        ccp_counter = sum(len(v) for v in audit_trail["ccps"].values())
        question_context = {
            "question": data_item["question"],
            "options": data_item.get("options"),
            "image_path": data_item.get("image_path"),
        }
        groups = []

        for i, config in enumerate(team_configs):
            members = []
            for j, member_config in enumerate(config['members']):
                agent_id = f"{config['group_id'].replace(' ','_').lower()}_member_{j+1}_{member_config['role'].replace(' ','_').lower()}"
                is_lead = member_config.get('is_lead', False)
                instruction_prefix = f"You are {member_config['role']} ({member_config['expertise']}) in team '{config['goal']}'."
                if is_lead:
                    instruction_prefix += " You are the LEAD of this team."
                agent = Agent(
                    agent_id=agent_id,
                    role=f"{member_config['role']}{' (Lead)' if is_lead else ''}",
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
            raw_report, discussion_log, keu_counter, ccp_counter = group.perform_internal_discussion(
                auditor_agent=self.auditor_agent,
                analysis_llm=self.analysis_llm,
                audit_trail=audit_trail,
                keu_counter=keu_counter,
                ccp_counter=ccp_counter
            )
            
            detailed_log["team_discussions"].append({
                "group_id": group.group_id,
                "goal": group.goal,
                "discussion_log": discussion_log
            })

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

        # [FIXED] Inject KEU and CCP context into the final decision prompt
        all_unresolved_ccps_list = []
        for group_id, ccp_list in audit_trail.get("ccps", {}).items():
            for ccp in ccp_list:
                if 'resolved' not in ccp.get('status', 'unresolved'):
                    all_unresolved_ccps_list.append(ccp)
        
        ccp_text_for_prompt = ""
        if all_unresolved_ccps_list:
            ccp_text_for_prompt += "\n\n[ATTENTION] The following Critical Conflict Points (CCPs) from team discussions remain UNRESOLVED and MUST be addressed:\n"
            for ccp in all_unresolved_ccps_list:
                ccp_text_for_prompt += f"- CCP ID: {ccp['ccp_id']}, Conflict: {ccp['conflict_summary']} (Involved: {', '.join(ccp['conflicting_agents'])})\n"
        
        keu_list_text = "\n\nKey Evidential Units (KEUs) proposed across all teams:\n"
        for keu_id, keu_obj in audit_trail["keus"].items():
            keu_list_text += f"- {keu_id}: '{keu_obj.content}' (from {keu_obj.source_agent})\n"

        synthesis_prompt = (
            f"You need to make the ultimate final decision for the following complex medical query based on reports from multiple specialized teams:\n\n"
            f"Original Query Context:\nQuestion: {data_item['question']}\n{options_str}"
            f"{'(Image associated)' if data_item.get('image_path') else ''}\n\n"
            f"--- Compiled Team Reports ---\n"
            f"{all_reports_for_synthesis}\n"
            f"--- End Reports ---\n\n"
            f"Available KEUs (Key Evidential Units):\n{keu_list_text}\n"
            f"Available CCPs (Critical Consensus Points):\n{ccp_text_for_prompt}\n\n"
            f"**CRITICAL INSTRUCTION:** Your task is to synthesize all available information into a single, definitive analysis. In your 'explanation', you **MUST selectively cite the most pivotal KEU-IDs** from across all teams to build your final argument and **explicitly address any unresolved CCPs**. Your goal is to demonstrate a deep, cross-team understanding.\n\n"
            f"Synthesize all this information into one final, definitive answer and explanation.\n"
            f"Respond with a JSON object containing 'answer' (letter for multiple-choice) and 'explanation' fields."
        )
        final_response, final_system_message, final_user_message = self.decision_maker_agent.chat(
            prompt=synthesis_prompt,
            response_format={"type": "json_object"},
            image_path=data_item.get("image_path")
        )
        
        detailed_log["final_decision_synthesis"] = {
            "agent_id": self.decision_maker_agent.agent_id,
            "agent_role": self.decision_maker_agent.role,
            "prompt": synthesis_prompt,
            "llm_log": final_llm_log
        }
        try:
            response_clean = preprocess_response_string(final_response)
            response_json = json.loads(response_clean)
            final_answer = response_json.get("answer", "")
            final_explanation = response_json.get("explanation", "No explanation provided.")
            detailed_log["final_decision_synthesis"]["parsed_response"] = response_json

            options = data_item.get('options')
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
            detailed_log["final_decision_synthesis"]["error"] = f"Error parsing response: {e}"
        print(f"Advanced Query Final Result: Answer='{final_answer}', Explanation='{final_explanation[:100]}...'")

        for keu_id, keu in audit_trail["keus"].items():
            if keu_id in final_explanation or keu.content in final_explanation:
                keu.present_in_final_decision = True
        
        all_unresolved_ccps = []
        for group_id, ccp_list in audit_trail.get("ccps", {}).items():
            for ccp in ccp_list:
                if ccp.get('status') == 'unresolved':
                    all_unresolved_ccps.append(ccp)
                    
        for ccp in all_unresolved_ccps:
            was_addressed, _ = self.analysis_llm.check_if_conflict_was_addressed(ccp, final_explanation)
            if was_addressed:
                ccp['status'] = 'resolved_in_final_decision'

        risk_audit = self.auditor_agent.audit_risk_and_quality(self.decision_maker_agent.agent_id, final_explanation, image_path=data_item.get("image_path"))
        quality_audit = self.auditor_agent.audit_single_argument_quality(data_item['question'], final_explanation, image_path=data_item.get("image_path"))
        audit_trail["collaboration_audits"]["advanced_final_decision"] = {**risk_audit, **quality_audit}
        
        return {
            "predicted_answer": final_answer,
            "explanation": final_explanation,
            "detailed_log": detailed_log,
            "complexity": ComplexityLevel.ADVANCED.value,
        }
    def run_query(self, data_item: Dict) -> Dict:
        qid = data_item["qid"]
        print(f"\n{'='*20} Processing QID: {qid} {'='*20}")
        start_time = time.time()
        process_log = []
        result_data = {}
        try:
            complexity, complexity_log = self._determine_complexity(data_item["question"], data_item.get("options"), data_item.get("image_path"))
            process_log.append(complexity_log)

            if complexity == ComplexityLevel.BASIC:
                result_data = self._process_basic_query(data_item=data_item)
            else:
                recruited, recruitment_log = self._recruit_experts(data_item["question"], data_item.get("options"), complexity, data_item.get("image_path"))
                process_log.append(recruitment_log)
                specialties = recruitment_log.get("specialties", [])
                # audit 2.1.1 role assignment
                audit_results_of_role_assignment = self.auditor_agent.audit_role_assignment(question=data_item["question"], image_path=data_item.get("image_path"), specialties=specialties)
                if complexity == ComplexityLevel.INTERMEDIATE:
                    result_data = self._process_intermediate_query(data_item, recruited, audit_results_of_role_assignment, specialties)
                elif complexity == ComplexityLevel.ADVANCED:
                    result_data = self._process_advanced_query(data_item, recruited, audit_results_of_role_assignment, specialties)

        except Exception as e:
            print(f"ERROR processing QID {qid}: {e}")
            result_data = {"predicted_answer": "Error", "explanation": str(e), "complexity": "unknown"}
            audit_trail["fatal_error"] = str(e)

        processing_time = time.time() - start_time
        

        return {
            "qid": qid, "question": data_item["question"], "options": data_item.get("options"),
            "ground_truth": data_item.get("answer"), "complexity_level": result_data.get("complexity"),
            "predicted_answer": result_data.get("predicted_answer"), "explanation": result_data.get("explanation"),
            "processing_time_seconds": processing_time,
            "process_log": process_log, 
            "case_history": result_data.get("case_history", {}),
        }

    def run_dataset(self, data: List[Dict]):
        for item in tqdm(data, desc=f"Running MDAgents on {self.dataset_name}"):
            qid = item.get("qid", "unknown_qid")
            result_path = os.path.join(self.log_dir, f"{qid}-result.json")
            if os.path.exists(result_path):
                print(f"Skipping {qid} - result file already exists.")
                continue
            try:
                result = self.run_query(data_item=item)
                save_json(result, result_path)
            except Exception as e:
                print(f"FATAL ERROR for QID {qid}: {e}")
                save_json({"qid": qid, "error": str(e)}, os.path.join(self.log_dir, f"{qid}-error.json"))

def main():
    parser = argparse.ArgumentParser(description="Run MDAgents Framework on medical datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], default="mc")
    parser.add_argument("--moderator_model", type=str, required=True,help = "deepseek-reasoner/gpt-5.1/gemini-2.5-flash")
    parser.add_argument("--recruiter_model", type=str, required=True, help = "deepseek-reasoner/gpt-5.1/gemini-2.5-flash")
    parser.add_argument("--agent_model", type=str, required=True, help = "qa = deepseek-reasoner/gpt-5.1/gemini-2.5-flash ,vqa = qwen-3-vl/gpt-5.1/gemini-2.5-flash") # this agent need to read the figure if we execute the vqa task
    parser.add_argument("--auditor_model", type=str, required=True, help="gemini-3-pro-preview")
    parser.add_argument("--config_path", type=str, required=True,help="Path to the config.toml file,default = utils/config.toml")
    parser.add_argument("--num_samples",type = int, required = True, help = "number of samples to run")
    parser.add_argument("--num_experts", type=int, default=DEFAULT_NUM_EXPERTS_INTERMEDIATE)
    parser.add_argument("--num_teams", type=int, default=DEFAULT_NUM_TEAMS_ADVANCED)
    parser.add_argument("--time_stamp", type=str, required=True, help="Timestamp for logging purposes")

    
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")

    timestamp = args.time_stamp
    current_model_name = current_file_name.split("_")[2]

    terminal_log_dir = project_root / "logs" / "observation" / timestamp / current_model_name / dataset_name / "terminal_log"
    terminal_log_dir.mkdir(parents=True, exist_ok=True)
    terminal_log_file = terminal_log_dir / f"{args.dataset}_{timestamp}_full_terminal.log"
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr) # 捕获报错和tqdm进度条

    logs_dir = project_root / "logs" / "observation" / timestamp / current_model_name / dataset_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved to: {logs_dir}")
    
    data_path = f"./my_datasets/processed/medqa/{args.dataset}/medqa_{args.qa_type}_test.json"
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    model_config = {
        "moderator": args.moderator_model,
        "recruiter": args.recruiter_model,
        "default_agent": args.agent_model
    }

    framework = MDAgentsFramework(
        log_dir=logs_dir, dataset_name=args.dataset, model_config=model_config,
        auditor_model_key=args.auditor_model, conflict_model_key=args.auditor_model,
        num_experts_intermediate=args.num_experts, num_teams_advanced=args.num_teams,
        config_path = args.config_path
    )

    framework.run_dataset(data=data[:args.num_samples])

if __name__ == "__main__":
    main()