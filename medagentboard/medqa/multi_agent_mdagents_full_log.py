import os
import time
import argparse
from openai import OpenAI
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from tqdm import tqdm
import json
import time
# Utilities from the ColaCare framework
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.encode_image import encode_image
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string

# --- Constants and Enums ---

class ComplexityLevel(Enum):
    """Enumeration for medical query complexity levels."""
    BASIC = "basic"        # Maps to paper's "Low"
    INTERMEDIATE = "intermediate" # Maps to paper's "Moderate"
    ADVANCED = "advanced"    # Maps to paper's "High"

class AgentRole(Enum):
    """Enumeration for different agent roles in MDAgents."""
    MODERATOR = "Moderator" # 分诊护士，根据问题的描述、选项、以及是否包含图片，将问题归类为 简单、中等、复杂 三个等级中的一个。
    RECRUITER = "Recruiter" # (“医务科主任”) 根据上一步的复杂度结论，开始“招人”。
    GENERAL_DOCTOR = "General Doctor" # (“全科医生”) 智能体。
    SPECIALIST = "Specialist" # (“专科医生”) 具有特定领域知识的智能体。
    TEAM_LEAD = "Team Lead" # (“团队负责人”) 负责协调团队内部的工作。
    DECISION_MAKER = "Decision Maker" # (“决策者”) 负责最终决策的智能体。

# Default settings (can be overridden by arguments)
DEFAULT_MODERATOR_MODEL = "gemini-2.5-flash" # 分类/招募用的文本模型
DEFAULT_RECRUITER_MODEL = "gemini-2.5-flash" # 招募专家用的文本模型
DEFAULT_AGENT_MODEL = "gemini-2.5-flash" # 分析智能体的默认多模态模型
DEFAULT_MAX_ROUNDS_INTERMEDIATE = 3 # Max discussion rounds for intermediate 中等复杂度的最大讨论轮数
DEFAULT_MAX_TURNS_INTERMEDIATE = 3  # Max turns per round for intermediate 每轮的最大对话次数
DEFAULT_NUM_EXPERTS_INTERMEDIATE = 5 #  招募的专家数量
DEFAULT_NUM_TEAMS_ADVANCED = 3 # 高复杂度的团队数量
DEFAULT_NUM_AGENTS_PER_TEAM_ADVANCED = 3 # 每个团队的智能体数量

# --- Base Agent Class ---

class BaseAgent:
    """Base class for all agents in the MDAgents framework, adapted from ColaCare."""

    def __init__(self,
                 agent_id: str,
                 role: Union[AgentRole, str], # Allow custom roles string 接收两种类型值，既可以是枚举，也可以是自定义字符串
                 model_key: str,
                 instruction: Optional[str] = None):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent.
            role: The role of the agent (e.g., Moderator, Specialist).
            model_key: Key for the LLM model configuration in LLM_MODELS_SETTINGS.
            instruction: System-level instruction defining the agent's persona and task.
        """
        self.agent_id = agent_id
        self.role = role if isinstance(role, str) else role.value
        self.model_key = model_key
        self.instruction = instruction or f"You are a helpful assistant playing the role of a {self.role}."
        self.memory = [] # List to store message history for this agent

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        # Set up OpenAI compatible client based on model settings
        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.llm_client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]

        print(f"Initialized Agent: ID={self.agent_id}, Role={self.role}, Model={self.model_key} ({self.model_name})")

    # MODIFICATION START: Enhanced call_llm to return detailed log information
    def call_llm(self,
                 messages: List[Dict[str, Any]],
                 response_format: Optional[Dict[str, str]] = None, # e.g., {"type": "json_object"}
                 max_retries: int = 3) -> (str, Dict[str, Any]):
        """
        Call the LLM with a list of messages and handle retries.

        Args:
            messages: List of message dictionaries (e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": ...}]).
            response_format: Optional dictionary specifying the desired response format (e.g., JSON).
            max_retries: Maximum number of retry attempts.

        Returns:
            A tuple containing:
            - str: LLM response text.
            - dict: A log dictionary with the request payload and the raw response object.

        Raises:
            Exception: If LLM call fails after all retries.
        """
    # MODIFICATION END

        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM ({self.model_name}). Attempt {retries + 1}/{max_retries}.")

                completion_params = {
                    "model": self.model_name,
                    "messages": messages,
                }
                if response_format:
                    completion_params["response_format"] = response_format

                completion = self.llm_client.chat.completions.create(**completion_params) # 字典解包赋值

                response_content = completion.choices[0].message.content
                print(f"Agent {self.agent_id} received response successfully.")

                # Add user message and assistant response to this agent's memory
                if messages[-1]['role'] == 'user':
                    self.memory.append(messages[-1])
                    self.memory.append({"role": "assistant", "content": response_content})

                # MODIFICATION START: Create a detailed log of the LLM call
                llm_call_log = {
                    "request": completion_params,
                    "raw_response": completion.model_dump() # Convert pydantic model to dict for JSON serialization
                }
                return response_content, llm_call_log
                # MODIFICATION END

            except Exception as e:
                retries += 1
                print(f"LLM API call error for agent {self.agent_id} (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    # MODIFICATION START: Return error information in the log
                    error_log = {
                        "error": str(e),
                        "failed_request": completion_params
                    }
                    raise Exception(f"LLM API call failed for agent {self.agent_id} after {max_retries} attempts: {e}")
                    # In a production system, you might return (None, error_log) instead of raising
                    # MODIFICATION END

                time.sleep(1)

        raise Exception(f"LLM call failed unexpectedly for agent {self.agent_id}.")

    # MODIFICATION START: Enhanced chat method to return detailed log information
    def chat(self,
             prompt: str,
             image_path: Optional[str] = None,
             use_memory: bool = True,
             response_format: Optional[Dict[str, str]] = None) -> (str, Dict[str, Any]):
        """
        Simplified chat interface for an agent.

        Args:
            prompt: The user's message/query to the agent.
            image_path: Optional path to an image file (for multimodal models).
            use_memory: Whether to include the agent's history in the LLM call.
            response_format: Optional dictionary specifying the desired response format.

        Returns:
            A tuple containing:
            - str: The assistant's response text.
            - dict: A detailed log of the LLM call.
        """
    # MODIFICATION END
        system_message = {"role": "system", "content": self.instruction}

        # Prepare user message content (text + optional image)
        user_content: Union[str, List[Dict[str, Any]]]
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path does not exist: {image_path}")

            base64_image = encode_image(image_path)
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        else:
            user_content = prompt

        user_message = {"role": "user", "content": user_content}

        # Construct messages list
        messages = [system_message]
        if use_memory and self.memory:
            messages.extend(self.memory)
        messages.append(user_message)

        # MODIFICATION START: Capture response and log from call_llm
        response, llm_call_log = self.call_llm(messages, response_format=response_format)
        return response, llm_call_log
        # MODIFICATION END

    def clear_memory(self):
        """Clears the agent's conversation memory."""
        self.memory = []
        print(f"Cleared memory for agent {self.agent_id}")

# --- MDAgents Group Class ---

class Group:
    """Represents a team of agents working towards a common goal."""

    def __init__(self,
                 group_id: str,
                 goal: str,
                 members: List[BaseAgent],
                 question_context: Dict[str, Any]):
        """
        Initialize a group of agents.

        Args:
            group_id: Unique identifier for the group.
            goal: The objective or purpose of this group.
            members: A list of BaseAgent instances that are part of this group.
            question_context: Dictionary containing 'question', optional 'options', 'image_path'.
        """
        self.group_id = group_id
        self.goal = goal
        self.members = members
        self.question_context = question_context
        # MODIFICATION START: The internal_log will now be a structured list of dictionaries for better analysis
        self.internal_log = [] # This will now store structured log entries.
        # MODIFICATION END

        print(f"Initialized Group: ID={self.group_id}, Goal='{self.goal}', Members={[m.agent_id for m in self.members]}")

        # Identify lead agent
        self.lead_agent = None
        for member in members:
            if 'lead' in member.role.lower() or 'lead' in member.agent_id.lower():
                self.lead_agent = member
                break

        if not self.lead_agent and members:
            print(f"Warning: No explicit lead found in group {self.group_id}. Assigning first member '{members[0].agent_id}' as lead.")
            self.lead_agent = members[0]
        elif not members:
            print(f"Warning: Group {self.group_id} created with no members.")

    def _log_interaction(self, message: str, data: Optional[Dict] = None):
        """Adds a message and optional structured data to the internal group log."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "data": data or {}
        }
        print(f"[Group {self.group_id} Log] {message}")
        self.internal_log.append(log_entry)

    # MODIFICATION START: The method now returns the final report and a detailed log of the discussion.
    def perform_internal_discussion(self) -> (str, List[Dict]):
        """
        Simulates the internal discussion process within the group to achieve its goal.

        Returns:
            A tuple containing:
            - str: A string representing the group's synthesized report or conclusion.
            - list: A structured log of the entire internal discussion.
        """
    # MODIFICATION END
        if not self.members:
            self._log_interaction("No members in the group to perform discussion.", {"error": "No members"})
            return "Error: Group has no members.", self.internal_log

        if not self.lead_agent:
            self._log_interaction("No lead agent identified for coordination.", {"error": "No lead agent"})
            return "Error: Group has no lead agent.", self.internal_log

        self._log_interaction(f"Starting internal discussion. Lead: {self.lead_agent.agent_id} ({self.lead_agent.role})")

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
            # Using the readable style you requested
            options_str = ""
            for key, value in self.question_context['options'].items():
                options_str += f"{key}: {value}\n"
            delivery_prompt += f"Options:\n{options_str}\n"

        if self.question_context.get('image_path'):
            delivery_prompt += f"An associated image is provided.\n"

        delivery_prompt += "--- End Query ---\n\nProvide a concise summary of required investigations or your direct analysis if no assistants."

        lead_request, lead_request_log = self.lead_agent.chat(
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

        # 2. Assistants provide their investigations/analysis
        investigations = []
        for a_mem in assist_members:
            investigation_prompt = (
                f"You are {a_mem.role} (ID: {a_mem.agent_id}) in medical group '{self.group_id}' with the goal: '{self.goal}'.\n"
                f"Your group lead ({self.lead_agent.role}, ID: {self.lead_agent.agent_id}) requires your input based on the following request:\n'{lead_request}'\n\n"
                f"Please provide your investigation summary or analysis focusing on your expertise regarding the medical query:\n"
                f"Question: {self.question_context['question']}\n"
            )

            if self.question_context.get('options'):
                options_str = ""
                for key, value in self.question_context['options'].items():
                    options_str += f"{key}: {value}\n"
                investigation_prompt += f"Options:\n{options_str}\n"

            if self.question_context.get('image_path'):
                investigation_prompt += f"(Image provided)\n"

            investigation_prompt += "\nKeep your response focused and relevant to the group's goal."

            investigation, investigation_log = a_mem.chat(
                prompt=investigation_prompt,
                image_path=self.question_context.get('image_path'),
            )

            assistant_report_data = {
                "role": a_mem.role,
                "id": a_mem.agent_id,
                "report": investigation
            }
            investigations.append(assistant_report_data)
            
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

        # 3. Lead synthesizes the information
        gathered_investigation = ""
        if investigations:
            gathered_investigation += "Gathered insights from assistant clinicians:\n"
            for inv in investigations:
                gathered_investigation += f"--- Report from {inv['role']} (ID: {inv['id']}) ---\n{inv['report']}\n---\n"
        else:
            gathered_investigation = "No assistant reports were generated. Relying solely on lead's analysis."

        synthesis_prompt = f"{gathered_investigation}\n\n"
        synthesis_prompt += f"As the lead ({self.lead_agent.role}) of group '{self.group_id}' aiming to '{self.goal}', synthesize the gathered information (including your own initial thoughts if applicable) "
        synthesis_prompt += f"to provide a comprehensive report or final answer for the group regarding the medical query:\n"
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

        synthesis_prompt += "\n--- Group Report ---\n"

        final_report, final_report_log = self.lead_agent.chat(
            prompt=synthesis_prompt,
            image_path=self.question_context.get('image_path'),
            response_format={"type": "json_object"},
        )

        self._log_interaction(
            f"Lead ({self.lead_agent.agent_id}) generated final group report.",
            data={
                "step": "3_lead_synthesis",
                "agent_id": self.lead_agent.agent_id,
                "agent_role": self.lead_agent.role,
                "prompt": synthesis_prompt,
                "response": final_report,
                "llm_log": final_report_log
            }
        )

        return final_report, self.internal_log

# --- MDAgents Framework Class ---

class MDAgentsFramework:
    """
    Orchestrates the MDAgents workflow: complexity check, recruitment,
    and query processing based on complexity.
    """

    def __init__(self,
                 log_dir: str,
                 dataset_name: str,
                 model_config: Dict[str, str], # Keys like 'moderator', 'recruiter', 'default_agent'
                 max_rounds_intermediate: int = DEFAULT_MAX_ROUNDS_INTERMEDIATE,
                 max_turns_intermediate: int = DEFAULT_MAX_TURNS_INTERMEDIATE,
                 num_experts_intermediate: int = DEFAULT_NUM_EXPERTS_INTERMEDIATE,
                 num_teams_advanced: int = DEFAULT_NUM_TEAMS_ADVANCED,
                 num_agents_per_team_advanced: int = DEFAULT_NUM_AGENTS_PER_TEAM_ADVANCED):
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.model_config = model_config
        self.max_rounds_intermediate = max_rounds_intermediate
        self.max_turns_intermediate = max_turns_intermediate
        self.num_experts_intermediate = num_experts_intermediate
        self.num_teams_advanced = num_teams_advanced
        self.num_agents_per_team_advanced = num_agents_per_team_advanced

        os.makedirs(self.log_dir, exist_ok=True)

        # --- Initialize Core Agents ---
        self.moderator_agent = BaseAgent(
            agent_id="moderator",
            role=AgentRole.MODERATOR,
            model_key=model_config.get('moderator', DEFAULT_MODERATOR_MODEL),
            instruction="You are a medical expert who conducts initial assessment. Your job is to decide the difficulty/complexity of the medical query based on the provided definitions. Respond in JSON format."
        )

        self.recruiter_agent = BaseAgent(
            agent_id="recruiter",
            role=AgentRole.RECRUITER,
            model_key=model_config.get('recruiter', DEFAULT_RECRUITER_MODEL),
            instruction="You are an experienced medical expert who recruits appropriate specialists based on the medical query and its complexity level. Respond in JSON format."
        )

        self.decision_maker_agent = BaseAgent(
            agent_id="final_decision_maker",
            role=AgentRole.DECISION_MAKER,
            model_key=model_config.get('moderator', DEFAULT_MODERATOR_MODEL),
            instruction="You are a final medical decision maker. Review all provided information (opinions, reports, discussions) and make the final, consolidated answer to the original medical query. Respond in JSON format."
        )

        print("MDAgentsFramework Initialized.")
        print(f" - Log Directory: {self.log_dir}")
        print(f" - Dataset: {self.dataset_name}")
        print(f" - Models: Moderator={self.moderator_agent.model_key}, Recruiter={self.recruiter_agent.model_key}, DefaultAgent={model_config.get('default_agent', DEFAULT_AGENT_MODEL)}")
        print(f" - Intermediate Settings: Max Rounds={self.max_rounds_intermediate}, Max Turns={self.max_turns_intermediate}, Experts={self.num_experts_intermediate}")
        print(f" - Advanced Settings: Teams={self.num_teams_advanced}, Agents Per Team={self.num_agents_per_team_advanced}")

    # MODIFICATION START: Method now returns complexity and a detailed log.
    def _determine_complexity(self, question: str, options: Optional[Dict] = None, image_path: Optional[str] = None) -> (ComplexityLevel, Dict[str, Any]):
    # MODIFICATION END
        """
        Uses the Moderator agent to classify the query complexity.
        Returns:
            A tuple containing The determined ComplexityLevel and a log dictionary.
        """
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

        # MODIFICATION START: Capture the full log from the chat call
        response, llm_log = self.moderator_agent.chat(
            prompt=prompt,
            image_path=None,
            response_format={"type": "json_object"},
        )
        # MODIFICATION END

        # MODIFICATION START: Create a structured log for this step
        step_log = {
            "step_name": "determine_complexity",
            "prompt": prompt,
            "llm_log": llm_log
        }
        # MODIFICATION END

        try:
            response_clean = preprocess_response_string(response)
            response_json = json.loads(response_clean)
            complexity_str = response_json.get("complexity", "").lower()
            
            step_log["parsed_response"] = response_json # Log the parsed JSON

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
            # MODIFICATION START: Log the error
            step_log["error"] = f"Error parsing response: {e}. Raw response: {response}"
            step_log["warning"] = "Defaulted to INTERMEDIATE due to parsing error."
            # MODIFICATION END
            return ComplexityLevel.INTERMEDIATE, step_log

    # MODIFICATION START: Method now returns recruitment results and a detailed log.
    def _recruit_experts(self,
                       question: str,
                       options: Optional[Dict],
                       complexity: ComplexityLevel,
                       image_path: Optional[str] = None) -> (Union[List[Dict], List[Dict[str, Any]]], Dict[str, Any]):
    # MODIFICATION END
        """
        Uses the Recruiter agent to identify necessary experts or teams.
        Returns:
            A tuple containing:
            - The list of recruited experts or teams.
            - A detailed log dictionary for this step.
        """
        print("\n--- Recruiting Experts/Teams ---")
        self.recruiter_agent.clear_memory()

        # MODIFICATION START: Initialize step log
        step_log = {"step_name": "recruit_experts", "complexity": complexity.value}
        # MODIFICATION END

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

        prompt = "" # Will be defined inside if/elif blocks
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
        
        # MODIFICATION START: Capture response and log from the call
        recruitment_response, llm_log = self.recruiter_agent.chat(
            prompt=prompt,
            image_path=None,
            response_format={"type": "json_object"},
        )
        print(f"Recruiter Response ({complexity.value}):\n{recruitment_response}")
        
        step_log["prompt"] = prompt
        step_log["llm_log"] = llm_log
        # MODIFICATION END

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

            print(f"Recruited Teams: {[t['goal'] for t in teams]}")
            step_log["recruited_personnel"] = teams
            return teams, step_log
        return [], step_log # Should not be reached

    def _process_basic_query(self, data_item: Dict) -> Dict:
        print("\n--- Processing Basic Query ---")
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)
        agent = BaseAgent(
            agent_id="basic_solver",
            role=AgentRole.GENERAL_DOCTOR,
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
            main_prompt += "\nProvide your answer as a JSON object with 'answer' (letter for multiple-choice) and 'explanation' fields."
        else:
            main_prompt += "\nProvide your answer as a JSON object with 'answer' and 'explanation' fields."

        response, llm_log = agent.chat(
            prompt=main_prompt,
            image_path=data_item.get('image_path'),
            response_format={"type": "json_object"},
        )
        
        # MODIFICATION START: Create a detailed log for this process
        detailed_log = {
            "step_name": "process_basic_query",
            "agent_id": agent.agent_id,
            "agent_role": agent.role,
            "prompt": main_prompt,
            "llm_log": llm_log
        }
        # MODIFICATION END

        try:
            response_clean = preprocess_response_string(response)
            response_json = json.loads(response_clean)
            predicted_answer = response_json.get("answer", "")
            explanation = response_json.get("explanation", "No explanation provided.")
            detailed_log["parsed_response"] = response_json

            if options and isinstance(predicted_answer, str):
                if predicted_answer.startswith('(') and predicted_answer.endswith(')'):
                    predicted_answer = predicted_answer[1:-1].strip()
                elif predicted_answer.startswith('(') and len(predicted_answer) > 2 and predicted_answer[1].isalpha() and predicted_answer[2] == ')':
                    predicted_answer = predicted_answer[1]
                elif len(predicted_answer) > 1 and predicted_answer[0].isalpha() and (predicted_answer[1] == '.' or predicted_answer[1] == ')'):
                    predicted_answer = predicted_answer[0]
        except Exception as e:
            print(f"Error parsing basic response: {e}. Raw response: {response}")
            predicted_answer = "Could not parse answer."
            explanation = "Error parsing model response."
            detailed_log["error"] = f"Error parsing response: {e}"

        print(f"Basic Query Result: Answer='{predicted_answer}', Explanation='{explanation[:100]}...'")

        return {
            "predicted_answer": predicted_answer,
            "explanation": explanation,
            "complexity": ComplexityLevel.BASIC.value,
            # MODIFICATION START: Add the detailed log to the result
            "detailed_log": detailed_log
            # MODIFICATION END
        }

    def _process_intermediate_query(self, data_item: Dict, expert_configs: List[Dict]) -> Dict:
        print("\n--- Processing Intermediate Query ---")
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)
        
        # MODIFICATION START: Initialize a more structured log
        detailed_log = {
            "step_name": "process_intermediate_query",
            "expert_configs": expert_configs,
            "initial_opinions": [],
            "final_synthesis": {}
        }
        # MODIFICATION END

        agents = []
        for i, config in enumerate(expert_configs):
            agent_id = f"expert_{i+1}_{config['role'].replace(' ','_').lower()}"
            agent = BaseAgent(
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
        for agent in agents:
            agent.clear_memory()
            prompt = (
                f"{question_context}\n"
                f"Based on your expertise as a {agent.role}, provide your initial analysis and answer.\n"
                f"Respond with a JSON object containing 'answer' and 'explanation' fields."
            )
            response, llm_log = agent.chat(
                prompt=prompt,
                image_path=image_path,
                response_format={"type": "json_object"},
            )

            # MODIFICATION START: Create a detailed log for each expert's opinion
            opinion_log = {
                "agent_id": agent.agent_id,
                "agent_role": agent.role,
                "prompt": prompt,
                "llm_log": llm_log,
            }
            # MODIFICATION END
            
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
            except Exception as e:
                print(f"Error parsing initial opinion from {agent.agent_id}: {e}. Raw response: {response}")
                ans = "Could not parse answer."
                expl = "Error parsing model response."
                opinion_log["error"] = f"Error parsing response: {e}"

            initial_report_parts.append(f"Expert {agent.role} ({agent.agent_id}):\nAnswer: {ans}\nExplanation: {expl}\n---")
            print(f"Agent {agent.agent_id} ({agent.role}) Initial Answer: {ans}")
            
            # MODIFICATION START: Add the detailed opinion log to the main log
            detailed_log["initial_opinions"].append(opinion_log)
            # MODIFICATION END

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

        final_response, final_llm_log = self.decision_maker_agent.chat(
            prompt=synthesis_prompt,
            response_format={"type": "json_object"},
        )
        
        # MODIFICATION START: Log the final synthesis step
        detailed_log["final_synthesis"] = {
            "agent_id": self.decision_maker_agent.agent_id,
            "agent_role": self.decision_maker_agent.role,
            "prompt": synthesis_prompt,
            "llm_log": final_llm_log
        }
        # MODIFICATION END

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

        return {
            "predicted_answer": final_answer,
            "explanation": final_explanation,
            # MODIFICATION START: Rename for clarity and add the new structured log
            "detailed_log": detailed_log,
            # MODIFICATION END
            "complexity": ComplexityLevel.INTERMEDIATE.value,
        }

    def _process_advanced_query(self, data_item: Dict, team_configs: List[Dict]) -> Dict:
        print("\n--- Processing Advanced Query ---")
        agent_model_key = self.model_config.get('default_agent', DEFAULT_AGENT_MODEL)
        
        # MODIFICATION START: Initialize a structured log for the advanced process
        detailed_log = {
            "step_name": "process_advanced_query",
            "team_configs": team_configs,
            "team_discussions": [],
            "final_decision_synthesis": {}
        }
        # MODIFICATION END

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
                agent = BaseAgent(
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

        # Process teams in a defined order: IAT, then others, then FRDT
        ordered_groups = sorted(groups, key=lambda g: 0 if "initial" in g.goal.lower() else (2 if "final" in g.goal.lower() else 1))

        for group in ordered_groups:
            print(f"\n-- Processing Team: {group.group_id} ({group.goal}) --")
            # MODIFICATION START: Capture both report and the detailed discussion log
            raw_report, discussion_log = group.perform_internal_discussion()
            
            # Add discussion log to the main detailed log
            detailed_log["team_discussions"].append({
                "group_id": group.group_id,
                "goal": group.goal,
                "discussion_log": discussion_log
            })
            # MODIFICATION END

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
            f"Synthesize all this information into one final, definitive answer and explanation.\n"
            f"Respond with a JSON object containing 'answer' (letter for multiple-choice) and 'explanation' fields."
        )

        final_response, final_llm_log = self.decision_maker_agent.chat(
            prompt=synthesis_prompt,
            response_format={"type": "json_object"},
        )
        
        # MODIFICATION START: Log the final synthesis step for the advanced query
        detailed_log["final_decision_synthesis"] = {
            "agent_id": self.decision_maker_agent.agent_id,
            "agent_role": self.decision_maker_agent.role,
            "prompt": synthesis_prompt,
            "llm_log": final_llm_log
        }
        # MODIFICATION END

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

        return {
            "predicted_answer": final_answer,
            "explanation": final_explanation,
            # MODIFICATION START: Add the detailed log
            "detailed_log": detailed_log,
            # MODIFICATION END
            "complexity": ComplexityLevel.ADVANCED.value,
        }

    def run_query(self, data_item: Dict) -> Dict:
        qid = data_item["qid"]
        print(f"\n{'='*20} Processing QID: {qid} {'='*20}")
        start_time = time.time()

        # MODIFICATION START: Initialize a list to hold all sequential log steps
        process_log = []
        # MODIFICATION END

        result_data = {}
        try:
            # 1. Determine Complexity
            complexity, complexity_log = self._determine_complexity(data_item["question"], data_item.get("options"), data_item.get("image_path"))
            process_log.append(complexity_log)

            # 2. Recruit and Process
            if complexity == ComplexityLevel.BASIC:
                result_data = self._process_basic_query(data_item)
                process_log.append(result_data.get("detailed_log", {}))

            elif complexity == ComplexityLevel.INTERMEDIATE:
                expert_configs, recruitment_log = self._recruit_experts(data_item["question"], data_item.get("options"), complexity, data_item.get("image_path"))
                process_log.append(recruitment_log)
                result_data = self._process_intermediate_query(data_item, expert_configs)
                process_log.append(result_data.get("detailed_log", {}))

            elif complexity == ComplexityLevel.ADVANCED:
                team_configs, recruitment_log = self._recruit_experts(data_item["question"], data_item.get("options"), complexity, data_item.get("image_path"))
                process_log.append(recruitment_log)
                result_data = self._process_advanced_query(data_item, team_configs)
                process_log.append(result_data.get("detailed_log", {}))

        except Exception as e:
            print(f"ERROR processing QID {qid}: {e}")
            result_data = {
                "predicted_answer": "Error occurred during processing",
                "explanation": f"Error: {str(e)}",
                "complexity": "unknown"
            }
            process_log.append({"fatal_error": str(e)})

        processing_time = time.time() - start_time
        print(f"Finished QID: {qid}. Time: {processing_time:.2f}s")

        # 3. Assemble final result object
        final_result = {
            "qid": qid,
            "timestamp": int(time.time()),
            "question": data_item["question"],
            "options": data_item.get("options"),
            "image_path": data_item.get("image_path"),
            "ground_truth": data_item.get("answer"),
            "complexity_level": result_data.get("complexity", "unknown"),
            "predicted_answer": result_data.get("predicted_answer", "Error"),
            "explanation": result_data.get("explanation", "N/A"),
            "processing_time_seconds": processing_time,
            # MODIFICATION START: Add the comprehensive, sequential process log
            "process_log": process_log
            # MODIFICATION END
        }
        return final_result

    def run_dataset(self, data: List[Dict]):
        print(f"\nStarting MDAgents processing for {len(data)} items in dataset '{self.dataset_name}'.")

        for item in tqdm(data, desc=f"Running MDAgents on {self.dataset_name}"):
            qid = item.get("qid", "unknown_qid")
            result_path = os.path.join(self.log_dir, f"{qid}-result.json")
            if os.path.exists(result_path):
                print(f"Skipping {qid} - result file already exists.")
                continue

            try:
                result = self.run_query(item)
                save_json(result, result_path)
            except Exception as e:
                print(f"FATAL ERROR during run_query for QID {qid}: {e}")
                # Save an error log file
                error_result = {"qid": qid, "error": f"A fatal error occurred: {str(e)}"}
                save_json(error_result, os.path.join(self.log_dir, f"{qid}-error.json"))

        print(f"Finished processing dataset '{self.dataset_name}'. Results saved in {self.log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run MDAgents Framework on medical datasets")
    parser.add_argument("--dataset", type=str,required = True, help="Specify dataset name (e.g., vqa_rad, pathvqa, medqa)")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], default="mc", help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--moderator_model", type=str, required=True,default=DEFAULT_MODERATOR_MODEL, help="Model key for the Moderator agent")
    parser.add_argument("--recruiter_model", type=str, required=True,default=DEFAULT_RECRUITER_MODEL, help="Model key for the Recruiter agent")
    parser.add_argument("--agent_model", type=str, required=True,default=DEFAULT_AGENT_MODEL, help="Default model key for solver agents")
    parser.add_argument("--num", type=int, required=True, help="Number of samples to run")

    # Advanced settings
    parser.add_argument("--num_experts", type=int, default=DEFAULT_NUM_EXPERTS_INTERMEDIATE, help="Number of experts for intermediate complexity")
    parser.add_argument("--num_teams", type=int, default=DEFAULT_NUM_TEAMS_ADVANCED, help="Number of teams for advanced complexity")
    parser.add_argument("--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS_INTERMEDIATE, help="Maximum discussion rounds for intermediate complexity")

    args = parser.parse_args()
    time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    method_name = f"MDAgents_{time_stamp}"  # Identify the method

    # Format paths
    data_path = f"./my_datasets/processed/medqa/{args.dataset}/medqa_{args.qa_type}_test.json"

    # Create logs directory structure consistent with ColaCare
    logs_dir = os.path.join("./logs", "medqa", args.dataset,
                           "multiple_choice" if args.qa_type == "mc" else "free-form",
                           method_name)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Using Log Directory: {logs_dir}")

    # Load the main dataset
    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return

    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Configure models
    model_config = {
        "moderator": args.moderator_model,
        "recruiter": args.recruiter_model,
        "default_agent": args.agent_model
    }

    # Initialize MDAgents Framework
    framework = MDAgentsFramework(
        log_dir=logs_dir,
        dataset_name=args.dataset,
        model_config=model_config,
        num_experts_intermediate=args.num_experts,
        num_teams_advanced=args.num_teams,
        max_rounds_intermediate=args.max_rounds
    )

    # Run the framework on the dataset
    framework.run_dataset(data[:args.num])


if __name__ == "__main__":
    main()