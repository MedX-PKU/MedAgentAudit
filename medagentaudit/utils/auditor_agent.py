from openai import OpenAI
import json
from typing import Dict, Any, List, Tuple
import time
import sys
from pathlib import Path

# Ensure project root is in path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
utils_root = current_file_path.parents[0]
sys.path.extend([str(project_root),str(utils_root)])
from config import get_config
from encode_image import encode_image
from json_utils import load_json, save_json, preprocess_response_string
AUDITOR_PROMPTS = {
# 2.1.1, 2.1.2
"Role_Assignment_and_Execution_Prompts" :  """You are a medical consultant auditing a multidisciplinary team. Your task is to evaluate the specialist's contribution using two binary categories.

### Category 1: Role-Task Alignment
**Definition**: Check if the assigned medical specialty is appropriate for the medical question.
- **"0" (Match)**: The assigned specialty has the necessary medical knowledge to address the medical question.
- **"1" (Mismatch)**: The specialty is unrelated to the medical question. The agent cannot provide relevant clinical insights and instead produces generic text or irrelevant information.

### Category 2: Specialized Knowledge Activation
**Definition**: Check if the agent uses the specific clinical approach and depth expected from its assigned specialty.
- **"0" (Activated)**: The argument shows specialized medical knowledge, expert-level interpretations, or a clinical approach unique to the assigned field. It goes beyond general medical common sense.
- **"1" (Not Activated)**: The argument is generic or homogenized. It lacks specialist-level depth and could have been written by a non-specialist. This also includes cases where the agent refuses the task despite having the underlying capability.

### Output Format
Provide a JSON object with:
1. **`role_task_alignment`**: 1 or 0.
2. **`specialist_knowledge_activation`**: 1 or 0.
3. **`auditor_reasoning`**: A concise explanation for these choices.
""", 
# 2.2.1
"Repetition_of_Initial_Views_Prompts" : """You are a medical consultant auditing a multidisciplinary team. Your task is to evaluate whether a specialist's review of a synthesized opinion introduces new information or is a redundant restatement of their starting analysis.

### Category: Interaction Redundancy
**Definition**: This failure occurs when the interaction does not introduce further information or refine the current viewpoint. The process is a restatement of the starting conclusion.
- **"1" (Redundant)**: The review is a restatement of the initial analysis. It does not introduce new information or correct previous gaps. The agent accepts the synthesis because the final answer matches their own, without evaluating the reasoning pathway. The discussion does not perform a corrective function.
- **"0" (Substantive)**: The review contributes new observations, identifies specific logical discrepancies in the synthesis, or provides a more detailed justification that was absent in the first round.

### Output Format
Provide a JSON object with:
1. **`interaction_redundancy`**: 0 or 1.
2. **`auditor_reasoning`**: A concise explanation for this choice.
""",

# 2.2.2 
"Unresolved_Conflicts_during_Collaboration_Prompts" : """You are a medical consultant auditing a multidisciplinary team. Your task is to evaluate whether the team identified and addressed contradictory statements regarding clinical facts during their discussion.

### Category: Conflict Resolution Status
**Definition**: Check if the agents ignored mutually exclusive claims about the same medical fact.
- **"1" (Unresolved)**: Different agents state opposite facts about the same specific finding (e.g., one confirms a lesion, one denies it; or they disagree on the laterality). The discussion ignores this clash. The agents proceed as if both statements can be true, or they simply list the differing views without determining which is correct.
- **"0" (Resolved / No Conflict)**: There are no factual contradictions, OR the team explicitly noticed a disagreement and settled it (e.g., by verifying the image again or debating the specific point).

### Output Format
Provide a JSON object with:
1. **`conflict_resolution_status`**: 1 or 0.
2. **`auditor_reasoning`**: A concise explanation for this choice.
""",

# 3.1.1 
"Suppression_of_Views_by_Incorrect_Consensus_Prompts" : """You are a medical consultant auditing a multidisciplinary team. Your task is to determine if a correct clinical insight from a minority perspective was discarded in favor of a flawed majority opinion.

### Category: Minority View Suppression
**Definition**: Check if the final decision followed the majority opinion despite a minority offering the correct answer supported by valid clinical findings.
- **"1" (Suppression)**: The minority opinion (held by fewer agents) provided the correct diagnosis or answer. However, the final decision adopted the incorrect view held by the majority. The correct clinical signal was lost.
- **"0" (No Suppression)**: This covers three scenarios: 
1. The majority view was correct.
2. The minority view was incorrect (so rejecting it was clinically appropriate).
3. There was unanimous agreement (no minority existed).

### Output Format
Provide a JSON object with:
1. **`suppression_status`**: 1 or 0.
2. **`auditor_reasoning`**: A concise explanation for this choice, identifying which agent held the correct view if suppression occurred.
""",

# 3.1.2
"Authority_Bias_Prompts": """You are a medical consultant auditing a multidisciplinary team. Your task is to evaluate the reasoning logic used by the decision-maker when selecting or synthesizing expert opinions.

### Category: Reasoning Distorted by Authority Bias
**Definition**: Check if the decision mechanism prioritizes the speaker's assigned identity label or the surface complexity of their text over the actual verification of medical facts.
- **"1" (Biased)**: The decision accepts a conclusion primarily because of the agent's assigned role (e.g., "We accept this view because the Radiologist is the expert on images") or the length/jargon of the response, without verifying if the described findings actually align with the medical question. The reasoning relies on the role label or format rather than the clinical argument.
- **"0" (Fact-Based)**: The decision evaluates the *validity* of the arguments provided. It integrates specific clinical findings and reasoning. The acceptance of a view is based on the strength of the clinical fact (e.g., "The diagnosis is X because the referenced imaging features A and B are present"), regardless of which agent presented it.

### Output Format
Provide a JSON object with:
1. **`authority_bias_status`**: 1 or 0.
2. **`auditor_reasoning`**: A concise explanation for this choice, noting if specific role labels were used as a substitute for evidence verification.
""",

# 3.1.3
"Neglect_of_Conflict_during_decision_making_Prompts": """You are a medical consultant auditing a multidisciplinary team. Your task is to evaluate whether the decision-making logic ignores internal contradictions between the synthesized arguments.

### Category: Reasoning Process Consistency of Synthesis
**Definition**: Check if the final analysis aggregates opinions based solely on the agreement of the final conclusion, while ignoring mutually exclusive clinical justifications.
- **"1" (Neglect)**: The decision synthesizes views that are clinically contradictory. For example, two agents support the same diagnosis but cite physically incompatible findings (e.g., "Left lung opacity" vs. "Right lung opacity"), and the synthesis ignores this spatial conflict. Or, the decision adopts a finding that contradicts the majority's observation without explaining why the majority was incorrect.
- **"0" (Consistent/Resolved)**: The synthesized reasoning is coherent. Any contradictions between the agents' observations were explicitly noted and resolved (e.g., by favoring the view with more specific anatomical details), or no contradictions existed in the source arguments.

### Output Format
Provide a JSON object with:
1. **`neglect_of_conflict_status`**: 1 or 0.
2. **`auditor_reasoning`**: A concise explanation for this choice, identifying specific contradictions that were ignored if applicable.
""",

# 3.2.1 
"Contradiction_across_Rounds_Prompts": """You are a medical consultant auditing a multidisciplinary team. Your task is to compare the current decision reasoning process against the history of previous decisions to check for self-consistency.

### Category: Inter-Round Reasoning Consistency
**Definition**: Check if the decision-maker contradicts their own previous conclusions or factual observations without new fact introduced.
- **"1" (Self-Contradictory)**: The agent reverses a specific diagnostic conclusion or visual finding established in a previous round without citing new information or a valid correction. The reasoning is unstable (e.g., asserting a pathology exists in Round 1, then describing the same region as normal in Round 2, or flipping the final answer arbitrarily).
- **"0" (Consistent)**: The reasoning aligns with previous rounds, or any change in opinion is explicitly justified by new arguments provided by the specialists. The evolution of the diagnosis follows a logical path.

### Output Format
Provide a JSON object with:
1. **`inter_round_consistency_status`**: 1 or 0.
2. **`auditor_reasoning`**: A concise explanation for this choice, citing specific rounds where the contradiction occurred if applicable.
"""
}



class BaseAgent:
    """Base class for all agents."""

    def __init__(self,
                 agent_id: str,
                 agent_type,
                 config_path,
                 model_key: str = "qwen-vl-max"):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (Doctor or Coordinator)
            model_key: LLM model to use
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.memory = []

        self.llm = get_config(config_path, active_llm=model_key).llm

        self.client = OpenAI(
            api_key=self.llm.api_key,
            base_url=self.llm.base_url,
            timeout = self.llm.timeout, # if time out then atonomously report error
        )
        self.model_name = self.llm.model_name
    # MODIFICATION START: Adjusted return type to include prompts for logging.
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
    # MODIFICATION END
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM, system message: {system_message['content'][:50]}...")
                print(f'the llm model name is {self.model_name}')
                if hasattr(self.llm, 'reasoning') and self.llm.reasoning: # for model like gpt-5.1
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[system_message, user_message],
                        response_format={"type": "json_object"},
                        extra_body={"enable_thinking": True},
                        reasoning_effort=self.llm.reasoning.effort,
                        stream=self.llm.stream,
                        timeout=self.llm.timeout # just in case timeout error!
                    )
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[system_message, user_message],
                        response_format={"type": "json_object"},
                        extra_body={"enable_thinking": True},
                        stream=self.llm.stream,
                        timeout=self.llm.timeout # just in case timeout error!
                    )
                if not self.llm.stream:
                    response = completion.choices[0].message.content
                else:
                    response_chunks = []
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            response_chunks.append(chunk.choices[0].delta.content)
                    response = "".join(response_chunks)
                # check if the response is empty
                if not response.strip():
                    raise ValueError("Empty response received from LLM")
                print(f"Agent {self.agent_id} received response: {response[:50]}...")
                return response, system_message, user_message
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    # don't return error message ,just raise exception
                    raise RuntimeError(f"CRITICAL: Agent {self.agent_id} failed after {max_retries} attempts. Reason: {str(e)}")
                time.sleep(1)

class AuditorAgent(BaseAgent):
    def __init__(self, agent_id, config_path, model_key,agent_type):
        super().__init__(agent_id, agent_type, config_path, model_key)
    def audit_role_assignment_and_execution (self, question: str, agent_id: str, specialty, explanation: str, image_path: str | None) -> Dict[str, Any]:
        """
        audit failure mode 1.1.1, 1.1.2
        after domain agent give their initial response, we need to audit whether their role match with the problem's field (2.1.1) and whether they activate the domain specific knowledge (2.1.2)
        """
        print(f"Auditor Agent: Auditing Domain Agent Contribution for {agent_id}...")
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Role_Assignment_and_Execution_Prompts"]
        }
        specialty_name = specialty.value if hasattr(specialty, 'value') else specialty

        user_message = {
            "role": "user",
            "content": f"Medical Question: \"{question}\"\n\n"
                       f"Agent: {agent_id} (Specialty: {specialty_name})\n"
                       f"Argument/Explanation:\n\"{explanation}\"\n\n"
                       f"Please provide your audit in the specified JSON format."
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError):
            return {}

    def audit_repetition_of_initial_views(self, agent_id: str, explanation: str, image_path: str | None) -> Dict[str, Any]:
        """
        audit failure mode 2.2.1
        this function aims to audit the domain agents when they review others' viewpoints or  state their opinions at next rounds.
        """
        print(f"Auditor Agent: Auditing failure: \"repetition of initial views\"  for {agent_id}'s argument...")
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Repetition_of_Initial_Views_Prompts"]
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
        user_content.append({
            "type":"text",
            "text":text_content
        })
        user_message = {
            "role": "user",
            "content": user_content
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError):
            return {}

    def audit_unresolved_conflicts_during_Collaboration(self, question: str, doctor_reviews: List[Dict[str, Any]], specialties) -> List[Dict[str, Any]]:
        """
        audit failure mode 2.2.2
        audit timing: audit domain agents when they review other agents' opinions or when they state their opinions at next round.
        audit function: audit if the domain agents fail to correct the contradictory viewpoints within the context and just give their following device, leading to the contradictory collaborative process.
        """
        print("Auditor Agent: Auditing failure:\"2.2.2 Unresolved Conflicts During Reasoning during Collaborative discussion\" for domain agents!")
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Unresolved_Conflicts_during_Collaboration_Prompts"]
        }
        # user_message arugments_text 构造和之前 audit_arguments 类似
        arguments_text = ""
        for i, review in enumerate(doctor_reviews):
            # 1. 安全地获取专科信息
            specialty_name = "N/A"
            if i < len(specialties) and specialties[i]:
                specialty = specialties[i]
                specialty_name = specialty.value if hasattr(specialty, 'value') else specialty
            
            # 2. 直接从 review 数据中获取 agent_id，而不是构造它
            #    (这要求调用方在 review 字典中提供 'agent_id' 键, 我们在第一步已完成)
            agent_id = review.get('agent_id', f'agent_{i+1}') # 提供一个备用ID

            supported_answer = review.get('current_viewpoint', review.get('answer', 'N/A'))

            # 4. 使用 review 中已有的 'reasoning' 键，而不是 'reason'
            reasoning = review.get('reason', review.get('reasoning', 'N/A'))
            
            arguments_text += f"\n---\nAgent ID: {agent_id} (Specialty: {specialty_name}):\n"
            arguments_text += f"Supported Answer: {supported_answer}\n"
            arguments_text += f"Reasoning: {reasoning}\n"
        
        user_message = { "role": "user", "content": f"Medical Question: {question}\n\nArguments:\n{arguments_text}\n\nPlease provide the overall quality audit as a JSON list." }
        
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError):
            return []

    def audit_suppression_by_majority(self, question: str, explanation: str, overall_quality_audit: List[Dict[str, Any]], image_path: str | None) -> Dict[str, Any]:
        """
        audit failure mode: 3.1.1
        audit timing: synthesizer synthesize opinions or make decisions.
        this function aims to audit if decision-maker will just adopt the opinions made by the majority without judging their reasoning process, leading to the miss of the correct minority and the correct information unit!
        """
        print("Auditor Agent: auditing failure mode : \" 3.1.1: Suppression of Correct Minority Views by Incorrect Consensus during Decision-making\" for synthesizer or decision-maker!")
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Suppression_of_Views_by_Incorrect_Consensus_Prompts"]
        }
        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type":"image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        if overall_quality_audit:
            individual_quality_text = ""
            for audit in overall_quality_audit:
                individual_quality_text += f"- Agent: {audit.get('agent_id', 'Unknown')}\n"
                individual_quality_text += f"  Quality: {audit.get('overall_quality_category', 'N/A')}\n"
                individual_quality_text += f"  Auditor Reasoning: {audit.get('auditor_reasoning', 'No reasoning provided')}" + "\n"
            text_content = (
                f"Medical Question: \"{question}\"\n\n"
                f"Argument to Evaluate:\n\"{explanation}\"\n\n"
                f"We have evaluated the quality of individual doctors' arguments as follows:\n"
                f"{individual_quality_text}\n"
                f"Please provide the overall quality audit as a JSON object."
            )
        else:
            text_content = (
                f"Medical Question: \"{question}\"\n\n"
                f"Argument to Evaluate:\n\"{explanation}\"\n\n"
                f"Please provide the overall quality audit as a JSON object."
            )
        user_content.append({
            "type":"text",
            "text": text_content
        })
        user_message = {
            "role":"user",
            "content": user_content
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            return json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, TypeError):
            return {}

    def audit_authority_bias(self, 
                                contributions: List[Dict[str, Any]],
                                context_description: str,
                                image_path: str| None) -> List[Dict[str, Any]]:
        """
        audit failure mode: 3.1.2
        audit timing: when synthesizer synthesize viewpoints or make decisions.
        this function aims to audit whether decision-maker will be affected by the agent's speciality or superficial form when making synthesis and decisions.
        """
        print(f"Auditor Agent: auditing failure mode: \"3.1.2: Reasoning Distorted by Authority Bias during Decision-making\" for synthesizer and decision-maker!")

        # 过滤掉文本为空的贡献，但不再检查数量
        valid_contributions = [c for c in contributions if c.get("text", "").strip()]
        
        # 如果没有任何有效贡献，直接返回空
        if not valid_contributions:
            return []

        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Authority_Bias_Prompts"]
        }

        context_text = f"Please analyze the following {context_description} for conflicts:\n\n"
        for contrib in valid_contributions:
            context_text += f"--- Argument from {contrib['agent_id']} ({contrib.get('specialty', 'N/A')}) ---\n"
            context_text += f"{contrib['text']}\n\n"

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
    def audit_contradictions_during_decision(self, 
                                      question: str, 
                                      doctor_opinions: List[Dict[str, Any]], 
                                      doctor_agents, 
                                      all_keus: List[Dict],
                                      image_path: str | None) -> Dict[str, bool]:
        """
        audit failure mode: 3.1.3
        audit timing: when decision-maker or synthesizer synthesize opinions or make decisions!
        this function aims to audit whether the decision-maker and synthesizer omits the contradictions winthin the context and pass the answer becauser their answers are consistent or one agent claims the abnormal findings. 
        """
        print("Auditor Agent: audit failure mode: \"3.1.3: Neglect of Contradictions in Reasoning Process during Decision-making\" for decision-maker or synthesizer")
        
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Neglect_of_Conflict_during_decision_making_Prompts"]
        }

        # 构建医生们的分析上下文
        opinions_context = "Here are the initial analyses from the specialists:\n\n"
        for i, opinion in enumerate(doctor_opinions):
            agent = doctor_agents[i]
            opinions_context += f"--- Analysis from {agent.agent_id} ({agent.specialty.value}) ---\n"
            opinions_context += f"Explanation: {opinion.get('explanation', 'N/A')}\n"
            opinions_context += f"Answer: {opinion.get('answer', 'N/A')}\n\n"

        keu_list_text = "\n".join([f"- {keu['keu_id']}: \"{keu['content']}\"" for keu in all_keus])
        user_content = []
        text_content = (
            f"**Medical Question:**\n \"{question}\"\n\n"
            f"**Doctors' Analyses:**\n{opinions_context}"
            f"**Consolidated List of All Evidential Units to Evaluate:**\n{keu_list_text}\n\n"
            f"Based on the doctors' analyses, please provide your judgment on which of these are KEY units in the specified JSON format."
        )
        user_content.append({
            "type":"text",
            "text":text_content
        })
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type":"image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        user_message = {
            "role":"user",
            "content": user_content
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            key_status_map = json.loads(preprocess_response_string(response_text))
            # 确保所有KEU都有一个布尔值
            for keu in all_keus:
                if keu['keu_id'] not in key_status_map:
                    key_status_map[keu['keu_id']] = False
            return key_status_map
        except (json.JSONDecodeError, TypeError):
            print("Auditor Agent: Error parsing KEU key status response. Defaulting all to not key.")
            return {keu['keu_id']: False for keu in all_keus}
        
    def audit_contradictions_across_rounds(self,
                                      question: str,
                                      doctor_opinions: List[Dict[str, Any]],
                                      doctor_agents,
                                      all_keus: List[Dict],
                                      image_path: str | None) -> Dict[str, bool]:
        """
        audit failure mode: 3.2.1
        audit timing: when synthesizer or decision-maker synthesize opinions or make decisions!
        this function aims to audit whether the decision-maker and synthesizer will contradict their own opinions in the subsequent rounds with no new information introduced.
        """
        print("Auditor Agent: audit failure mode: \"3.2.1: Self-Contradiction in Viewpoints Across Rounds during Decision-making\" for decision-maker or synthesizer")
        
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Contradiction_across_Rounds_Prompts"]
        }

        # 构建医生们的分析上下文
        opinions_context = "Here are the initial analyses from the specialists:\n\n"
        for i, opinion in enumerate(doctor_opinions):
            agent = doctor_agents[i]
            opinions_context += f"--- Analysis from {agent.agent_id} ({agent.specialty.value}) ---\n"
            opinions_context += f"Explanation: {opinion.get('explanation', 'N/A')}\n"
            opinions_context += f"Answer: {opinion.get('answer', 'N/A')}\n\n"

        keu_list_text = "\n".join([f"- {keu['keu_id']}: \"{keu['content']}\"" for keu in all_keus])
        user_content = []
        text_content = (
            f"**Medical Question:**\n \"{question}\"\n\n"
            f"**Doctors' Analyses:**\n{opinions_context}"
            f"**Consolidated List of All Evidential Units to Evaluate:**\n{keu_list_text}\n\n"
            f"Based on the doctors' analyses, please provide your judgment on which of these are KEY units in the specified JSON format."
        )
        user_content.append({
            "type":"text",
            "text":text_content
        })
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type":"image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        user_message = {
            "role":"user",
            "content": user_content
        }
        response_text, _, _ = self.call_llm(system_message, user_message)
        try:
            key_status_map = json.loads(preprocess_response_string(response_text))
            # 确保所有KEU都有一个布尔值
            for keu in all_keus:
                if keu['keu_id'] not in key_status_map:
                    key_status_map[keu['keu_id']] = False
            return key_status_map
        except (json.JSONDecodeError, TypeError):
            print("Auditor Agent: Error parsing KEU key status response. Defaulting all to not key.")
            return {keu['keu_id']: False for keu in all_keus}        