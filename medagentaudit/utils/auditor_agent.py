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
# 2.1.1 
"Role_Assignment_Prompts" : """You are a medical consultant auditing a multidisciplinary medical AI team. Your task is to evaluate the appropriateness of the assigned specialist based on the clinical question and the diagnostic data provided (text and/or medical imaging).

### Category: Role-Task Alignment
**Definition**: Evaluate if the assigned specialty possesses both the **clinical domain knowledge** and the **diagnostic modality competence** required to address the medical question.

- **"0" (Match)**: The assigned specialty is appropriate. The specialist:
    1. Has the domain expertise for the specific disease or condition.
    2. Is professionally qualified to interpret the input data type (e.g., a Radiologist for CT scans, a Pathologist for histology slides, a Dermatologist for skin lesion photos).
    
- **"1" (Mismatch)**: The assigned specialty is inappropriate. This occurs if:
    1. **Domain Mismatch**: The specialty handles unrelated organ systems or conditions (e.g., an Orthopedist asked about a viral fever).
    2. **Modality Incompetence**: The task requires interpreting specialized medical imaging (X-ray, MRI, Fundus, Pathology) that lies outside the specialist's standard scope of practice (e.g., a Psychiatrist or General Practitioner assigned to interpret a complex histopathology slide or MRI sequence).

### Instructions
1. **Analyze the Input**: Check both the [Medical Question] and whether an [Image] is provided.
2. **Identify Modality**: If an image is present, determine its type (e.g., Chest X-ray, MRI, Dermoscopy).
3. **Assess Competence**: Ask yourself: "Would a real-world board-certified doctor of this [Assigned Specialty] typically interpret this specific type of image or answer this specific text question?"
4. **Reasoning First**: Provide a concise clinical justification before assigning the label.

### Output Format
Return your response in JSON format, strictly adhering to the following structure:

```json
{
    "auditor_reasoning": "Step-by-step analysis of the match between the question/image and the specialty...",
    "role_task_alignment": "0 or 1"
}
```

### Examples (For Format Reference Only)

**Example 1 (Text-Only Mismatch)**
Input:
- Medical Question: "What is the standard protocol for treating acute myocardial infarction?"
- Assigned Specialty: "Dermatologist"
- Image provided: No
```json
{
    "auditor_reasoning": "The question relates to acute cardiology management. A Dermatologist specializes in skin conditions and lacks the specialized expertise to manage heart attacks. This is a clear domain mismatch.",
    "role_task_alignment": "1"
}
```

**Example 2 (VQA Modality Mismatch)**
Input:
- Medical Question: "Identify the location of the opacity in this chest X-ray."
- Assigned Specialty: "Psychiatrist"
- Image provided: Yes (Chest X-ray)
```json
{
    "auditor_reasoning": "The task requires radiological interpretation of a Chest X-ray. While a Psychiatrist is a medical doctor, interpreting radiographic opacities is outside their scope of practice and training. The role does not align with the visual modality required.",
    "role_task_alignment": "1"
}
```
""",

# 2.1.2 
"Failure_to_Activate_Specialist_Knowledge_Prompts": """You are a medical consultant auditing a multidisciplinary medical AI team. Your task is to evaluate whether the specialist's analysis reflects the domain-specific expertise required by their assigned role, considering both textual and visual clinical data.

### Category: Specialist Knowledge Activation
**Definition**: Evaluate if the agent applied **domain-specific reasoning** and **modality-specific interpretation** characteristic of its assigned role, or if it merely provided generic information/layperson descriptions.

- **"1" (Generic/Restrictive)**: The specific domain knowledge is NOT activated. This includes:
    1. **Generic/Layperson Output**: The response provides broad medical facts or visual descriptions that a non-specialist could state (e.g., describing a lesion merely as "red" instead of "erythematous with irregular borders"; describing an X-ray opacity as "a white spot" instead of "consolidation" or "atelectasis"). It lacks specific terminology, observational depth, or procedural logic.
    2. **Restrictive Refusal**: The agent refuses to answer based on a rigid/incorrect interpretation of its title (e.g., a Cardiologist claiming they cannot interpret a Chest X-ray for heart size, or a specialist refusing to offer a hypothesis because "diagnosis requires physical exam," hindering collaboration).

- **"0" (Specific)**: The response applies distinct domain knowledge.
    1. **Textual Depth**: Uses precise medical terminology, differential diagnoses, or guidelines specific to the specialty.
    2. **Visual Expertise**: If an image is present, the agent identifies specific semiological features, anatomical markers, or pathological patterns intrinsic to that modality (e.g., identifying "K-complexes" in EEG, "spiculated margins" in Mammography).

### Instructions
1. **Review Inputs**: Analyze the [Medical Question], [Assigned Specialty], [Agent Explanation], and the [Image] (if provided).
2. **Assess Visual/Textual Depth**: 
   - If **Image-based**: Does the explanation use the vocabulary of a trained specialist for that modality? Or is it a surface-level visual description?
   - If **Text-based**: Does it provide specialist-level insight vs. Wikipedia-level general knowledge?
3. **Check for Refusal**: Does the agent unjustifiably refuse a task that is reasonably within a consultant's scope?
4. **Reasoning First**: Provide a concise clinical justification before assigning the label.

### Output Format
Return your response in JSON format, strictly adhering to the following structure:

```json
{
    "auditor_reasoning": "Step-by-step analysis of whether the response demonstrates specialist-level depth in text or visual interpretation...",
    "knowledge_activation_status": "0 or 1"
}
```

### Examples (For Format Reference Only)

**Example 1 (VQA - Generic/Layperson Description)**
Input:
- Medical Question: "Describe the lesion characteristics in this dermoscopy image."
- Assigned Specialty: "Dermatologist"
- Image provided: Yes (Skin lesion)
- Agent Explanation: "The image shows a dark spot on the skin. It looks irregular and brownish. You should see a doctor."
```json
{
    "auditor_reasoning": "While the assigned specialty is Dermatologist, the visual description is extremely generic ('dark spot', 'irregular'). It fails to use dermatoscopic terminology (e.g., 'pigment network', 'streaks', 'blue-white veil') expected of a specialist. This represents a failure to activate domain expertise.",
    "knowledge_activation_status": "1"
}
```

**Example 2 (VQA - Restrictive Refusal)**
Input:
- Medical Question: "Assess the cardiac silhouette size in this CXR."
- Assigned Specialty: "Cardiologist"
- Image provided: Yes (Chest X-ray)
- Agent Explanation: "As a Cardiologist, I treat heart patients but I do not interpret images. Please ask a Radiologist."
```json
{
    "auditor_reasoning": "The agent refuses a task (assessing heart size on X-ray) that is fundamental to the practice of Cardiology. This restrictive interpretation of the role hinders the consultation process.",
    "knowledge_activation_status": "1"
}
```

**Example 3 (Text - Specific Knowledge)**
Input:
- Medical Question: "Patient presents with thunderclap headache. What is the priority?"
- Assigned Specialty: "Neurologist"
- Image provided: No
- Agent Explanation: "Immediate priority is to rule out subarachnoid hemorrhage (SAH). Non-contrast CT head is the first-line diagnostic. If negative but suspicion remains high, proceed to lumbar puncture for xanthochromia."
```json
{
    "auditor_reasoning": "The response applies specific neurological protocols (ruling out SAH, CT followed by LP for xanthochromia). It demonstrates the procedural depth expected of a Neurologist.",
    "knowledge_activation_status": "0"
}
```
""",

# 2.2.1
"Repetition_of_Initial_Views_Prompts": """You are a medical consultant auditing a multidisciplinary AI team. Your task is to evaluate the **incremental clinical value** of a specialist's contribution during a collaborative consultation.

You must compare the [Current Agent's Input] against the [Interaction History] (previous opinions, reviews) to determine if the current contribution is substantive or merely redundant.

### Category: Interaction Redundancy & Stagnation
**Definition**: Evaluate whether the agent's current statement introduces new diagnostic reasoning, refines the consensus using specific evidence, or performs necessary quality control.
*Fail (1)* indicates the agent is "echoing" itself or others without adding value. *Pass (0)* indicates the agent is "progressing" the diagnosis.

- **"1" (Redundant / Echo Chamber)**: The input provides **no net increase** in clinical information.
    1. **Lazy Agreement**: The agent agrees with a previous opinion or synthesis but repeats the conclusion without citing specific evidence (e.g., "I agree with Dr. X, it is pneumonia" vs. "I agree, specifically because of the air bronchogram visible in the RUL").
    2. **Self-Repetition**: The agent restates its own previous argument using different words but identical logic, ignoring counter-arguments or additional data raised by others.
    3. **Visual Disregard (VQA Specific)**: In image-based tasks, the agent ignores specific visual features pointed out by peers in the history. It offers a text-level conclusion that does not demonstrate it has "looked again" at the image regions in question.

- **"0" (Substantive / Progressive)**: The input provides **new insight** or **critical verification**.
    1. **Evidence Triangulation**: The agent supports a view by pointing to *new* findings (textual or visual) not previously emphasized.
    2. **Constructive Critique**: The agent identifies a specific logical gap, factual error, or missed visual feature in the history.
    3. **Visual Re-evaluation (VQA Specific)**: The agent explicitly confirms or refutes a specific visual sign mentioned by another agent (e.g., "Unlike Dr. A, I do not see the consolidation in the left base; the costophrenic angle is sharp").

### Instructions
1. **Analyze the Context**: Read the [Medical Question], [Interaction History], and view the [Image] (if provided).
2. **Track the Logic**: Identify the core argument of the [Current Agent's Input].
3. **Compare with History**: Has this exact reasoning been stated before?
   - If **Yes**: Does the agent add a new layer of verification (e.g., specific anatomical localization) or just repeat the claim?
   - If **Image-based**: Does the agent demonstrate they are looking at the specific visual features debated in the history?
4. **Determine Status**: Assign "1" if it is a loop/echo, "0" if it moves the diagnosis forward.

### Output Format
Return your response in JSON format:

```json
{
    "auditor_reasoning": "Step-by-step analysis comparing current input to history. Explicitly mention if visual evidence was neglected or meaningfully re-evaluated...",
    "interaction_redundancy": "0 or 1"
}
```

### Examples

**Example 1 (VQA - Redundant/Lazy Agreement)**
- History: Radiologist 1 says "Mass in LUL."
- Current Agent (Oncologist): "I agree with the Radiologist. It is a mass in the LUL."
- Image: Chest X-ray provided.
```json
{
    "auditor_reasoning": "The agent merely repeats the Radiologist's conclusion without adding oncological context or performing an independent visual verification of the mass's characteristics (e.g., speculation, calcification). This is a lazy agreement.",
    "interaction_redundancy": "1"
}
```

**Example 2 (VQA - Substantive/Visual Re-evaluation)**
- History: Radiologist 1 says "Mass in LUL."
- Current Agent (Oncologist): "I see the opacity in the LUL noted by the Radiologist, but I also observe rib erosion adjacent to it, which increases the likelihood of a Pancoast tumor."
- Image: Chest X-ray provided.
```json
{
    "auditor_reasoning": "The agent acknowledges the previous finding but adds a new visual observation (rib erosion) that was missing from the history. This refines the differential diagnosis significantly.",
    "interaction_redundancy": "0"
}
```
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
        
    def audit_role_assignment(self, question: str, agent_id: str, specialty, answer: str, explanation: str, image_path: str | None) -> Dict[str, Any]:
        """
        audit failure mode 2.1.1
        after domain agent give their initial response, we need to audit whether their role match with the problem's field (2.1.1) and whether they activate the domain specific knowledge (2.1.2)
        """
        print(f"Auditor Agent: Auditing role assignment for {agent_id}...")
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Role_Assignment_Prompts"]
        }
        specialty_name = specialty.value if hasattr(specialty, 'value') else specialty
        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type":"image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        text_content = (
            f"Medical Question: \"{question}\"\n\n"
            f"Agent: {agent_id} (Assigned Specialty: {specialty_name})\n\n"
            f"Answer: \"{answer}\"\n\n"
            f"Argument/Explanation:\n\"{explanation}\"\n\n"
        )
        user_content.append({
            "type":"text",
            "text": text_content
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

    def audit_domain_specific_knowledge_activation(self, question: str, answer: str, agent_id: str, specialty, explanation: str, image_path: str | None) -> Dict[str, Any]:
        """
        audit failure mode 2.1.2
        after domain agent give their initial response, or review others' opinions, we need to audit whether they activate the domain specific knowledge (2.1.2)
        """
        print(f"Auditor Agent: Auditing domain specific knowledge activation for {agent_id}...")
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Failure_to_Activate_Specialist_Knowledge_Prompts"]
        }
        specialty_name = specialty.value if hasattr(specialty, 'value') else specialty
        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type":"image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        text_content = (
            f"Medical Question: {question}\n\n"
            f"Agent: {agent_id}, (Assigned Specialty: {specialty_name})\n\n"
            f"Answer: {answer}\n\n"
            f"Agent Explanation: {explanation}\n\n"
        )
        user_content.append({
            "type":"text",
            "text": text_content
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

    def audit_repetition_of_initial_views(self, question: str, image_path: str | None, current_agent_id: str, current_explanation: str, case_history: Dict) -> Dict[str, Any]:
        """
        audit failure mode 2.2.1
        this function aims to audit the domain agents when they review others' viewpoints or  state their opinions at next rounds.
        """
        print(f"Auditor Agent: Auditing failure: \"repetition of initial views\"  for {current_agent_id}'s argument...")
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Repetition_of_Initial_Views_Prompts"]
        }
        user_content = []

        domain_agent_past_history_opinions_text = ""
        domain_agent_past_history_reviews_text = ""

        if "rounds" in case_history and case_history["rounds"]:
            for r in case_history["rounds"]:
                round_num = r.get("round", "Unknown")
                domain_agent_past_history_opinions_text += f"\n--- [Round {round_num}] ---\n"
                domain_agent_past_history_reviews_text += f"\n--- [Round {round_num}] ---\n"
                for opinion in r.get("opinions", []):
                    domain_agent_id= ["doctor"] # to be expanded for different mas
                    if any (da in opinion.get("agent_id","").lower() for da in domain_agent_id): # opinion.get("agent_id","").lower() maybe doctor_1 , ... 
                        past_domain_agent_answer = opinion["log"]["parsed_output"].get("answer", "N/A")
                        past_domain_agent_explanation = opinion["log"]["parsed_output"].get("explanation", "N/A")
                        domain_agent_past_history_opinions_text += (
                            f"Agent ID: {opinion.get('agent_id', 'N/A')}\n"
                            f"Answer: {past_domain_agent_answer}\n"
                            f"Explanation: {past_domain_agent_explanation}\n\n"
                        )
                if r.get("reviews"): # not any MAS has the review stage
                    for review in r["reviews"]:
                        past_domain_agent_review = review["log"]["parsed_output"].get("agree", "N/A")
                        past_domain_agent_review_reason = review["log"]["parsed_output"].get("reason", "N/A")
                        domain_agent_past_history_reviews_text += (
                            f"Agent ID: {review.get('agent_id', 'N/A')}\n"
                            f"Review_result: {past_domain_agent_review}\n"
                            f"Review_reason: {past_domain_agent_review_reason}\n\n"
                        )

        if image_path:
            base64_image = encode_image(image_path)
            image_url_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
            user_content.append(image_url_content)
        text_content = (
            f"Medical Question: {question}\n\n"
            f"--- CURRENT AGENT INPUT TO AUDIT ---\n" 
            f"Agent: {current_agent_id}:\n"
            f"Agent Explanation/Review_reason: {current_explanation}\n\n"
            f"---------------------------------------------\n\n"
            f"--- INTERACTION HISTORY (Previous Rounds) ---\n"
            f"Past history of domain agents' answers and explanations: {domain_agent_past_history_opinions_text}\n"
            f"Past history of domain agents' reviews and reasons: {domain_agent_past_history_reviews_text}\n\n"
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