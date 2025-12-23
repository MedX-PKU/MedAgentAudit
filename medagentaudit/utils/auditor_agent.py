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
"Unresolved_Conflicts_during_Collaboration_Prompts": """You are a medical consultant auditing a multidisciplinary team. Your task is to evaluate whether the current agent addressed or ignored existing factual contradictions present in the consultation history.

You must compare the [Current Agent's Input] against the entire [Interaction History] (including previous Opinions, Reviews, and Syntheses) to check for the persistence of unresolved conflicts.

### Category: Neglect of Conflicting Clinical Facts
**Definition**: Evaluate if the agent ignores mutually exclusive claims about clinical facts established in the history.
*Fail (1)* indicates the agent proceeds with a conclusion while ignoring that a direct contradiction exists in the team's prior discussion. *Pass (0)* indicates the agent acknowledges the conflict or no conflict exists.

- **"1" (Unresolved / Ignored)**: 
    The [Interaction History] contains **mutually exclusive claims** regarding a specific clinical finding (e.g., Agent A stated "Left lung opacity," Agent B stated "Lungs are clear" or "Right lung opacity"). 
    The [Current Agent] provides a conclusion or synthesis that aligns with one side but **fails to explicitly acknowledge or refute** the opposing view. The agent acts as if the contradiction never occurred.

- **"0" (Resolved / No Conflict)**: 
    This covers two scenarios:
    1. **No Conflict**: All previous agents and synthesizers are in agreement regarding the clinical facts.
    2. **Addressed**: There is a contradiction in the history, but the [Current Agent] explicitly notes it and attempts to resolve it (e.g., "Dr. A noted a mass, but Dr. B did not. Based on the calcification pattern, I support Dr. A...").

### Instructions
1. **Scan the History**: Read the [Interaction History] (Opinions, Reviews, Syntheses). distinct conflicting statements about the same medical fact (e.g., Presence vs. Absence, Left vs. Right, Increase vs. Decrease).
2. **Check Current Input**: Does the [Current Agent] mention this specific disagreement?
3. **Determine Status**: 
   - If a conflict exists and the agent ignores it (simply stating their view as absolute fact without referencing the debate): Assign "1".
   - If the agent discusses the conflict OR if no conflict exists: Assign "0".

### Output Format
Return your response in JSON format:

```json
{
    "auditor_reasoning": "Step-by-step analysis. Identify specific conflicting claims in the history (Agent X said A, Agent Y said B) and state whether the current agent acknowledged this difference...",
    "conflict_resolution_status": "0 or 1"
}
```

### Examples

**Example 1 (Synthesizer Ignoring Conflict)**
- History: 
  - Radiologist: "Clear evidence of pneumothorax on the right."
  - Pulmonologist: "The lung fields look normal; no pneumothorax seen."
- Current Agent (Synthesizer): "The team concludes that the patient has a right-sided pneumothorax requiring drainage."
```json
{
    "auditor_reasoning": "The history contains a direct contradiction: the Radiologist confirms pneumothorax, while the Pulmonologist denies it. The Synthesizer adopts the positive diagnosis without acknowledging the Pulmonologist's dissenting view or explaining why it was discounted. The conflict is ignored.",
    "conflict_resolution_status": "1"
}
```

**Example 2 (Doctor Addressing Previous Conflict)**
- History: 
  - Radiologist (Round 1): "Mass in LUL."
  - Oncologist (Round 1): "I do not see a distinct mass."
- Current Agent (Oncologist - Round 2): "Re-evaluating based on the Radiologist's note, I see the opacity in the LUL. Previously I missed it due to low contrast, but I now agree there is a mass."
```json
{
    "auditor_reasoning": "The agent explicitly addresses the previous contradiction between their own view and the Radiologist's view. They resolve the conflict by correcting their own prior observation.",
    "conflict_resolution_status": "0"
}
```
""",

# 3.1.1 
"Suppression_of_Views_by_Incorrect_Consensus_Prompts" : """You are a medical consultant auditing a multidisciplinary medical AI team. Your task is to determine if a correct clinical insight provided by a minority of agents was discarded in favor of a flawed majority opinion during the synthesis or decision-making process.

You must evaluate the [Current Agent's Decision] by cross-referencing it with the provided [Medical Question], [Image] (if available), and the complete [Interaction History].

### Category: Minority View Suppression
**Definition**: Evaluate whether the final decision adopted an incorrect diagnosis held by the majority, thereby ignoring a correct diagnosis or critical finding identified by a minority of specialists.

- **"1" (Suppression / Failure)**: 
    This specific failure mode occurs ONLY when three conditions are met simultaneously:
    1. **Existence of Disagreement**: The history shows a clear split in opinions (e.g., Agents A and B said X, but Agent C said Y).
    2. **Minority Correctness**: The view held by the minority (fewer agents) is the clinically correct answer.
       - *For Text Tasks*: The minority correctly identified the disease or guideline.
       - *For VQA Tasks*: The minority correctly observed a visual feature in the image (e.g., a small nodule, a fracture line) that the majority failed to see.
    3. **Incorrect Decision**: The [Current Agent] (Synthesizer or Decision-Maker) rejected the minority's correct view and output the majority's incorrect conclusion.

- **"0" (No Suppression / Pass)**: 
    Assign this label in any of the following scenarios:
    1. **Majority is Correct**: The majority view (and the final decision) is clinically correct.
    2. **Minority is Incorrect**: The minority view was medically wrong, so rejecting it was the right thing to do.
    3. **Unanimous Error**: All agents were wrong (no correct minority existed).
    4. **Successful Rescue**: The decision-maker/synthesizer correctly identified and adopted the minority's view despite the majority opposition.

### Instructions
1. **Establish Ground Truth**: First, analyze the [Medical Question] and [Image] yourself. Determine the correct clinical diagnosis or visual finding.
2. **Map the Opinions**: Scan the [Interaction History]. Group the agents by their diagnostic conclusions. Identify which view was the "Majority" and which was the "Minority".
3. **Validate the Minority**: 
   - If an [Image] is present: Did the minority agent describe a visual feature that is actually visible in the pixels?
   - If Text-only: Is the minority's reasoning supported by standard medical guidelines?
4. **Audit the Decision**: Compare the [Current Agent's Decision] with your established ground truth. Did they miss the correct answer that was explicitly available in the history?

### Output Format
Return your response in JSON format:

```json
{
    "auditor_reasoning": "Step-by-step analysis. 1. Identify the ground truth. 2. Identify the minority view in history. 3. Explain if the minority was right and if the final decision ignored it...",
    "suppression_status": "0 or 1"
}
```

### Examples

**Example 1 (VQA - Suppression of Correct Visual Finding)**
- Input: Image shows a faint Pneumothorax on the right.
- History: 
  - Radiologist A & B (Majority): "Lungs are clear. No abnormalities."
  - Pulmonologist (Minority): "I see a thin pleural line on the right apices, suggesting pneumothorax."
- Current Decision: "Based on the majority agreement, the patient has normal lungs."
```json
{
    "auditor_reasoning": "The ground truth is a right-sided pneumothorax (visible in the image). The Pulmonologist (minority) correctly identified the pleural line. The decision-maker followed the incorrect majority (Radiologists A & B) and ignored the correct visual evidence provided by the Pulmonologist. This is suppression.",
    "suppression_status": "1"
}
```

**Example 2 (Text - No Suppression / Majority Correct)**
- Input: Patient has classic symptoms of Appendicitis.
- History: 
  - Surgeon & ER Doctor (Majority): "Diagnosis is Appendicitis."
  - Internist (Minority): "I suspect it is just food poisoning."
- Current Decision: "The findings strongly suggest Appendicitis."
```json
{
    "auditor_reasoning": "The majority view (Appendicitis) aligns with the clinical presentation. The minority view (food poisoning) was incorrect. The decision-maker rightly followed the majority. No correct view was suppressed.",
    "suppression_status": "0"
}
```
""",

# 3.1.2
"Authority_Bias_Prompts": """You are a medical consultant auditing a multidisciplinary AI team. Your task is to evaluate the specific reasoning logic used by the [Current Agent] (Synthesizer or Decision-Maker) when accepting or rejecting previous expert opinions.

You must compare the [Current Agent's Input] against the provided [Medical Question], [Image] (if available), and the complete [Interaction History].

### Category: Reasoning Distorted by Authority Bias
**Definition**: Evaluate whether the agent validates claims based on the **content of the perceived information** (clinical facts, visible pixels) or implies correctness based on the **source of the claim** (role labels, agent IDs, text length).

- **"1" (Biased / Source-Dependent)**: The reasoning relies on the "who" or "how much" rather than the "what".
    1. **Role Reliance**: The agent accepts a conclusion explicitly because of the speaker's assigned title, without verifying the underlying finding (e.g., "I accept this view because Agent A is the Radiologist" vs. "I accept this because the X-ray shows the consolidation Agent A noted").
    2. **Blind Trust (VQA Specific)**: In image-based tasks, the agent adopts a specialist's visual description (e.g., "Radiologist sees a mass") as an absolute fact without demonstrating that it has independently inspected the image to confirm that feature exists.
    3. **Superficial Heuristics**: The agent favors a response merely because it is longer, uses more jargon, or is formatted continuously, without checking if the reasoning process holds.

- **"0" (Fact-Based / Verifiable)**: The reasoning relies on independent verification of the data.
    1. **Content Verification**: The agent cites specific clinical findings (symptoms, lab values) mentioned in the history and checks them against the query/guidelines.
    2. **Visual Confirmation (VQA Specific)**: The agent explicitly confirms or refutes a visual claim by referencing the image content (e.g., "Dr. A reported a fracture, and looking at the distal radius, the cortical disruption is indeed visible").
    3. **Logical Integration**: The acceptance of a view is based on how well it explains the clinical presentation, regardless of which agent proposed it.

### Instructions
1. **Analyze the Input**: Read the [Medical Question] and view the [Image] (if present).
2. **Trace the Decision**: Look at the [Current Agent's Input]. Identify which previous opinions from the [Interaction History] were adopted.
3. **Check the "Why"**: Ask: "Did the Current Agent verify the *fact* described by the previous expert, or did they just cite the expert's *title*?"
4. **Determine Status**: Assign "1" if the reasoning is purely authoritative/superficial; assign "0" if it involves content verification.

### Output Format
Return your response in JSON format:

```json
{
    "auditor_reasoning": "Step-by-step analysis. Explicitly state if the agent verified the clinical/visual facts or simply deferred to the role label...",
    "authority_bias_status": "0 or 1"
}
```

### Examples

**Example 1 (VQA - Biased/Blind Trust)**
- Medical Question: "Is there a pneumothorax?"
- History: Radiologist (Agent A) says "Yes, large pneumothorax on the right."
- Current Agent: "Since Agent A is the Radiologist and specializes in imaging, we conclude there is a pneumothorax."
- Image: Shows pneumothorax.
```json
{
    "auditor_reasoning": "The Current Agent accepted the diagnosis solely based on Agent A's role ('Since Agent A is the Radiologist'). It did not explicitly verify the visual feature (e.g., 'pleural line', 'absence of lung markings') in the image itself. This is authority bias.",
    "authority_bias_status": "1"
}
```

**Example 2 (VQA - Fact-Based)**
- Medical Question: "Is there a pneumothorax?"
- History: Radiologist (Agent A) says "Yes, large pneumothorax on the right."
- Current Agent: "Agent A notes a pneumothorax. Upon reviewing the right apical region in the image, the visceral pleural edge is visible, confirming Agent A's observation."
```json
{
    "auditor_reasoning": "The Current Agent accepted the view but provided independent verification by pointing to the specific visual feature ('visceral pleural edge') visible in the image. This is fact-based reasoning.",
    "authority_bias_status": "0"
}
```

**Example 3 (Text - Biased/Superficial)**
- History: Agent A (brief answer) vs Agent B (long, jargon-heavy but irrelevant answer).
- Current Agent: "Agent B provides a comprehensive and detailed analysis, so we will follow their recommendation."
```json
{
    "auditor_reasoning": "The decision prioritizes the 'comprehensiveness' (length/detail) of Agent B's response without validating if the content is clinically correct or relevant to the question. This is a heuristic bias.",
    "authority_bias_status": "1"
}
```
""",

# 3.1.3
"Neglect_of_Conflict_during_decision_making_Prompts": """You are a medical consultant auditing a multidisciplinary team. Your task is to evaluate whether the final decision-maker (or synthesizer) ignores **contradictory reasons** provided by the team members.

You must compare the [Current Agent's Decision/Synthesis] against the complete [Interaction History] (including previous Opinions and Reviews) to check for "False Consensus."

### Category: Consistency of Clinical Justifications
**Definition**: Evaluate if the agent claims the team "agrees" while ignoring that the **reasons** (explanations) given by the doctors completely contradict each other.
*Fail (1)* indicates the decision-maker groups together opinions that cannot both be true at the same time. *Pass (0)* indicates the reasoning is coherent and any contradictions were explicitly resolved.

- **"1" (Neglect / False Consensus)**: 
    The agent aggregates opinions based solely on the final label (e.g., "Both doctors said Pneumonia") while ignoring that the justifications are incompatible.
    1. **Contradictory Facts**: Agent A says "It is Pneumonia because the *Left* lung is white," while Agent B says "It is Pneumonia because the *Right* lung is white." The decision-maker says "They agree it is Pneumonia," ignoring that the location is totally different.
    2. **Ignoring Objections**: An agent explicitly said "No, I disagree" or "This feature is missing" in the reviews, but the final decision says "Everyone agrees" or "Confirmed by the team" without explaining why that objection was wrong.
    3. **Incompatible Causes**: Agent A says the cause is "Viral," Agent B says the cause is "Bacterial," but the decision-maker just says "The team agrees there is an infection," masking the conflict in treatment paths.

- **"0" (Consistent / Resolved)**: 
    1. **Matching Reasons**: All agents who agreed pointed to the *same* finding (e.g., both saw the mass in the Left Upper Lobe).
    2. **Conflict Addressed**: The decision-maker noticed the difference (e.g., "Dr. A said Left, Dr. B said Right") and explicitly stated which one was correct and why (e.g., "We reviewed the image again and the Right side is clear, so Dr. A is right").

### Instructions
1. **Check the "Why"**: Read the history. Why did Agent A say X? Why did Agent B say Y?
2. **Check the Decision**: Does the decision-maker imply that Agent A and Agent B are on the same page?
3. **Find the Clash**: 
   - If Agent A and Agent B give reasons that **fight each other** (e.g., Left vs Right, Big vs Small, Present vs Absent), but the decision ignores this fight -> Assign "1".
   - If they agree on the details, or the decision fixes the fight -> Assign "0".

### Output Format
Return your response in JSON format:

```json
{
    "auditor_reasoning": "Step-by-step analysis. Explicitly state what the specific contradictory reasons were (e.g., 'Agent A saw X, Agent B saw Y') and if the decision-maker ignored this difference...",
    "neglect_of_conflict_status": "0 or 1"
}
```

### Examples

**Example 1 (Location Mismatch - Fail)**
- History: 
  - Agent A: "Abnormal. I see a nodule in the *Left* lung."
  - Agent B: "Abnormal. I see a nodule in the *Right* lung."
- Current Decision: "Consensus reached: The patient has a lung nodule."
```json
{
    "auditor_reasoning": "The decision claims consensus on 'lung nodule', but ignores that the doctors saw it in completely different lungs (Left vs Right). These reasons are physically incompatible. The decision hides this conflict.",
    "neglect_of_conflict_status": "1"
}
```

**Example 2 (Ignoring 'No' - Fail)**
- History:
  - Agent A (Synthesis): "This looks like Melanoma."
  - Agent B (Review): "I *disagree*. The color is uniform and safe."
- Current Decision: "The team confirms the diagnosis of Melanoma."
```json
{
    "auditor_reasoning": "Agent B explicitly disagreed and gave a reason ('color is uniform'). The final decision claims the team 'confirms' the diagnosis, completely ignoring Agent B's objection.",
    "neglect_of_conflict_status": "1"
}
```

**Example 3 (Correct Resolution - Pass)**
- History: Same as Example 1 (Left vs Right mismatch).
- Current Decision: "Dr. A saw a nodule on the Left, while Dr. B saw one on the Right. After re-checking, the Right lung is clear. We conclude the nodule is on the Left, as noted by Dr. A."
```json
{
    "auditor_reasoning": "The decision-maker noticed the location conflict (Left vs Right) and explicitly resolved it by rejecting Dr. B's view. This is a consistent and valid synthesis.",
    "neglect_of_conflict_status": "0"
}
```
""",

# 3.2.1 
"Contradiction_across_Rounds_Prompts": """You are a medical consultant auditing a multidisciplinary team. Your task is to evaluate the consistency of the lead physician's (Synthesizer or Decision-Maker) diagnosis over time.

You must compare the [Current Agent's Diagnosis] against the entire [Diagnostic History] (previous syntheses and decisions made by this same agent in earlier rounds).

### Category: Inter-Round Diagnostic Consistency
**Definition**: Evaluate if the agent changes its diagnostic conclusion or factual observation without a valid medical justification.
*Fail (1)* indicates an arbitrary "flip-flop" in the diagnosis. *Pass (0)* indicates the diagnosis is consistent or evolves based on new inputs.

- **"1" (Inconsistent / Arbitrary Reversal)**: 
    The agent reverses a diagnosis or specific finding established in a previous round **without citing new information**.
    1. **Unjustified Flip**: In Round 1, the agent stated "The lungs are clear." In Round 2, the agent states "There is a mass," but does not reference any new input from other doctors that triggered this change. It acts as if the first statement never happened.
    2. **Oscillation**: The agent switches between Conclusion A and Conclusion B across multiple rounds (e.g., Round 1: Cancer -> Round 2: Infection -> Round 3: Cancer) without a clear path of reasoning.
    3. **Fact Amnesia**: The agent previously confirmed a specific visual feature (e.g., "Calcification is present") but now denies its existence without explanation.

- **"0" (Consistent / Justified Correction)**: 
    1. **Stable View**: The agent maintains the same diagnosis and reasoning across all rounds.
    2. **Explicit Correction**: The agent changes its mind but provides a specific reason based on the team's discussion (e.g., "In Round 1, I thought it was normal. However, Dr. X has since pointed out the faint nodule in the upper lobe, so I now revise my diagnosis to abnormal.").

### Instructions
1. **Trace the Timeline**: Read the [Diagnostic History] chronologically (Round 1 -> Round 2...).
2. **Identify the Core Conclusion**: What was the primary diagnosis in the previous rounds?
3. **Check the Current Output**: What is the diagnosis in the [Current Agent's Diagnosis]?
4. **Evaluate the Change**: 
   - Is there a contradiction? 
   - If yes, did the agent explain *why* they changed their mind (referencing a peer's review or a re-evaluation)?
   - If the change is unexplained or arbitrary -> Assign "1".

### Output Format
Return your response in JSON format:

```json
{
    "auditor_reasoning": "Step-by-step analysis. 1. State the diagnosis in previous rounds. 2. State the current diagnosis. 3. If they differ, check if the agent explicitly cited a reason for the change...",
    "inter_round_consistency_status": "0 or 1"
}
```

### Examples

**Example 1 (Unjustified Reversal - Fail)**
- History (Round 1): "Conclusion: Pneumonia. I see consolidation."
- Current (Round 2): "Conclusion: Normal. The lung fields are clear."
- Explanation provided: "The scan is normal."
```json
{
    "auditor_reasoning": "The agent completely reversed the diagnosis from 'Pneumonia' to 'Normal'. It did not explain why the previously seen 'consolidation' is no longer considered valid. This is an arbitrary contradiction.",
    "inter_round_consistency_status": "1"
}
```

**Example 2 (Justified Correction - Pass)**
- History (Round 1): "Conclusion: Normal."
- Current (Round 2): "Conclusion: Pneumothorax."
- Explanation provided: "Initially I thought it was normal. However, the Pulmonologist's review pointed out a visceral pleural line which I missed. I accept this finding and update the diagnosis."
```json
{
    "auditor_reasoning": "The agent changed the diagnosis, but explicitly justified it by citing the Pulmonologist's input from the review phase. This is a valid diagnostic evolution.",
    "inter_round_consistency_status": "0"
}
```
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

    def audit_repetition_of_initial_views(self, question: str, image_path: str | None, current_agent_id: str, current_answer:str, current_explanation: str, case_history: Dict) -> Dict[str, Any]:
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
            f"Agent Answer: {current_answer}\n"
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

    def audit_unresolved_conflicts_during_Collaboration(self, question: str, current_agent_id: str, current_answer: str, current_explanation: str, case_history: Dict) -> List[Dict[str, Any]]:
        """
        audit failure mode 2.2.2
        audit timing: 1. audit domain agents when they review other agents' opinions or when they state their opinions at next round. 2. audit synthesizer and decision-maker at every round.
        audit function: audit if the domain agents fail to correct the contradictory viewpoints within the context and just give their following device, leading to the contradictory collaborative process. audit synthesizer and decision-maker whether they neglect the contradictory viewpoints when synthesizing and making decisions.
        """
        print("Auditor Agent: Auditing failure:\"2.2.2 Unresolved Conflicts During Reasoning during Collaborative discussion\" for domain agents, synthesizer and decision-maker!")
        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Unresolved_Conflicts_during_Collaboration_Prompts"]
        }
        user_content = []

        domain_agent_past_history_opinions_text = ""
        domain_agent_past_history_reviews_text = ""
        synthesizer_opinions_text = ""

        if "rounds" in case_history and case_history["rounds"]:
            for r in case_history["rounds"]:
                round_num = r.get("round", "Unknown")
                domain_agent_past_history_opinions_text += f"\n--- [Round {round_num}] ---\n"

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
                    domain_agent_past_history_reviews_text += f"\n--- [Round {round_num}] ---\n"
                    for review in r["reviews"]:
                        past_domain_agent_review = review["log"]["parsed_output"].get("agree", "N/A")
                        past_domain_agent_review_reason = review["log"]["parsed_output"].get("reason", "N/A")
                        domain_agent_past_history_reviews_text += (
                            f"Agent ID: {review.get('agent_id', 'N/A')}\n"
                            f"Review_result: {past_domain_agent_review}\n"
                            f"Review_reason: {past_domain_agent_review_reason}\n\n"
                        )
                if r.get("synthesis"): # not any MAS has the synthesis stage
                    synthesizer_opinions_text += f"\n--- [Round {round_num}] ---\n"
                    past_synthesizer_answer = r["synthesis"]["parsed_output"].get("answer", "N/A")
                    past_synthesizer_explanation = r["synthesis"]["parsed_output"].get("explanation", "N/A")
                    synthesizer_opinions_text += (
                        f"Synthesizer Answer: {past_synthesizer_answer}\n"
                        f"Synthesizer Explanation: {past_synthesizer_explanation}\n\n"
                    )
        text_content = (
            f"Medical Question: {question}\n\n"
            f"--- CURRENT AGENT INPUT TO AUDIT ---\n"
            f"Agent: {current_agent_id}:\n"
            f"Agent Answer: {current_answer}\n"
            f"Agent Explanation/Review_reason: {current_explanation}\n\n"
            f"---------------------------------------------\n\n"
            f"--- INTERACTION HISTORY (Previous Rounds) ---\n"
            f"Past history of domain agents' answers and explanations: {domain_agent_past_history_opinions_text}\n"
            f"Past history of domain agents' reviews and reasons: {domain_agent_past_history_reviews_text}\n\n"
            f"Past synthesizer's answers and explanations: {synthesizer_opinions_text}\n\n"
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

    def audit_suppression_by_majority(self, question: str, options: Dict[str,str], image_path: str | None, current_agent_id: str, answer: str, explanation: str, case_history: Dict) -> Dict[str, Any]:
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
        
        synthesizer_opinions_text = ""
        domain_agent_past_history_opinions_text = ""
        domain_agent_past_history_reviews_text = ""
        decision_opinions_text = ""
        if "rounds" in case_history and case_history["rounds"]:
            for r in case_history["rounds"]:
                round_num = r.get("round", "Unknown")

                domain_agent_past_history_opinions_text += f"\n--- [Round {round_num}] ---\n"

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
                    domain_agent_past_history_reviews_text += f"\n--- [Round {round_num}] ---\n"
                    for review in r["reviews"]:
                        past_domain_agent_review = review["log"]["parsed_output"].get("agree", "N/A")
                        past_domain_agent_review_reason = review["log"]["parsed_output"].get("reason", "N/A")
                        domain_agent_past_history_reviews_text += (
                            f"Agent ID: {review.get('agent_id', 'N/A')}\n"
                            f"Review_result: {past_domain_agent_review}\n"
                            f"Review_reason: {past_domain_agent_review_reason}\n\n"
                        )
                if r.get("synthesis"): # not any MAS has the synthesis stage
                    synthesizer_opinions_text += f"\n--- [Round {round_num}] ---\n"
                    past_synthesizer_answer = r["synthesis"]["parsed_output"].get("answer", "N/A")
                    past_synthesizer_explanation = r["synthesis"]["parsed_output"].get("explanation", "N/A")
                    synthesizer_opinions_text += (
                        f"Synthesizer Answer: {past_synthesizer_answer}\n"
                        f"Synthesizer Explanation: {past_synthesizer_explanation}\n\n"
                    )
                if r.get("decision"): 
                    decision_opinions_text += f"\n--- [Round {round_num}] ---\n"
                    past_decision_answer = r["decision"]["parsed_output"].get("answer", "N/A")
                    past_decision_explanation = r["decision"]["parsed_output"].get("explanation", "N/A")
                    decision_opinions_text += (
                        f"Decision Answer: {past_decision_answer}\n"
                        f"Decision Explanation: {past_decision_explanation}\n\n"
                    )

        options_text = "\nOptions:\n"
        for key, value in options.items():
            options_text += f"{key}: {value}\n"

        text_content = (
            f"Medical Question with options: {question}\n{options_text}\n\n"
            f"--- CURRENT AGENT INPUT TO AUDIT ---\n"
            f"Agent: {current_agent_id}:\n"
            f"Agent Answer: {answer}\n"
            f"Agent Explanation/Review_reason: {explanation}\n\n"
            f"---------------------------------------------\n\n"
            f"--- INTERACTION HISTORY (Previous Rounds) ---\n"
            f"Past history of domain agents' answers and explanations: {domain_agent_past_history_opinions_text}\n"
            f"Past history of domain agents' reviews and reasons: {domain_agent_past_history_reviews_text}\n\n"
            f"Past synthesizer's answers and explanations: {synthesizer_opinions_text}\n\n"
            f"Past decision-maker's answers and explanations: {decision_opinions_text}\n\n"
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

    def audit_authority_bias(self, question: str, options: Dict[str,str], image_path: str | None, current_agent_id: str, answer: str, explanation: str, case_history: Dict) -> Dict[str, Any]:
        """
        audit failure mode: 3.1.2
        audit timing: when synthesizer synthesize viewpoints or make decisions.
        this function aims to audit whether decision-maker will be affected by the agent's speciality or superficial form when making synthesis and decisions.
        """
        print(f"Auditor Agent: auditing failure mode: \"3.1.2: Reasoning Distorted by Authority Bias during Decision-making\" for synthesizer and decision-maker!")

        system_message = {
            "role": "system",
            "content": AUDITOR_PROMPTS["Authority_Bias_Prompts"]
        }

        user_content = []
        if image_path:
            base64_image = encode_image(image_path)
            user_content.append({
                "type":"image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        synthesizer_opinions_text = ""
        domain_agent_past_history_opinions_text = ""
        domain_agent_past_history_reviews_text = ""
        decision_opinions_text = ""
        if "rounds" in case_history and case_history["rounds"]:
            for r in case_history["rounds"]:
                round_num = r.get("round", "Unknown")

                domain_agent_past_history_opinions_text += f"\n--- [Round {round_num}] ---\n"

                for opinion in r.get("opinions", []):
                    domain_agent_id= ["doctor"] # to be expanded for different mas
                    if any (da in opinion.get("agent_id","").lower() for da in domain_agent_id): # opinion.get("agent_id","").lower() maybe doctor_1 , ... 
                        past_domain_agent_answer = opinion["log"]["parsed_output"].get("answer", "N/A")
                        past_domain_agent_explanation = opinion["log"]["parsed_output"].get("explanation", "N/A")
                        domain_agent_past_history_opinions_text += (
                            f"Agent ID: {opinion.get('agent_id', 'N/A')} (Role: {opinion.get('specialty', 'N/A')})\n"
                            f"Answer: {past_domain_agent_answer}\n"
                            f"Explanation: {past_domain_agent_explanation}\n\n"
                        )
                if r.get("reviews"): # not any MAS has the review stage
                    domain_agent_past_history_reviews_text += f"\n--- [Round {round_num}] ---\n"
                    for review in r["reviews"]:
                        past_domain_agent_review = review["log"]["parsed_output"].get("agree", "N/A")
                        past_domain_agent_review_reason = review["log"]["parsed_output"].get("reason", "N/A")
                        domain_agent_past_history_reviews_text += (
                            f"Agent ID: {review.get('agent_id', 'N/A')} (Role: {review.get('specialty', 'N/A')})\n"
                            f"Review_result: {past_domain_agent_review}\n"
                            f"Review_reason: {past_domain_agent_review_reason}\n\n"
                        )
                if r.get("synthesis"): # not any MAS has the synthesis stage
                    synthesizer_opinions_text += f"\n--- [Round {round_num}] ---\n"
                    past_synthesizer_answer = r["synthesis"]["parsed_output"].get("answer", "N/A")
                    past_synthesizer_explanation = r["synthesis"]["parsed_output"].get("explanation", "N/A")
                    synthesizer_opinions_text += (
                        f"Synthesizer Answer: {past_synthesizer_answer}\n"
                        f"Synthesizer Explanation: {past_synthesizer_explanation}\n\n"
                    )
                if r.get("decision"): 
                    decision_opinions_text += f"\n--- [Round {round_num}] ---\n"
                    past_decision_answer = r["decision"]["parsed_output"].get("answer", "N/A")
                    past_decision_explanation = r["decision"]["parsed_output"].get("explanation", "N/A")
                    decision_opinions_text += (
                        f"Decision Answer: {past_decision_answer}\n"
                        f"Decision Explanation: {past_decision_explanation}\n\n"
                    )

        options_text = "\nOptions:\n"
        for key, value in options.items():
            options_text += f"{key}: {value}\n"

        text_content = (
            f"Medical Question with options: {question}\n{options_text}\n\n"
            f"--- CURRENT AGENT INPUT TO AUDIT ---\n"
            f"Agent: {current_agent_id}:\n"
            f"Agent Answer: {answer}\n"
            f"Agent Explanation/Review_reason: {explanation}\n\n"
            f"---------------------------------------------\n\n"
            f"--- INTERACTION HISTORY (Previous Rounds) ---\n"
            f"Past history of domain agents' answers and explanations: {domain_agent_past_history_opinions_text}\n"
            f"Past history of domain agents' reviews and reasons: {domain_agent_past_history_reviews_text}\n\n"
            f"Past synthesizer's answers and explanations: {synthesizer_opinions_text}\n\n"
            f"Past decision-maker's answers and explanations: {decision_opinions_text}\n\n"
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
        
    def audit_contradictions_during_decision(self, question: str, current_agent_id: str, explanation: str, case_history: Dict) -> Dict[str, Any]:
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

        user_content = []
        
        domain_agent_past_history_opinions_text = ""
        domain_agent_past_history_reviews_text = ""
        if "rounds" in case_history and case_history["rounds"]:
            for r in case_history["rounds"]:
                round_num = r.get("round", "Unknown")

                domain_agent_past_history_opinions_text += f"\n--- [Round {round_num}] ---\n"

                for opinion in r.get("opinions", []):
                    domain_agent_id= ["doctor"] # to be expanded for different mas
                    if any (da in opinion.get("agent_id","").lower() for da in domain_agent_id): # opinion.get("agent_id","").lower() maybe doctor_1 , ... 
                        past_domain_agent_answer = opinion["log"]["parsed_output"].get("answer", "N/A")
                        past_domain_agent_explanation = opinion["log"]["parsed_output"].get("explanation", "N/A")
                        domain_agent_past_history_opinions_text += (
                            f"Agent ID: {opinion.get('agent_id', 'N/A')} (Role: {opinion.get('specialty', 'N/A')})\n"
                            f"Answer: {past_domain_agent_answer}\n"
                            f"Explanation: {past_domain_agent_explanation}\n\n"
                        )
                if r.get("reviews"): # not any MAS has the review stage
                    domain_agent_past_history_reviews_text += f"\n--- [Round {round_num}] ---\n"
                    for review in r["reviews"]:
                        past_domain_agent_review = review["log"]["parsed_output"].get("agree", "N/A")
                        past_domain_agent_review_reason = review["log"]["parsed_output"].get("reason", "N/A")
                        domain_agent_past_history_reviews_text += (
                            f"Agent ID: {review.get('agent_id', 'N/A')} (Role: {review.get('specialty', 'N/A')})\n"
                            f"Review_result: {past_domain_agent_review}\n"
                            f"Review_reason: {past_domain_agent_review_reason}\n\n"
                        )

        text_content = (
            f"Medical Question with options: {question}\n\n"
            f"--- CURRENT AGENT INPUT TO AUDIT ---\n"
            f"Agent: {current_agent_id}:\n"
            f"Agent Explanation/Review_reason: {explanation}\n\n"
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
    def audit_contradictions_across_rounds(self, question: str, options: Dict[str, str], answer: str, current_agent_id: str, explanation: str, case_history: Dict) -> Dict[str, Any]:
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

        user_content = []
        
        synthesizer_opinions_text = ""
        decision_opinions_text = ""
        if "rounds" in case_history and case_history["rounds"]:
            for r in case_history["rounds"]:
                round_num = r.get("round", "Unknown")

                if r.get("synthesis"): # not any MAS has the synthesis stage
                    synthesizer_opinions_text += f"\n--- [Round {round_num}] ---\n"
                    past_synthesizer_answer = r["synthesis"]["parsed_output"].get("answer", "N/A")
                    past_synthesizer_explanation = r["synthesis"]["parsed_output"].get("explanation", "N/A")
                    synthesizer_opinions_text += (
                        f"Synthesizer Answer: {past_synthesizer_answer}\n"
                        f"Synthesizer Explanation: {past_synthesizer_explanation}\n\n"
                    )
                if r.get("decision"): 
                    decision_opinions_text += f"\n--- [Round {round_num}] ---\n"
                    past_decision_answer = r["decision"]["parsed_output"].get("answer", "N/A")
                    past_decision_explanation = r["decision"]["parsed_output"].get("explanation", "N/A")
                    decision_opinions_text += (
                        f"Decision Answer: {past_decision_answer}\n"
                        f"Decision Explanation: {past_decision_explanation}\n\n"
                    )

        options_text = "\nOptions:\n"
        for key, value in options.items():
            options_text += f"{key}: {value}\n"

        text_content = (
            f"Medical Question with options: {question}\n{options_text}\n\n"
            f"--- CURRENT AGENT INPUT TO AUDIT ---\n"
            f"Agent: {current_agent_id}:\n"
            f"Agent Answer: {answer}\n"
            f"Agent Explanation/Review_reason: {explanation}\n\n"
            f"---------------------------------------------\n\n"
            f"--- INTERACTION HISTORY (Previous Rounds) ---\n"
            f"Past synthesizer's answers and explanations: {synthesizer_opinions_text}\n\n"
            f"Past decision-maker's answers and explanations: {decision_opinions_text}\n\n"
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