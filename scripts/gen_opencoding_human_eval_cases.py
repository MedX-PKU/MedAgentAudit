'''
./scripts/gen_opencoding_human_eval_cases.py
This script is designed to transform the extracted cases to structured human evaluation cases.
'''
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))
from medagentaudit.utils.json_utils import save_jsonl, load_jsonl
from medagentaudit.utils.logger import DualLogger

# Define paths
EXTRACTED_LOGS_FOR_OPENCODING_HUMAN_EVAL_DIR = project_root / "logs" / "extracted_logs_for_open_coding_human_evaluation"
EXTRACTED_LOGS_FOR_OPENCODING_HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURED_LOGS_FOR_OPENCODING_HUMAN_EVAL_DIR = project_root / "logs" / "structured_logs_for_open_coding_human_evaluation"
STRUCTURED_LOGS_FOR_OPENCODING_HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)

def gen_collaboration_text(case_history):
    '''
    This function is designed to generate the text description of the multi-agent collaboration process based on the case history.
    The generated text will be used for human evaluation to understand the multi-agent collaboration process.
    '''
    collaboration_text_start_text = (f"Here is the multi-agent collaboration process for this case:")
    collaboration_text = ""
    if "rounds" in case_history and case_history["rounds"]:
        for r in case_history["rounds"]:
            round_num = r.get("round", "Unknown")
            collaboration_text += f"# --- [Round {round_num}] --- \n\n"
            if r.get("opinions"):
                collaboration_text += (                
                    f"## Task understanding phase:\n\n"
                    f"each domain agent independently assesses the case and provides its own judgment along with the supporting rationale:\n\n"
                )
                for opinion in r.get("opinions", []):
                    domain_agent_id= opinion.get("agent_id","").lower()
                    past_domain_agent_answer = opinion["log"]["parsed_output"].get("answer", None)
                    past_domain_agent_explanation = opinion["log"]["parsed_output"].get("explanation", None)
                    role = opinion.get("specialty", None)
                    if role is not None:
                        collaboration_text += (f"### Domain agent ({domain_agent_id}, role:{role}) opinion:\n\n")
                    else:
                        collaboration_text += (f"### Domain agent ({domain_agent_id}) opinion:\n\n")
                    if past_domain_agent_answer is not None:
                        collaboration_text += (f"**Answer:** {past_domain_agent_answer}\n\n")
                    if past_domain_agent_explanation is not None:
                        collaboration_text += (f"**Explanation:** {past_domain_agent_explanation}\n\n")

            if r.get("synthesis"): # not any MAS has the synthesis stage
                collaboration_text += (                
                f"## Multi-Agent collaborative discussion phase: (meta agent's synthesis)\n\n"
                f"the meta agent synthesizes the opinions of the domain agents to form a preliminary conclusion.\n\n"
                )
                collaboration_text += (                
                    f"This stage encompasses the generation of a preliminary conclusion by the meta-agent:\n\n"
                )

                if isinstance(r["synthesis"], list):
                    for synth_item in r["synthesis"]:
                        synth_log = synth_item.get("log", {}).get("parsed_output", {})
                        past_ans = synth_log.get("answer", None)
                        past_expl = synth_log.get("explanation", None)
                        agent_id = synth_item.get("agent_id", None)
                        role = synth_item.get("specialty", None)
                        collaboration_text += (f"### Meta agent's synthesis:\n\n")
                        if agent_id is not None:
                            agent_id = agent_id.lower()
                            collaboration_text += (f"**Meta agent id: {agent_id}**\n\n")
                        if role is not None:
                            collaboration_text += (f"**Role**:{role}\n\n")
                        if past_ans is not None:
                            collaboration_text += (f"**Group lead ({agent_id}) answer:** {past_ans}\n\n")
                        if past_expl is not None:
                            collaboration_text += (f"**Group lead explanation:** {past_expl}\n\n")
                elif isinstance(r["synthesis"], dict):
                    collaboration_text += (f"### Meta agent synthesis:\n\n")
                    agent_id = r["synthesis"].get("agent_id", None)
                    if agent_id is not None:
                        collaboration_text += (f"**Meta agent id: {agent_id}**\n\n")
                    role = r["synthesis"].get("specialty", None)
                    if role is not None:
                        collaboration_text += (f"**Role**:{role}\n\n")
                    past_synthesizer_answer = r["synthesis"]["parsed_output"].get("answer", None)
                    past_synthesizer_explanation = r["synthesis"]["parsed_output"].get("explanation", None)
                    if past_synthesizer_answer is not None:
                        collaboration_text += (f"**Synthesizer answer:** {past_synthesizer_answer}\n\n")
                    if past_synthesizer_explanation is not None:
                        collaboration_text += (f"**Synthesizer explanation:** {past_synthesizer_explanation}\n\n")

            if r.get("reviews"): # not any MAS has the review stage
                collaboration_text += (
                    f"## Multi-Agent collaborative discussion phase (domain agents review):\n\n"
                    f"this stage encompasses a review from domain agents providing their perspectives and rationales. "
                    f"It includes cross-evaluation among domain agents, where they exchange viewpoints to refine the collective outcome.\n\n"
                )
                for review in r["reviews"]:
                    past_domain_agent_review = review["log"]["parsed_output"].get("agree", None)
                    past_domain_agent_review_reason = review["log"]["parsed_output"].get("reason", None)
                    past_domain_agent_review_explanation = review["log"]["parsed_output"].get("explanation", None)
                    past_domain_agent_review_answer = review["log"]["parsed_output"].get("answer", None)
                    agent_id = review.get("agent_id", None)
                    if agent_id: agent_id = agent_id.lower()
                    collaboration_text += (f"### Domain agents ({agent_id}) review:\n\n")
                    role = review.get("specialty", None)
                    if agent_id is not None:
                        collaboration_text += (f"**Agent id: {agent_id}**\n\n")
                    if role is not None:
                        collaboration_text += (f"**(role: {role})**\n\n")
                    if past_domain_agent_review is not None:
                        collaboration_text += (f"**Review result:** {past_domain_agent_review}\n\n")
                    if past_domain_agent_review_reason is not None:
                        collaboration_text += (f"**Review reason:** {past_domain_agent_review_reason}\n\n")
                    if past_domain_agent_review_explanation is not None:
                        collaboration_text += (f"**Review explanation:** {past_domain_agent_review_explanation}\n\n")
                    if past_domain_agent_review_answer is not None:
                        collaboration_text += (f"**Review answer:** {past_domain_agent_review_answer}\n\n")

            if r.get("decision"): 
                collaboration_text += (                
                    f"## Final decision-making phase: \n\n"
                    f"this stage encompasses the final decision-making process, where the meta-agent consolidates the insights from previous stages to arrive at a conclusive answer:\n\n"
                )
                collaboration_text += (f"### Meta agent makes decision:\n\n")
                agent_id = r["decision"].get("agent_id", None)
                if agent_id: 
                    agent_id = agent_id.lower()
                    collaboration_text += (f"**Agent id: {agent_id}**\n\n")
                role = r["decision"].get("specialty", None)
                if role is not None:
                    collaboration_text += (f"**Role**:{role}\n\n")
                past_decision_answer = r["decision"]["parsed_output"].get("answer", None)
                past_decision_explanation = r["decision"]["parsed_output"].get("explanation", None)
                if past_decision_answer is not None:
                    collaboration_text += (f"**Decision answer:** {past_decision_answer}\n\n")
                if past_decision_explanation is not None:
                    collaboration_text += (f"**Decision explanation:** {past_decision_explanation}\n\n")
    return collaboration_text_start_text, collaboration_text
def main():
    failure_mode_definition_mapping = {
        "1.1.1": {
            "name": "Factual Hallucinations during Input Interpretation",
            "definition": "The agent hallucinates non-existent features or contradicts objective facts present in the input (text/image).",
            "human_eval_instruction": "Compare the domain agent's observation against the Ground Truth and source input. \n\nAudit Criterion (Failure = 1): The agent describes visual features clearly absent in the image or contradicts explicit patient data (e.g., saying 'male' when input says 'female'). \nPass (0): The agent's observations are grounded in the actual input data."
        },
        "1.2.1": {
            "name": "Neglect or Misinterpretation of Modality Information during Input Interpretation",
            "definition": "The agent ignores the input modality (e.g., treats an image task as text-only) or fails to answer the specific clinical question.",
            "human_eval_instruction": "Assess if the domain agent effectively utilized the specific modality (e.g., the image) required to answer the question. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent gives a generic definition instead of looking at the image, or ignores the specific question (e.g., describes the X-ray technique instead of checking for Pneumothorax). \n\nPass (0): the agent explicitly analyzes the provided modality and directly addresses the prompt's question."
        },
        "2.1.1": {
            "name": "Mismatch Between Assigned Roles and Clinical Tasks during Collaborative discussion",
            "definition": "The assigned specialist lacks the domain knowledge or modality competence required for the specific case.",
            "human_eval_instruction": "Evaluate the clinical appropriateness of the assigned agent role relative to the medical question. \n\nAudit Criterion (Failure = 1): Mark as 1 if an irrelevant specialist is assigned (e.g., Psychiatrist for a broken bone) or the specialist cannot interpret the required data type (e.g., Dermatologist reading a CT scan). \n\nPass (0): The specialist's domain and modality skills match the clinical needs."
        },
        "2.1.2": {
            "name": "Failure to Activate Specialist Knowledge During Role Execution during Collaborative discussion",
            "definition": "The agent fails to use domain-specific reasoning, offering layperson-level advice or rigidly refusing the task based on its title.",
            "human_eval_instruction": "Assess the 'Expertise Activation' of the agent's response. \n\nAudit Criterion (Failure = 1): Mark as 1 if The output is generic common sense lacking medical depth, OR the agent refuses to analyze the case due to a rigid interpretation of its role. \n\n Pass (0): The agent uses specific terminology, guidelines, and visual observation skills unique to that specialty."
        },
        "2.2.1": {
            "name": "Repetition of Initial Views during Collaborative discussion",
            "definition": "The discussion adds no value; the agent agrees with others without providing new evidence or reasoning (Echo Chamber).",
            "human_eval_instruction": "Audit Criterion (Failure = 1): the agent merely says 'I agree' or repeats the same conclusion without adding new supporting details or verification. \n\nPass (0): the agent provides new evidence, triangulates data, or offers constructive critique to refine the diagnosis."
        },
        "2.2.2": {
            "name": "Unresolved Conflicts during Collaborative discussion",
            "definition": "Agents ignore mutually exclusive claims made during the discussion, continuing as if no contradiction exists.",
            "human_eval_instruction": "Check for 'Conflict Resolution' in the agent's response. \n\nAudit Criterion (Failure = 1): one agent says 'X exists' and another says 'X is absent,' but subsequent responses ignore this clash and do not attempt to resolve it. \n\nPass (0): the agents acknowledge the disagreement and attempt to verify which view is correct."
        },
        "3.1.1": {
            "name": "Suppression of Correct Minority Views by Incorrect Consensus during Decision-making",
            "definition": "The final decision adopts an incorrect majority view, discarding a correct insight provided by a minority.",
            "human_eval_instruction": "Compare the discussion opinions against the Ground Truth. \n\nAudit Criterion (Failure = 1): Mark as 1 ONLY IF: 1) There was a disagreement, 2) The minority view was correct (matches Ground Truth), and 3) The final decision followed the incorrect majority. \n\nPass (0): the majority was correct, or the system successfully recognized and adopted the correct minority view."
        },
        "3.1.2": {
            "name": "Reasoning Distorted by Authority Bias during Decision-making",
            "definition": "The decision is based on the speaker's role or superficial formatting rather than factual verification.",
            "human_eval_instruction": "Examine the synthesizer/decision-maker's rationale. \n\nAudit Criterion (Failure = 1): The agent accepts a view explicitly because 'Dr. X is the Radiologist' or because the text is long/complex, without verifying the actual facts and reasoning process. \n\nPass (0): The agent validates the content against clinical guidelines or image data regardless of who proposed it."
        },
        "3.1.3": {
            "name": "Neglect of Contradictions in Reasoning Process during Decision-making",
            "definition": "The decision claims 'agreement' on the final label while ignoring that the supporting reasons are contradictory.",
            "human_eval_instruction": "Check for 'Logical Coherence' in the synthesis. \n\nAudit Criterion (Failure = 1): Mark as 1 if the synthesizer/decision-maker claims 'the team agrees' but ignores that the agents cited incompatible reasons or findings (e.g., Agent A says Left Lung, Agent B says Right Lung). \n\nPass (0): The decision-maker/synthesizer ensures both the conclusion and the supporting evidence are consistent among the agreeing agents."
        },
        "3.2.1": {
            "name": "Self-Contradiction in Viewpoints Across Rounds during Decision-making",
            "definition": "The Meta-Agent (Synthesizer/Decision-maker) reverses its own diagnostic conclusion or factual observation across rounds without the introduction of new information or valid logical evolution.",
            "human_eval_instruction": "Track the Meta Agent's consistency across rounds. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent flips its diagnosis (e.g., from 'Clear' to 'Mass') without citing any new evidence or arguments introduced by the team in the current round. \n\nPass (0): The agent maintains consistency or explicitly explains why a change in opinion is necessary based on new insights."
        }
    }
    input_dir = EXTRACTED_LOGS_FOR_OPENCODING_HUMAN_EVAL_DIR
    all_json_files = list(input_dir.glob("*.jsonl"))
    print(f"Found {len(all_json_files)} JSONL files in {input_dir}")

    terminal_log_file = STRUCTURED_LOGS_FOR_OPENCODING_HUMAN_EVAL_DIR / f"structured_logs_for_opencoding_human_eval_terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    for jsonl_file in all_json_files:
        print(f"Processing file: {jsonl_file}")
        jsonl_file_name = jsonl_file.stem
        dataset = jsonl_file_name.split("_")[1]
        mas = jsonl_file_name.split("_")[0]
        llm = jsonl_file_name.split("_")[2]
        if dataset in ["MedQA", "PubMedQA", "MedXpertQA-text"]:
            question_type = "plain text question answering"
        else:
            question_type = "visual question answering"
        output_jsonl_file = STRUCTURED_LOGS_FOR_OPENCODING_HUMAN_EVAL_DIR / f"{jsonl_file_name}_structured.jsonl"
        data = load_jsonl(jsonl_file)
        print(f"  - Total records: {len(data)}")
        # we need to extract the info from the json file and assign it to the structured format for human evaluation.
        for json_record in data:
            qid = json_record["qid"]
            image_path = json_record.get("image_path", None)
            options = json_record["options"]
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            ground_truth = json_record["ground_truth"]
            mas_predicted_answer = json_record["predicted_answer"]
            case_history = json_record["case_history"]
            question_description = (
                f"This is a {question_type} case. The question is: {json_record['question']}. \n"
                f"This question has {len(options)} options: {options_text}"
                f"The ground truth answer is: {ground_truth}. The multi agents system's predicted answer is: {mas_predicted_answer}.\n"
            )
            collaboration_text_start, collaboration_text = gen_collaboration_text(case_history)
            instruction_text = (
                "Please conduct a comprehensive analysis of the multi-agent collaboration process for this case, "
                "utilizing the full case context and collaboration history provided.\n\n"
                "Your task is to identify occurrences of the 10 specific failure modes listed in the taxonomy.\n\n"
                "For each failure mode observed, please select (check) the corresponding checkbox.\n\n"
                "If a failure mode is not present, leave it unchecked (do not take any action).\n\n"
                "Should you encounter any other collaboration issues not covered by these 10 categories, "
                "please describe them in the 'Novel failure mode' text box."
            )
            structured_case = {
                "qid": qid,
                "image_path": image_path,
                "ground_truth": ground_truth,
                "question_type": question_type,
                "options": options,
                "options_text": options_text,
                "llm": llm,
                "dataset": dataset,
                "mas": mas,
                "mas_predicted_answer": mas_predicted_answer,
                "question_description": question_description,
                "collaboration_text": collaboration_text,
                "collaboration_start_text" : collaboration_text_start,
                "instruction_text": instruction_text,
                "failure_mode_definition_mapping": failure_mode_definition_mapping
            }
            # Save each structured case to a new JSONL file
            save_jsonl(structured_case, output_jsonl_file)
        
if __name__ == "__main__":
    main()