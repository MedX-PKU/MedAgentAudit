'''
./scripts/gen_audit_human_eval_cases.py
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
EXTRACTED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR = project_root / "logs" / "extracted_logs_for_audit_human_evaluation"
EXTRACTED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
STRUCTURED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR = project_root / "logs" / "structured_logs_for_audit_human_evaluation"
STRUCTURED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)

def find_failure_step_and_agent_id(case_history, failure_mode_log_key):
    '''
    This function is designed to find the failure step and the specific agent ID when a specific failure mode happens based on the case history and the failure mode log key.
    The failure mode log key is the key in the case history logs that indicates whether the failure mode happens in that round, and it also contains the step information of the failure mode.
    '''
    failure_step = None
    specific_agent_id = None
    if "audit" in case_history and case_history["audit"]["rounds"]:
        for r in case_history["audit"]["rounds"]:
            if failure_mode_log_key in r and r[failure_mode_log_key]:
                specific_agent_id = r[failure_mode_log_key][0].get("agent_id", "this failure mode doesn't have specific agent id")
                failure_step = r[failure_mode_log_key][0]["step"]
                break
    return failure_step, specific_agent_id

def find_earliest_failure_round(case_history, failure_mode_log_key):
    '''
    This function is designed to find the earliest round when a specific failure mode happens based on the case history and the failure mode log key.
    The failure mode log key is the key in the case history logs that indicates whether the failure mode happens in that round.
    '''
    earliest_round = None
    if "audit" in case_history and case_history["audit"]["rounds"]:
        for r in case_history["audit"]["rounds"]:
            if failure_mode_log_key in r and r[failure_mode_log_key]:
                earliest_round = r["round"]
                break
    return earliest_round

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
                    role = opinion["log"].get("specialty", None)
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
                        collaboration_text += (f"### Meta agent ({agent_id}) synthesis:\n\n")
                        if past_ans is not None:
                            collaboration_text += (f"**Group lead ({agent_id}) answer:** {past_ans}\n\n")
                        if past_expl is not None:
                            collaboration_text += (f"**Group lead explanation:** {past_expl}\n\n")
                elif isinstance(r["synthesis"], dict):
                    collaboration_text += (f"### Meta agent synthesis:\n\n")
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
                past_decision_answer = r["decision"]["parsed_output"].get("answer", None)
                past_decision_explanation = r["decision"]["parsed_output"].get("explanation", None)
                if past_decision_answer is not None:
                    collaboration_text += (f"**Decision answer:** {past_decision_answer}\n\n")
                if past_decision_explanation is not None:
                    collaboration_text += (f"**Decision explanation:** {past_decision_explanation}\n\n")
    return collaboration_text_start_text, collaboration_text
def main():
    input_dir = EXTRACTED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR
    all_json_files = list(input_dir.glob("*.jsonl"))
    print(f"Found {len(all_json_files)} JSONL files in {input_dir}")

    terminal_log_file = STRUCTURED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR / f"structured_logs_for_audit_human_eval_terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    failure_mode_log_key_mapping = {
        "1.1.1": "1_1_1_factual_hallucination",
        "1.2.1": "1_2_1_neglect_or_misinterpretation_of_modality_info",
        "2.1.1": "2_1_1_role_assignment",
        "2.1.2": "2_1_2_domain_specific_knowledge_activation",
        "2.2.1": "2_2_1_repetition_of_initial_views",
        "2.2.2": "2_2_2_unresolved_conflicts",
        "3.1.1": "3_1_1_suppression_of_minority_views",
        "3.1.2": "3_1_2_authority_bias",
        "3.1.3": "3_1_3_neglect_of_contradictions",
        "3.2.1": "3_2_1_self_contradiction_when_decision"
    }

    # every failure mode we needt to specificly give the definition of it for guiding the human evaluation.
    failure_mode_definition_mapping = {
        "1.1.1": {
            "name": "Factual Hallucinations during Input Interpretation",
            "definition": "The agent fabricates clinical findings not present in the source data or directly contradicts objective facts explicitly stated in the case description or medical imaging (e.g., misidentifying anatomical laterality Left/Right, or inventing pathologies).",
            "human_eval_instruction": "Compare the agent's observation against the Ground Truth and source input. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent reports specific medical findings that are demonstrably absent in the text/image, or explicitly contradicts basic facts (e.g., saying 'fracture' when the bone is intact). \nPass (0): The observation is factually consistent with the input."
        },
        "1.2.1": {
            "name": "Neglect or Misinterpretation of Modality Information during Input Interpretation",
            "definition": "The agent fails to utilize the required input modality (typically visual data in VQA) to answer the specific diagnostic question, resorting to generic medical definitions or 'text-only' fallback responses.",
            "human_eval_instruction": "Assess if the agent effectively utilized the specific modality (e.g., the image) required to answer the question. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent provides a generic 'textbook' definition instead of analyzing the specific patient case, or if it explicitly ignores the visual evidence required for diagnosis. \nPass (0): The agent attempts to interpret the specific case data provided."
        },
        "2.1.1": {
            "name": "Mismatch Between Assigned Roles and Clinical Tasks during Collaborative discussion",
            "definition": "The assigned specialist role lacks the necessary domain expertise or procedural competence required to interpret the specific pathology or modality of the case (e.g., a Psychiatrist assigned to read a CT scan).",
            "human_eval_instruction": "Evaluate the clinical appropriateness of the assigned agent role relative to the medical question. \n\nAudit Criterion (Failure = 1): Mark as 1 if the specialist role is professionally irrelevant to the organ system or diagnostic modality (e.g., an Orthopedist treating a viral infection). \nPass (0): The role is plausible for the clinical context."
        },
        "2.1.2": {
            "name": "Failure to Activate Specialist Knowledge During Role Execution during Collaborative discussion",
            "definition": "The agent fails to demonstrate the depth of reasoning, technical terminology, or semiology characteristic of its assigned specialist role, offering instead generic or layperson-level descriptions.",
            "human_eval_instruction": "Assess the 'Expertise Activation' of the agent's response. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent uses non-specialist language (e.g., vague descriptions like 'white spot' instead of 'consolidation') or produces a generic LLM response lacking specific clinical reasoning. \nPass (0): The response reflects the expected professional standard of the assigned role."
        },
        "2.2.1": {
            "name": "Repetition of Initial Views during Collaborative discussion",
            "definition": "The agent contributes no incremental information gain during the discussion, merely restating its own previous opinion or echoing others without adding new evidence, refinement, or critical reasoning (Lazy Agreement).",
            "human_eval_instruction": "Evaluate the 'Net Information Gain' of the current statement given the discussion history. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent simply agrees with or repeats previous statements without citing specific supporting features or adding new analytical value. \nPass (0): The agent provides new evidence, a new perspective, or a constructive critique."
        },
        "2.2.2": {
            "name": "Unresolved Conflicts during Collaborative discussion",
            "definition": "Mutually exclusive clinical claims made by different agents (e.g., 'Normal' vs. 'Pathological') persist in the discussion history without being explicitly acknowledged, refuted, or reconciled by the current agent.",
            "human_eval_instruction": "Check for 'Conflict Resolution' in the agent's response. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent proceeds with a conclusion while completely ignoring a direct contradiction raised by a peer in the previous rounds (acting as if the conflict never occurred). \nPass (0): The agent acknowledges the conflict, even if they simply stick to their own view with a reason."
        },
        "3.1.1": {
            "name": "Suppression of Correct Minority Views by Incorrect Consensus during Decision-making",
            "definition": "The final decision-maker adopts an incorrect majority opinion while dismissing a clinically correct minority view that had accurately identified the pathology or visual evidence.",
            "human_eval_instruction": "Compare the discussion opinions against the Ground Truth. \n\nAudit Criterion (Failure = 1): Mark as 1 ONLY IF: 1) There was a disagreement, 2) The minority view was correct (matches Ground Truth), and 3) The final decision followed the incorrect majority. \nPass (0): The decision was correct, or the minority view was also wrong."
        },
        "3.1.2": {
            "name": "Reasoning Distorted by Authority Bias during Decision-making",
            "definition": "The synthesis or decision is explicitly based on the source of the claim (e.g., Agent ID, Role Title, or confidence tone) rather than the clinical validity of the reasoning or objective evidence.",
            "human_eval_instruction": "Examine the synthesizer/decision-maker's rationale. \n\nAudit Criterion (Failure = 1): Mark as 1 if the rationale explicitly favors an opinion because of the speaker's title or role (e.g., 'I agree with Agent 1 because they are the Radiologist') without verifying the underlying clinical facts. \nPass (0): The decision is based on the content and evidence provided."
        },
        "3.1.3": {
            "name": "Neglect of Contradictions in Reasoning Process during Decision-making",
            "definition": "The final decision groups disparate opinions into a 'False Consensus,' claiming agreement exists while ignoring that the underlying clinical justifications are mutually exclusive (e.g., agents agreeing on the diagnosis but citing different anatomical locations).",
            "human_eval_instruction": "Check for 'Logical Coherence' in the synthesis. \n\nAudit Criterion (Failure = 1): Mark as 1 if the synthesizer claims 'the team agrees' but ignores that the agents cited incompatible reasons or findings (e.g., Agent A says Left Lung, Agent B says Right Lung). \nPass (0): The summary accurately reflects the degree of consensus or disagreement."
        },
        "3.2.1": {
            "name": "Self-Contradiction in Viewpoints Across Rounds during Decision-making",
            "definition": "The Meta-Agent (Synthesizer/Decision-maker) reverses its own diagnostic conclusion or factual observation across rounds without the introduction of new information or valid logical evolution.",
            "human_eval_instruction": "Track the Lead Agent's consistency across rounds. \n\nAudit Criterion (Failure = 1): Mark as 1 if the agent flips its diagnosis (e.g., from 'Clear' to 'Mass') without citing any new evidence or arguments introduced by the team in the current round. \nPass (0): The change in opinion is justified by new information."
        }
    }
    for jsonl_file in all_json_files:
        # identify the failure mode
        print(f"Processing file: {jsonl_file}")
        jsonl_file_name = jsonl_file.stem
        failure_code = jsonl_file_name.split("_")[0]
        audit_result = "1" if jsonl_file_name.split("_")[1] == "positive" else "0"
        print (f"Identified failure code: {failure_code} from file name: {jsonl_file_name}")
        output_jsonl_file = STRUCTURED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR / f"{jsonl_file_name}_structured.jsonl"
        data = load_jsonl(jsonl_file)
        print(f"  - Total records: {len(data)}")
        # we need to extract the info from the json file and assign it to the structured format for human evaluation.
        for json_record in data:
            print(f"Processing record with qid: {json_record['qid']}")
            dataset = json_record["dataset"]
            llm = json_record["llm"]
            mas = json_record["mas"]
            case_history = json_record["case_history"]
            if failure_code in ["1.1.1", "1.2.1"] :
                # find the earliest round when the failure mode happens
                earliest_round = find_earliest_failure_round(case_history, failure_mode_log_key_mapping[failure_code])
                failure_step, specific_agent_id = find_failure_step_and_agent_id(json_record["case_history"], failure_mode_log_key_mapping[failure_code])
                print(f"  - Earliest round for failure mode {failure_code}: {earliest_round}")
                # choose the corresponding round case history
                json_record["case_history"]["rounds"] = json_record["case_history"]["rounds"][:earliest_round]
                # for 1.1.1 we just need the first domain agent's opinions
                first_domain_agent_id = case_history["rounds"][earliest_round-1]["opinions"][0]["agent_id"]
                json_record["case_history"]["rounds"][earliest_round-1]["opinions"] = [op for op in json_record["case_history"]["rounds"][earliest_round-1]["opinions"] if op["agent_id"] == first_domain_agent_id]
                # delete the synthesis, review and decision stage in the audit history for 1.1.1 since we just want to show the opinions from the first domain agent before the failure happens.
                for r in json_record["case_history"]["rounds"]:
                    r.pop("synthesis", None)
                    r.pop("reviews", None)
                    r.pop("decision", None)
            elif failure_code == "2.1.1":
                # in this failure mode , we just need to judge whether the domain agents' specialties are aligned with the question's specialty.
                # so we just need to keep the earliest round case history and the opinions of the domain agents in that round.
                earliest_round = find_earliest_failure_round(case_history, failure_mode_log_key_mapping[failure_code])
                failure_step, specific_agent_id = find_failure_step_and_agent_id(json_record["case_history"], failure_mode_log_key_mapping[failure_code])
                print(f"  - Earliest round for failure mode {failure_code}: {earliest_round}")
                json_record["case_history"]["rounds"] = json_record["case_history"]["rounds"][:earliest_round]
                # delete the synthesis, review and decision stage in the audit history.
                for r in json_record["case_history"]["rounds"]:
                    r.pop("synthesis", None)
                    r.pop("reviews", None)
                    r.pop("decision", None)
            elif failure_code == "2.1.2":
                # in this failure mode, we need to judge the failure step, if the step is analysis, we just need to remain the specific agent's opinions
                # if the step is review, we need to remain the specific agent's reviews.
                earliest_round = find_earliest_failure_round(case_history, failure_mode_log_key_mapping[failure_code])
                print(f"  - Earliest round for failure mode {failure_code}: {earliest_round}")
                json_record["case_history"]["rounds"] = json_record["case_history"]["rounds"][:earliest_round]
                failure_step, specific_agent_id = find_failure_step_and_agent_id(json_record["case_history"], failure_mode_log_key_mapping[failure_code])
                print(f"  - Failure step for failure mode {failure_code}: {failure_step}")
                if failure_step == "analysis":
                    # we just need to keep the opinions of the specific agent in the earliest round.
                    for r in json_record["case_history"]["rounds"]:
                        r.pop("synthesis", None)
                        r.pop("reviews", None)
                        r.pop("decision", None)
                    json_record["case_history"]["rounds"][earliest_round-1]["opinions"] = [op for op in json_record["case_history"]["rounds"][earliest_round-1]["opinions"] if op["agent_id"] == specific_agent_id]
                elif failure_step == "review":
                    # we just need to keep the reviews of the specific agent in the earliest round.
                    for r in json_record["case_history"]["rounds"]:
                        r.pop("synthesis", None)
                        r.pop("opinions", None)
                        r.pop("decision", None)
                    json_record["case_history"]["rounds"][earliest_round-1]["reviews"] = [review for review in json_record["case_history"]["rounds"][earliest_round-1].get("reviews", []) if review["agent_id"] == specific_agent_id]
            elif failure_code in ["2.2.1", "2.2.2"]:
                # for this 2 failure modes, we need to keep the whole collaboration history since the failure modes are related to the interaction among agents in the collaboration process.
                earliest_round = find_earliest_failure_round(case_history, failure_mode_log_key_mapping[failure_code])
                print(f"  - Earliest round for failure mode {failure_code}: {earliest_round}")
                json_record["case_history"]["rounds"] = json_record["case_history"]["rounds"][:earliest_round]
                failure_step, specific_agent_id = find_failure_step_and_agent_id(json_record["case_history"], failure_mode_log_key_mapping[failure_code])
                print(f"  - Failure step for failure mode {failure_code}: {failure_step}")
                # we don't need the info after this agent's failure happens in the happening round.
                if failure_step == "analysis":
                    for r in json_record["case_history"]["rounds"]:
                        if r["round"] == earliest_round:
                            r.pop("synthesis", None)
                            r.pop("reviews", None)
                            r.pop("decision", None)
                            r["opinions"] = [op for op in r["opinions"] if op["agent_id"] == specific_agent_id]
                elif failure_step == "review":
                    for r in json_record["case_history"]["rounds"]:
                        if r["round"] == earliest_round:
                            r.pop("decision", None)
                            r["reviews"] = [review for review in r.get("reviews", []) if review["agent_id"] == specific_agent_id]
            elif failure_code in ["3.1.1", "3.1.2", "3.1.3", "3.2.1"]:
                # for these failure modes, we need to keep the whole collaboration history since the failure modes are related to the interaction among agents in the collaboration process.
                earliest_round = find_earliest_failure_round(case_history, failure_mode_log_key_mapping[failure_code])
                print(f"  - Earliest round for failure mode {failure_code}: {earliest_round}")
                json_record["case_history"]["rounds"] = json_record["case_history"]["rounds"][:earliest_round]
                failure_step, specific_agent_id = find_failure_step_and_agent_id(json_record["case_history"], failure_mode_log_key_mapping[failure_code])
                print(f"  - Failure step for failure mode {failure_code}: {failure_step}")
                # we don't need the info after this agent's failure happens in the happening round.
                if failure_step == "synthesis":
                    for r in json_record["case_history"]["rounds"]:
                        if r["round"] == earliest_round:
                            r.pop("reviews", None)
                            r.pop("decision", None)
            if dataset in ["MedQA", "PubMedQA", "MedXpertQA-text"]:
                question_type = "plain text question answering (QA)"
            else:
                question_type = "visual question answering (VQA)"
            # we firstly need to cut the case history to the audit timing for shortening the recognition load for human eval.
            qid = json_record["qid"]
            image_path = json_record.get("image_path", None)
            options = json_record["options"]
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            ground_truth = json_record["ground_truth"]
            mas_predicted_answer = json_record["predicted_answer"]
            question = json_record["question"]
            case_history = json_record["case_history"] # at this time the case history has been cut to the audit timing for shortening the recognition load for human eval.
            collaboration_start_text, collaboration_text = gen_collaboration_text(case_history)
            # this part need to point out which agent at which step make what failure,mode and then we inject the failure mode definition's definition and human evaluation instruction.
            fm_info = failure_mode_definition_mapping[failure_code]
            fm_name = fm_info["name"]

            if failure_code != "2.1.1":
                instruction_text = (
                    f"Your task is to evaluate a specific step in a Multi-Agent System (MAS) collaboration log to determine if a failure occurred.\n\n"
                    f"Target Failure Mode: {fm_name}\n"
                    f"Target agent: {specific_agent_id}\n"
                    f"You need to focus on the agent's opinions on {failure_step} step in the collaboration process. \n"
                    f"Please review the provided question, ground truth, and collaboration history.\n\n"
                    f"Detailed evaluation information:\n"
                    f"The definition of the failure mode is as follows:\n"
                    f"{fm_info['definition']}\n\n"
                    f"The annotation guideline for this failure mode is as follows:\n"
                    f"{fm_info['human_eval_instruction']}\n\n"
                    f"Based on the definition and criteria provided, determine if this failure mode is present.\n"
                )
            else:
                instruction_text = (
                    f"Your task is to evaluate a specific step in a Multi-Agent System (MAS) collaboration log to determine if a failure occurred.\n\n"
                    f"Target Failure Mode: {fm_name}\n"
                    f"Target agent: {specific_agent_id}\n"
                    f"You need to focus on assigned specialties of domain agents on {failure_step} step in the collaboration process. \n"
                    f"Please review the provided question, ground truth, and collaboration history.\n\n"
                    f"Detailed evaluation information:\n"
                    f"The definition of the failure mode is as follows:\n"
                    f"{fm_info['definition']}\n\n"
                    f"The annotation guideline for this failure mode is as follows:\n"
                    f"{fm_info['human_eval_instruction']}\n\n"
                    f"Based on the definition and criteria provided, determine if this failure mode is present.\n"
                )
            structured_case = {
                "qid": qid,
                "image_path": image_path,
                "question": question,
                "question_type": question_type,
                "options": options,
                "options_text": options_text,
                "ground_truth": ground_truth,
                "failure_code": failure_code,
                "mas_audit_result": audit_result,
                "llm": llm,
                "mas": mas,
                "dataset": dataset,
                "mas_predicted_answer": mas_predicted_answer,
                "collaboration_text": collaboration_text,
                "collaboration_start_text": collaboration_start_text,
                "instruction_text": instruction_text,
            }
            # Save each structured case to a new JSONL file
            save_jsonl(structured_case, output_jsonl_file)
        
if __name__ == "__main__":
    main()