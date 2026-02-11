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

def gen_collaboration_text(case_history):
    '''
    This function is designed to generate the text description of the multi-agent collaboration process based on the case history.
    The generated text will be used for human evaluation to understand the multi-agent collaboration process.
    '''
    collaboration_text = (
        f"Here is the multi-agent collaboration process for this case:\n\n"
        f"Task Understanding Phase: Each domain agent independently assesses the case and provides its own judgment along with the supporting rationale:\n"
    )
    if "rounds" in case_history and case_history["rounds"]:
        for r in case_history["rounds"]:
            round_num = r.get("round", "Unknown")
            collaboration_text += f"\n--- [Round {round_num}] ---\n"

            for opinion in r.get("opinions", []):
                domain_agent_id= opinion.get("agent_id","").lower()
                past_domain_agent_answer = opinion["log"]["parsed_output"].get("answer", "N/A")
                past_domain_agent_explanation = opinion["log"]["parsed_output"].get("explanation", "N/A")
                collaboration_text += (
                    f"agent ID: {domain_agent_id} (role: {opinion.get('specialty', 'N/A')})\n"
                    f"answer: {past_domain_agent_answer}\n"
                    f"explanation: {past_domain_agent_explanation}\n\n"
                )
            collaboration_text += (                
                f"Multi-Agent Collaborative Discussion Phase:\n"
            )
            if r.get("synthesis"): # not any MAS has the synthesis stage
                collaboration_text += (                
                    f"This stage encompasses the generation of a preliminary conclusion by the meta-agent:\n"
                )
                if isinstance(r["synthesis"], list):
                    for synth_item in r["synthesis"]:
                        synth_log = synth_item.get("log", {}).get("parsed_output", {})
                        past_ans = synth_log.get("answer", "N/A")
                        past_expl = synth_log.get("explanation", "N/A")
                        agent_id = synth_item.get("agent_id", "Unknown Lead")
                        collaboration_text += (
                            f"group lead ({agent_id}) answer: {past_ans}\n"
                            f"group lead explanation: {past_expl}\n\n"
                        )
                elif isinstance(r["synthesis"], dict):
                    past_synthesizer_answer = r["synthesis"]["parsed_output"].get("answer", "N/A")
                    past_synthesizer_explanation = r["synthesis"]["parsed_output"].get("explanation", "N/A")
                    collaboration_text += (
                        f"synthesizer answer: {past_synthesizer_answer}\n"
                        f"synthesizer explanation: {past_synthesizer_explanation}\n\n"
                    )

            if r.get("reviews"): # not any MAS has the review stage
                collaboration_text += (                
                    f"This stage encompasses a review from domain agent providing their perspectives and rationales. "
                    f"It includes cross-evaluation among domain agents, where they exchange viewpoints to refine the collective outcome.\n"
                )
                for review in r["reviews"]:
                    past_domain_agent_review = review["log"]["parsed_output"].get("agree", "N/A")
                    past_domain_agent_review_reason = review["log"]["parsed_output"].get("reason", "N/A")
                    past_domain_agent_review_explanation = review["log"]["parsed_output"].get("explanation", "N/A")
                    past_domain_agent_review_answer = review["log"]["parsed_output"].get("answer", "N/A")
                    collaboration_text += (
                        f"agent ID: {review.get('agent_id', 'N/A')} (Role: {review.get('specialty', 'N/A')})\n"
                        f"review_result: {past_domain_agent_review}\n"
                        f"review_reason: {past_domain_agent_review_reason}\n"
                        f"review_explanation: {past_domain_agent_review_explanation}\n"
                        f"review_answer: {past_domain_agent_review_answer}\n\n"
                    )

            if r.get("decision"): 
                collaboration_text += (                
                    f"This stage encompasses the final decision-making process, where the meta-agent consolidates the insights from previous stages to arrive at a conclusive answer:\n"
                )
                past_decision_answer = r["decision"]["parsed_output"].get("answer", "N/A")
                past_decision_explanation = r["decision"]["parsed_output"].get("explanation", "N/A")
                collaboration_text += (
                    f"decision answer: {past_decision_answer}\n"
                    f"decision explanation: {past_decision_explanation}\n\n"
                )
    return collaboration_text
def main():
    input_dir = EXTRACTED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR
    all_json_files = list(input_dir.glob("*.jsonl"))
    print(f"Found {len(all_json_files)} JSONL files in {input_dir}")

    terminal_log_file = STRUCTURED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR / f"structured_logs_for_audit_human_eval_terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    failure_mode_status_mapping = {
        "1.1.1": "factual_hallucination_status",
        "1.2.1": "modality_neglect_status",
        "2.1.1": "role_task_alignment",
        "2.1.2": "knowledge_activation_status",
        "2.2.1": "interaction_redundancy",
        "2.2.2": "conflict_resolution_status",
        "3.1.1": "suppression_status",
        "3.1.2": "authority_bias_status",
        "3.1.3": "neglect_of_conflict_status",
        "3.2.1": "inter_round_consistency_status"
    }

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
            dataset = json_record["dataset"]
            llm = json_record["llm"]
            mas = json_record["mas"]
            case_history = json_record["case_history"]
            if failure_code in ["1.1.1"] :
                # find the earliest round when the failure mode happens
                failure_mode_log_key = failure_mode_log_key_mapping[failure_code]
                earliest_round = None
                for r in case_history["audit"]["rounds"]:
                    if failure_mode_log_key in r and r[failure_mode_log_key]:
                        earliest_round = r["round"]
                        break
                print(f"  - Earliest round for failure mode {failure_code}: {earliest_round}")
                failure_mode_status = failure_mode_status_mapping[failure_code]
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
                
            if dataset in ["MedQA", "PubMedQA", "MedXpertQA-text"]:
                question_type = "plain text question answering"
            else:
                question_type = "visual question answering"
            # we firstly need to cut the case history to the audit timing for shortening the recognition load for human eval.
            qid = json_record["qid"]
            image_path = json_record.get("image_path", None)
            options = json_record["options"]
            options_text = "\nOptions:\n"
            for key, value in options.items():
                options_text += f"{key}: {value}\n"
            ground_truth = json_record["ground_truth"]
            mas_predicted_answer = json_record["predicted_answer"]
            question_description = (
                f"This is a {question_type} case. The question is: {json_record['question']}. \n"
                f"This question has {len(options)} options: {options_text}"
                f"The ground truth answer is: {ground_truth}. The multi agents system's predicted answer is: {mas_predicted_answer}.\n"
            )
            case_history = json_record["case_history"] # at this time the case history has been cut to the audit timing for shortening the recognition load for human eval.
            collaboration_text = gen_collaboration_text(case_history)
            instruction_text = (
                f"Please conduct a comprehensive analysis of the multi-agent collaboration process for this case, utilizing the full case context and collaboration history provided.\n"
                f"Your task is to evaluate the collaboration against 10 specific failure modes. For each of the 10 modes, you must provide a clear annotation (e.g., 1 (fail)/ 0 (pass))."
                f"If you observe any other failure modes in the collaboration that fall outside the 10 categories, please document them as new failure modes."
            )
            structured_case = {
                "qid": qid,
                "image_path": image_path,
                "question_description": question_description,
                "collaboration_text": collaboration_text,
                "instruction_text": instruction_text
            }
            # Save each structured case to a new JSONL file
            save_jsonl(structured_case, output_jsonl_file)
        
if __name__ == "__main__":
    main()