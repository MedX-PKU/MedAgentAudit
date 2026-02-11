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

def gen_case_history_text(case_history):
    synthesizer_opinions_text = ""
    domain_agent_past_history_opinions_text = ""
    domain_agent_past_history_reviews_text = ""
    decision_opinions_text = ""
    
    if "rounds" in case_history and case_history["rounds"]:
        for r in case_history["rounds"]:
            round_num = r.get("round", "Unknown")
            domain_agent_past_history_opinions_text += f"\n--- [Round {round_num}] ---\n"

            for opinion in r.get("opinions", []):
                domain_agent_id= opinion.get("agent_id","").lower()
                past_domain_agent_answer = opinion["log"]["parsed_output"].get("answer", "N/A")
                past_domain_agent_explanation = opinion["log"]["parsed_output"].get("explanation", "N/A")
                domain_agent_past_history_opinions_text += (
                    f"agent ID: {domain_agent_id} (role: {opinion.get('specialty', 'N/A')})\n"
                    f"answer: {past_domain_agent_answer}\n"
                    f"explanation: {past_domain_agent_explanation}\n\n"
                )
            if r.get("reviews"): # not any MAS has the review stage
                domain_agent_past_history_reviews_text += f"\n--- [Round {round_num}] ---\n"
                for review in r["reviews"]:
                    past_domain_agent_review = review["log"]["parsed_output"].get("agree", "N/A")
                    past_domain_agent_review_reason = review["log"]["parsed_output"].get("reason", "N/A")
                    past_domain_agent_review_explanation = review["log"]["parsed_output"].get("explanation", "N/A")
                    past_domain_agent_review_answer = review["log"]["parsed_output"].get("answer", "N/A")
                    domain_agent_past_history_reviews_text += (
                        f"agent ID: {review.get('agent_id', 'N/A')} (Role: {review.get('specialty', 'N/A')})\n"
                        f"review_result: {past_domain_agent_review}\n"
                        f"review_reason: {past_domain_agent_review_reason}\n"
                        f"review_explanation: {past_domain_agent_review_explanation}\n"
                        f"review_answer: {past_domain_agent_review_answer}\n\n"
                    )
            if r.get("synthesis"): # not any MAS has the synthesis stage
                if isinstance(r["synthesis"], list):
                    synthesizer_opinions_text += f"\n--- [Round {round_num} - Group Reports] ---\n"
                    for synth_item in r["synthesis"]:
                        synth_log = synth_item.get("log", {}).get("parsed_output", {})
                        past_ans = synth_log.get("answer", "N/A")
                        past_expl = synth_log.get("explanation", "N/A")
                        agent_id = synth_item.get("agent_id", "Unknown Lead")
                        synthesizer_opinions_text += (
                            f"group lead ({agent_id}) answer: {past_ans}\n"
                            f"group lead explanation: {past_expl}\n\n"
                        )
                elif isinstance(r["synthesis"], dict):
                    synthesizer_opinions_text += f"\n--- [Round {round_num}] ---\n"
                    past_synthesizer_answer = r["synthesis"]["parsed_output"].get("answer", "N/A")
                    past_synthesizer_explanation = r["synthesis"]["parsed_output"].get("explanation", "N/A")
                    synthesizer_opinions_text += (
                        f"synthesizer answer: {past_synthesizer_answer}\n"
                        f"synthesizer explanation: {past_synthesizer_explanation}\n\n"
                    )
            if r.get("decision"): 
                decision_opinions_text += f"\n--- [Round {round_num}] ---\n"
                past_decision_answer = r["decision"]["parsed_output"].get("answer", "N/A")
                past_decision_explanation = r["decision"]["parsed_output"].get("explanation", "N/A")
                decision_opinions_text += (
                    f"decision answer: {past_decision_answer}\n"
                    f"decision explanation: {past_decision_explanation}\n\n"
                )
    return synthesizer_opinions_text, domain_agent_past_history_opinions_text, domain_agent_past_history_reviews_text, decision_opinions_text

def main():
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
            synthesizer_opinions_text, domain_agent_past_history_opinions_text, domain_agent_past_history_reviews_text, decision_opinions_text = gen_case_history_text(case_history)
            collaboration_process = (
                f"Here is the multi-agent collaboration process for this case:\n"
                f"Task Understanding Phase: Each domain agent independently assesses the case and provides its own judgment along with the supporting rationale:\n"
                f"{domain_agent_past_history_opinions_text}\n\n"
                f"Multi-Agent Collaborative Discussion Phase: This stage encompasses the generation of a preliminary conclusion by the meta-agent (some cases), followed by a review from domain agent providing their perspectives and rationales. "
                f"It includes cross-evaluation among domain agents, where they exchange viewpoints to refine the collective outcome.\n"
                f"Domain agents' reviews: {domain_agent_past_history_reviews_text}\n"
                f"{synthesizer_opinions_text}\n"
                f"{decision_opinions_text}\n"
            )
            # Save each structured case to a new JSONL file
            save_jsonl([structured_case], output_file, mode="a")
        
if __name__ == "__main__":
    main()