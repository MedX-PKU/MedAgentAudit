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
            collaboration_text = gen_collaboration_text(case_history)
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
                "instruction_text": instruction_text
            }
            # Save each structured case to a new JSONL file
            save_jsonl(structured_case, output_jsonl_file)
        
if __name__ == "__main__":
    main()