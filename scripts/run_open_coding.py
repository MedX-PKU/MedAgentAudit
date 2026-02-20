'''
./scripts/run_open_coding.py
'''
import os
import json
import argparse
import time
from typing import Dict, Any, Optional, List, Union
from tqdm import tqdm
import sys
from typing import Tuple
from pathlib import Path
from medagentaudit.utils.logger import DualLogger
from medagentaudit.utils.encode_image import encode_image
from medagentaudit.utils.json_utils import load_jsonl, save_jsonl, preprocess_response_string
from medagentaudit.core.base_agent import BaseAgent
from medagentaudit.common.agent_type import AgentType
# Ensure project root is in path
current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))


class Opencoding(BaseAgent):
    """
    open-coding automated open-coder for constructing full version taxonomy!
    """
    def __init__(self, model_key: str, max_retries: int = 3, retry_delay: int = 5, config_path: str = "config.toml"):
        """
        Args:
            model_key: The model key from LLM_MODELS_SETTINGS.
            max_retries: Maximum number of retries for API call failures.
            retry_delay: Delay time (in seconds) between each retry.
            config_path: Path to the configuration file.
        """
        super().__init__(agent_id="open_coder", agent_type=AgentType.OPENCODER, config_path=config_path, model_key=model_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        print(f"Initializing open-coder agent, ID: open_coder, Model: {model_key}")

    def opencoding(self, system_message: str, user_message: Union[str, List[Dict[str, Any]]], qid: str) -> Optional[Dict[str, Any]]:
        """
        Annotate a single case log using the LLM.
        """
        user_message["content"] = [item for item in user_message["content"] if item.get("type") != "image_url"]
        retries = 0
        opencoding_log = {
            "qid": qid,
            "model_name": self.model_name,
            "request": {
                "system": system_message,
                "user": user_message
            }
        }

        while retries < self.max_retries:
            try:
                response_content, reasoning_content, _, _ = self.call_llm(system_message = system_message, 
                                                                                       user_message = user_message, 
                                                                                       response_format={"type": "json_object"})
                print(f"Raw response for qid {qid}: {repr(response_content[:100])}")
                if not response_content or response_content.strip() == "":
                    raise ValueError("Empty response from API")
                
                result = json.loads(preprocess_response_string(response_content))

                if result is None:
                    raise ValueError("Could not extract valid JSON from response")
                opencoding_log["parsed_output"] = result
                opencoding_log["reasoning_content"] = reasoning_content
                return opencoding_log

            except Exception as e:
                retries += 1
                error_message = f"LLM API call error for qid {qid} (attempt {retries}/{self.max_retries}): {e}"
                print(error_message)
                opencoding_log["error"] = error_message
                if retries >= self.max_retries:
                    opencoding_log["error"] = str(e)
                    print(f"LLM API call failed for qid {qid} after all retries.")
                    return opencoding_log
                return opencoding_log

def gen_collaboration_text(case_history):
    '''
    This function is designed to generate the text description of the multi-agent collaboration process based on the case history.
    The generated text will be used for human evaluation to understand the multi-agent collaboration process.
    '''
    collaboration_text = ""
    if "rounds" in case_history and case_history["rounds"]:
        for r in case_history["rounds"]:
            round_num = r.get("round", "Unknown")
            collaboration_text += f"# --- [Round {round_num}] --- \n\n"
            if r.get("opinions"):
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
                if isinstance(r["synthesis"], list):
                    for synth_item in r["synthesis"]:
                        synth_log = synth_item.get("log", {}).get("parsed_output", {})
                        past_ans = synth_log.get("answer", None)
                        past_expl = synth_log.get("explanation", None)
                        agent_id = synth_item.get("agent_id", None)
                        role = synth_item.get("specialty", None)
                        collaboration_text += ("### Meta agent's synthesis:\n\n")
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
                    collaboration_text += ("### Meta agent synthesis:\n\n")
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
                for review in r["reviews"]:
                    past_domain_agent_review = review["log"]["parsed_output"].get("agree", None)
                    past_domain_agent_review_reason = review["log"]["parsed_output"].get("reason", None)
                    past_domain_agent_review_explanation = review["log"]["parsed_output"].get("explanation", None)
                    past_domain_agent_review_answer = review["log"]["parsed_output"].get("answer", None)
                    agent_id = review.get("agent_id", None)
                    if agent_id: 
                        agent_id = agent_id.lower()
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
                collaboration_text += ("### Meta agent makes decision:\n\n")
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
    return collaboration_text


def build_open_coding_prompts(item: Dict[str, Any], mas: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    construct the prompt basing on the characteristics of different multi-agent frameworks, return system message and user message respectively.
    """
    system_text = (
        "You are a distinguished expert in medical artificial intelligence, "
        "possessing profound domain expertise in both clinical medicine and the architecture of medical multi-agent systems.")
    system_message = {"role": "system", "content": system_text}
    if mas.lower() == 'colacare':
        mas_description = ("ColaCare employs a static role assignment strategy initialized before collaboration begins. The workflow follows a sequential structure: "
                           "Initial Analysis: Assigned Doctor Agents independently provide their initial medical opinions based on the case. "
                           "Synthesis: A Meta Agent aggregates these disparate opinions to formulate a preliminary conclusion. "
                           "Peer Review: Doctor Agents review this preliminary conclusion, indicating their position (Agree/Disagree) and providing the rationale for their judgment. "
                           "Decision Making: The Decision-Maker analyzes the full context of the reviews to determine the final conclusion. "
                           "The system outputs a final answer only when a unanimous consensus is reached among all reviewers or when the maximum number of discussion rounds is exhausted. ")
    elif mas.lower() == 'mac':
        mas_description = ("The MAC system operates without specific role assignments (e.g., cardiologist, neurologist); instead, all participating agents are prompted to act as undifferentiated, general medical experts. "
                           "The collaborative workflow follows an iterative consensus mechanism: "
                           "1. Initial Analysis: In each round, multiple doctor agents first independently provide their initial analysis and reasoning regarding the case. "
                           "2. Supervisor Evaluation: A Supervisor agent synthesizes these inputs to evaluate whether a consensus has been reached among the doctor agents. "
                           "3. Decision or Iteration: "
                           "- If the Supervisor determines that a consensus exists, it outputs the final answer and concludes the session. "
                           "- If no consensus is found, the Supervisor triggers a new round of discussion for further deliberation. ")
    elif mas.lower() == 'healthcareagent':
        mas_description = ("The HealthcareAgent framework operates through a structured \"Plan-Analyze-Review-Decide\" workflow designed to ensure safety and accuracy in clinical reasoning: "
                           "-Planning & Inquiry: The system first evaluates the clarity of the medical problem. "
                           "If the query is deemed ambiguous, it initiates an inquiry process, generating three relevant follow-up questions to enrich the context before proceeding. "
                           "-Initial Analysis: A domain agent performs the preliminary medical analysis and diagnostic reasoning based on the (potentially enriched) context. "
                           "-Safety & Ethics Review: The initial analysis undergoes a rigorous review by specialized Safety & Ethics Reviewers. "
                           "This stage includes specific checks for Medical Ethics, Medical Risks, and Medical Errors. "
                           "-Decision Making: Finally, a Decision-maker aggregates the initial analysis along with the feedback from all safety reviewers to formulate the final clinical decision. ")
    elif mas.lower() == 'mdagents':
        mas_description = ("MDAgents utilizes an adaptive framework that first assesses the complexity of the medical query (Basic, Intermediate, or Advanced) to determine the collaboration structure. "
                           "1. Intermediate Complexity: "
                            "-Recruitment:** The system dynamically recruits a panel of experts (default: 5) tailored to the specific medical query. "
                            "-Process: These experts conduct an independent Initial Analysis. Their findings are then aggregated and finalized by a Decision-Maker agent. "
                            "2. Advanced Complexity: "
                            "-Recruitment: The system recruits multiple specialized groups (default: 3 groups, including an Initial Assessment Team and a Final Review and Decision Team [FDT], totaling 9 experts. "
                            "-Hierarchy: Each group operates under a Team Leader. "
                            "-Process: The IDT reports first, followed by intermediate groups, and finally the FDT. "
                            "The workflow culminates in a report generated by the team leaders, which is then used by Decision-Maker to render the final diagnosis. ")
    elif mas.lower() == 'medagent':
        mas_description = (
            "MedAgent operates on a consensus-driven, multi-stage collaboration framework: "
            "Dynamic Recruitment: Based on the input medical problem, the system dynamically recruits relevant domain expert agents. "
            "Initial Analysis: Each recruited expert conducts a preliminary analysis of the case. "
            "Synthesis: A dedicated Synthesizer agent aggregates the distinct viewpoints from the domain experts into a unified preliminary answer. "
            "Review & Feedback: The domain experts review the Synthesizer's output. They must explicitly state whether they agree with the synthesis and provide their own revised answers/reasoning. "
            "Conditional Decision-Making: The Decision-maker finalizes the conclusion only when all domain agents reach a unanimous consensus (all agree) or when the maximum number of discussion rounds is reached. If consensus is not met, the system proceeds to the next round of collaboration. "
        )
    elif mas.lower() == 'reconcile':
        mas_description = (
            "The system begins with an initial analysis by initializing multiple general medical experts (agents are not hard-coded with specific specialties). "
            "It then proceeds to a multi-round discussion phase, which consists exclusively of a review process. "
            "In this stage, the agents (acting as reviewers) evaluate the generated answers, providing their reasoning and their own determination of the correct answer. "
            "The loop terminates if a consensus is reached among reviewers or if the maximum number of discussion rounds is exceeded. "
            "The final answer is determined through a weighted voting mechanism based on the confidence levels of the agents' outputs. "
        )
    # add the instruction
    user_text_prompt = (
        "Your current task is to perform open coding on the multi-agent collaboration failure modes observed in this case, "
        "based on the overview of the medical multi-agent system and its collaboration logs for answering medical questions. "
    )
    # add the case information to the prompt
    options = item['options']
    options_text = "\nOptions:\n"
    for key, value in options.items():
        options_text += f"{key}: {value}\n"
    user_text_prompt += (
        f"The case's question is: '{item['question']}'. "
        f"The question's answer options are: {options_text} "
        f"The ground truth answer is: '{item['ground_truth']}'. "
        f"The multi-agent system's predicted answer is: '{item['predicted_answer']}'. "
    )
    # if there is an image, add the information to the prompt
    if item.get('image_path'):
        user_text_prompt += "Image provided: Yes\n"

    # add the MAS description to the prompt
    user_text_prompt += f"The multi-agent system used in this case is {mas}. Here is a description of its workflow and collaboration process: {mas_description} "

    # add the collaboration log info
    case_history = item["case_history"]
    collaboration_text = gen_collaboration_text(case_history)
    user_text_prompt += f"The detailed multi-agent collaboration process is as follows: {collaboration_text} "

    # add the output format instruction
    user_text_prompt += (
        "Return a JSON object containing a list. If no failure is found, the list should be empty. "
        "Each item in the list must include: "
        "1. 'failure_mode': A concise phrase defining the specific error. "
        "2. 'explanation': The rationale for this classification. "
        "3. 'evidence': Direct quote or summary of the specific log content that proves this failure. "
        "Example Output:"
    )
    user_text_prompt += """
```json
[
  {
    "failure_mode": "Inconsistent Reasoning among Agents",
    "explanation": "...",
    "evidence": "..."
  },
  {
    "failure_mode": "Hallucinated Medical Guidelines",
    "explanation": "...",
    "evidence": "..."
  }
]
```
"""
    user_content = [
        {"type": "text", "text": user_text_prompt}
    ]
    # 检查是否有图片需要处理
    image_path = item.get("image_path")
    if image_path and os.path.exists(image_path):
        base64_image = encode_image(image_path)
        print(f"Encoded image for qid {item['qid']}: {image_path}")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
        user_message = {
            "role": "user",
            "content": user_content
        }
    else:
        if image_path:
             print(f"Warning: Image path does not exist for qid {item['qid']}: {image_path}")
        user_message = {"role": "user", "content": [{"type": "text", "text": user_text_prompt}]}
        
    return system_message, user_message



def open_coding(item: Dict[str, Any], model_key: str, config_path: str, mas: str) -> Optional[Dict[str, Any]]:
    # prepare the system message and user message for open-coding
    system_message, user_message = build_open_coding_prompts(item=item, mas = mas)
    # initialize the open-coding agent
    opencoder_agent = Opencoding(model_key=model_key, config_path=config_path)
    # call the open-coding function to get the annotation result
    opencoding_log = opencoder_agent.opencoding(system_message=system_message, user_message=user_message, qid=item["qid"])
    return opencoding_log

def main():
    parser = argparse.ArgumentParser(description="Run open-coding on multi-agent collaboration logs for medical question answering tasks.")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset name,like PathVQA,VQA-RAD")
    parser.add_argument("--config_path", type=Path, default=project_root / "config.toml",help="Path to the config.toml file")
    parser.add_argument("--mas", type=str, required=True, help="Specify the multi-agent system name, like ColaCare, MAC, HealthcareAgent, MDAgents, MedAgent, Reconcile")
    parser.add_argument("--llm", type=str, required=True, help="Specify the LLM used in the multi-agent system, like gpt-4.1, gpt-5.2")
    args = parser.parse_args()
    mas = args.mas
    dataset = args.dataset
    llm = args.llm
    config_path = args.config_path
    jsonl_file_path = project_root / "logs" / "extracted_logs_for_open_coding" / f"{mas}_{dataset}_{llm}_open_coding.jsonl"
    print(f"Loading data from {jsonl_file_path} for open-coding...")

    # adding logs
    terminal_log_file = project_root / "logs" / "open_coding_results" / f"{mas}_{dataset}_{llm}_terminal.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    output_file = project_root / "logs" / "open_coding_results" / f"{mas}_{dataset}_{llm}.jsonl"
    error_output_file = project_root / "logs" / "open_coding_results" / f"{mas}_{dataset}_{llm}_errors.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_output_file.parent.mkdir(parents=True, exist_ok=True)

    existing_qids = set()
    if output_file.exists():
        print(f"Output file {output_file} already exists. Appending new results.")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                try:
                    record = json.loads(line)
                    if "qid" in record:
                        existing_qids.add(record["qid"])
                except json.JSONDecodeError:
                    print("Warning: Found a corrupted line in jsonl file, skipping.")
        print(f"Found {len(existing_qids)} already processed cases. They will be skipped.")

    jsonl_file = load_jsonl(jsonl_file_path)
    for item in tqdm(jsonl_file, desc=f"Running open-coding on {dataset}-{mas}-{llm}"): 
        qid = item["qid"]
        if qid in existing_qids:
            print(f"Skipping {qid} - already processed")
            continue
        try:
            open_coding_result = open_coding(item = item, model_key="gpt-5.2", config_path=config_path, mas = mas)
            failure_lst = open_coding_result["parsed_output"]
            reasoning_content = open_coding_result["reasoning_content"]

            item_result = {
                "qid": qid,
                "timestamp": int(time.time()),
                "question": item["question"],
                "options": item.get("options"),
                "image_path": item.get("image_path"),
                "ground_truth": item.get("ground_truth"),
                "predicted_answer": item.get("predicted_answer"),
                "failure_modes": failure_lst,
                "reasoning_content": reasoning_content
            }

            save_jsonl(item_result, str(output_file))
            existing_qids.add(qid)

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            # Optionally, save an error log
            error_log = {
                "qid": qid,
                "error": str(e)
            }
            save_jsonl(error_log, str(error_output_file))

if __name__ == "__main__":
    main()