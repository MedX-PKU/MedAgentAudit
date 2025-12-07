"""
medagentboard/medqa/multi_agent_reconcile.py

This module implements the Reconcile framework for multi-model,
multi-agent discussion. Each agent generates an answer with step-by-step
reasoning and an estimated confidence level. Then, the agents engage in
multi-round discussions and a confidence-weighted vote produces the final team answer.
This version has been modified to include extensive logging for detailed analysis
of the multi-agent collaboration process, as requested for the WWW2026 research proposal.
"""

import os
import json
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import argparse
from tqdm import tqdm
import sys
import traceback
                    
# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import ColaCare utilities and analysis tools
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string
from medagentboard.utils.encode_image import encode_image
from medagentboard.utils.keu import KEU
from medagentboard.utils.analysishelper import AnalysisHelperLLM
from multi_agent_colacare_full_log_add_mechanism import AuditorAgent # We can reuse the AuditorAgent directly


class DiscussionPhase(Enum):
    """Enumeration of discussion phases in the Reconcile framework."""
    INITIAL = "initial"        # Initial answer generation
    DISCUSSION = "discussion"  # Multi-round discussion
    FINAL = "final"            # Final team answer


class ReconcileAgent:
    """
    An agent participating in the Reconcile framework.
    This version is enhanced for quantitative observation.
    """
    def __init__(self, agent_id: str, model_key: str):
        """
        Initialize a Reconcile agent.
        """
        self.agent_id = agent_id
        self.model_key = model_key
        self.discussion_history = []
        self.memory = []

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not configured in LLM_MODELS_SETTINGS")
        self.model_config = LLM_MODELS_SETTINGS[model_key]

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("OpenAI client is not installed. Please install it.") from e

        self.client = OpenAI(
            api_key=self.model_config["api_key"],
            base_url=self.model_config["base_url"],
        )
        self.model_name = self.model_config["model_name"]
        print(f"Initialized agent {self.agent_id} with model {self.model_name}")

    def call_llm(self, messages: List[Dict[str, Any]], max_retries: int = 3) -> Tuple[str, Dict[str, Any]]:
        """
        Call the LLM with the provided messages and a retry mechanism.
        Returns the raw response text and a detailed log of the call.
        """
        attempt = 0
        wait_time = 1

        call_log = {
            "llm_prompt": messages,
            "llm_raw_response": "",
            "error_message": ""
        }

        while attempt < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM with model {self.model_name} (attempt {attempt+1}/{max_retries})")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                response_text = completion.choices[0].message.content
                print(f"Agent {self.agent_id} received response: {response_text[:100]}...")

                call_log["llm_raw_response"] = response_text
                return response_text, call_log
            except Exception as e:
                attempt += 1
                error_details = f"Agent {self.agent_id} LLM call attempt {attempt}/{max_retries} failed: {e}"
                print(error_details)
                call_log["error_message"] = error_details
                if attempt < max_retries:
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

        print(f"Agent {self.agent_id} all LLM call attempts failed, returning default response")
        failed_response_text = json.dumps({
            "reasoning": "LLM call failed after multiple attempts",
            "answer": "",
            "confidence": 0.0,
            "keus": []
        })
        call_log["llm_raw_response"] = failed_response_text
        return failed_response_text, call_log

    def generate_initial_response(self,
                                question: str,
                                options: Optional[Dict[str, str]] = None,
                                image_path: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate an initial response, now including Key Evidential Units (KEUs).
        """
        print(f"Agent {self.agent_id} generating initial response")

        # MODIFICATION: System prompt now requires KEU extraction.
        system_message = {
            "role": "system",
            "content": (
                "You are a medical expert assistant. Analyze the following medical case. "
                "Your output must be a JSON object with four fields: "
                "1. 'reasoning': Your detailed step-by-step analysis. "
                "2. 'answer': Your final conclusion or selected option letter. "
                "3. 'confidence': Your confidence in the answer, a float between 0.0 and 1.0. "
                "4. 'keus': A list of key evidential units. Each KEU in the list must be a string representing a "
                "single, verifiable piece of evidence from the case (e.g., 'A 2cm nodule is visible in the upper "
                "left lung lobe.', 'The patient's white blood cell count is 15,000/µL.')."
            )
        }

        user_content = []
        if image_path and os.path.exists(image_path):
            try:
                base64_image = encode_image(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")

        question_text = question
        if options:
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_text = f"{question}\n\nOptions:\n{options_text}"

        prompt_text = (
            f"Case Description:\n{question_text}\n\n"
            "Please provide your complete analysis as a single, well-formatted JSON object "
            "strictly adhering to the four required fields: 'reasoning', 'answer', 'confidence', and 'keus'."
        )
        user_content.append({"type": "text", "text": prompt_text})
        user_message = {"role": "user", "content": user_content}

        messages = [system_message, user_message]
        response_text, call_log = self.call_llm(messages)
        result, parse_log = self._parse_response(response_text)

        step_log = {**call_log, **parse_log}
        self.memory.append({"phase": DiscussionPhase.INITIAL.value, "response": result})
        return result, step_log

    def generate_discussion_response(self,
                                  question: str,
                                  discussion_prompt: str,
                                  keu_list_text: str,
                                  ccp_text: str,
                                  current_round: int,
                                  options: Optional[Dict[str, str]] = None,
                                  image_path: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a discussion response, now with viewpoint attribution and KEU/CCP integration.
        """
        print(f"Agent {self.agent_id} generating discussion response for round {current_round}")

        own_previous_analysis = self.memory[-1]['response'] if self.memory else {}

        # MODIFICATION: System prompt now requires viewpoint attribution.
        system_message = {
            "role": "system",
            "content": (
                f"You are a medical expert participating in round {current_round} of a multi-agent discussion. "
                "Review the summary of your peers' opinions, your previous analysis, and the available evidence. "
                "Your task is to provide an updated, comprehensive analysis. Your output MUST be a JSON object, including: "
                "1. 'reasoning': Your detailed textual explanation for your current decision. This MUST reference relevant KEU-IDs. "
                "2. 'current_viewpoint': Your current final answer after this review (e.g., 'A', 'B'). "
                "3. 'confidence': Your updated confidence level (0.0 to 1.0). "
                "4. 'viewpoint_changed': A boolean (true/false) indicating if your 'current_viewpoint' is different from your previous answer. "
                "5. 'justification_type': A string, must be one of ['evidence_based', 'consensus_based']. "
                "Choose 'evidence_based' if your decision is primarily driven by specific KEU facts. "
                "Choose 'consensus_based' if you are primarily aligning with the group's opinion. "
                "6. 'cited_references': A list of strings containing the KEU-IDs or Agent-IDs that most influenced your decision."
            )
        }

        user_content = []
        if image_path and os.path.exists(image_path):
            try:
                base64_image = encode_image(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")

        question_text = question
        if options:
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            question_text = f"{question}\n\nOptions:\n{options_text}"
        
        own_analysis_text = (
            f"Your Previous Analysis (Answer: {own_previous_analysis.get('answer', 'N/A')}, "
            f"Confidence: {own_previous_analysis.get('confidence', 0.0):.2f}):\n"
            f"{own_previous_analysis.get('reasoning', 'No previous reasoning recorded.')}\n"
        )

        # MODIFICATION: User prompt now includes KEUs and CCPs.
        prompt_text = (
            f"Original Question:\n{question_text}\n\n"
            f"--- YOUR PREVIOUS ANALYSIS ---\n{own_analysis_text}\n"
            f"--- CURRENT DISCUSSION SUMMARY ---\n{discussion_prompt}\n\n"
            f"--- AVAILABLE EVIDENCE & CONFLICTS ---\n"
            f"Key Evidential Units (KEUs):\n{keu_list_text}\n\n"
            f"Critical Conflict Points (CCPs) to Address:\n{ccp_text if ccp_text else 'None identified.'}\n\n"
            f"--- YOUR TASK ---\n"
            f"Note that potential conflicts (CCPs) have been identified; a robust synthesis must acknowledge or resolve these points.\n\n"
            f"Based on all the information above, provide your updated analysis. Your 'reasoning' MUST reference "
            f"**CRITICAL INSTRUCTION:** Your task is to synthesize these diverse opinions into a single, coherent analysis. In your 'explanation', you **MUST selectively cite only the most important KEU-IDs** that support your synthesized view. **DO NOT simply list all available KEUs.** Your goal is to demonstrate a deep understanding by building a new, consolidated argument from the strongest evidence (e.g., 'Synthesizing the specialists' views, the consensus leans towards X, primarily supported by the crucial findings in KEU-2 and KEU-5...').\n\n "
            f"Your response MUST be a single JSON object, strictly adhering to the 6-field structure defined in your system instructions."
        )
        user_content.append({"type": "text", "text": prompt_text})
        user_message = {"role": "user", "content": user_content}
        
        messages = [system_message, user_message]
        response_text, call_log = self.call_llm(messages)
        result, parse_log = self._parse_response(response_text, is_discussion=True)

        step_log = {**call_log, **parse_log}
        step_log["discussion_context_provided"] = discussion_prompt

        self.memory.append({
            "phase": DiscussionPhase.DISCUSSION.value,
            "round": current_round,
            "response": result
        })
        return result, step_log

    def _parse_response(self, response_text: str, is_discussion: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Parse the LLM response into a structured format, handling both initial and discussion structures.
        """
        parse_log = {"parsing_success": False, "parsing_method": "json.loads"}
        try:
            result = json.loads(preprocess_response_string(response_text))

            # --- Field validation and normalization ---
            if "reasoning" not in result: result["reasoning"] = "No reasoning provided"
            
            # Unify answer field naming
            if is_discussion:
                if "current_viewpoint" in result: result["answer"] = result.pop("current_viewpoint")
                elif "answer" not in result: result["answer"] = ""
            elif "answer" not in result:
                result["answer"] = ""

            try:
                result["confidence"] = float(result.get("confidence", 0.0))
                result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            except (ValueError, TypeError):
                result["confidence"] = 0.0

            if not is_discussion:
                if "keus" not in result or not isinstance(result["keus"], list): result["keus"] = []
            else: # Discussion-specific fields
                if "viewpoint_changed" not in result: result["viewpoint_changed"] = False
                if "justification_type" not in result: result["justification_type"] = "unknown"
                if "cited_references" not in result: result["cited_references"] = []

            parse_log["parsing_success"] = True
            return result, parse_log
        except json.JSONDecodeError:
            print(f"Agent {self.agent_id} failed to parse JSON response: {response_text[:100]}...")
            parse_log["parsing_method"] = "fallback_manual_extraction"
            # Fallback for critical fields if JSON parsing fails
            return {"reasoning": response_text, "answer": "", "confidence": 0.0}, parse_log


class ReconcileCoordinator:
    """
    The coordinator for the Reconcile framework, enhanced with auditing capabilities.
    """
    def __init__(self, agent_configs: List[Dict[str, str]], max_rounds: int = 3, auditor_model_key: str = "gemini-2.5-pro", conflict_analysis_model_key: str = "gemini-2.5-pro"):
        """
        Initialize the Reconcile coordinator with auditing agents.
        """
        self.agents = [ReconcileAgent(cfg["agent_id"], cfg["model_key"]) for cfg in agent_configs]
        self.max_rounds = max_rounds
        # MODIFICATION: Instantiate Auditor and Analysis agents
        self.auditor_agent = AuditorAgent("auditor", auditor_model_key)
        self.analysis_llm = AnalysisHelperLLM(model_key=conflict_analysis_model_key)
        print(f"Initialized ReconcileCoordinator with {len(self.agents)} agents, max_rounds={max_rounds}")

    def _group_answers(self, answers: List[Dict[str, Any]]) -> str:
        # (This method remains unchanged from the original file)
        groups = {}
        for ans in answers:
            answer_value = ans.get("answer", "")
            if not isinstance(answer_value, str): answer_value = str(answer_value)
            answer_text = answer_value.strip().lower()
            if not answer_text: continue
            
            confidence = ans.get("confidence", 0.0)
            if answer_text not in groups:
                groups[answer_text] = {"count": 0, "explanations": [], "confidence_sum": 0.0}
            groups[answer_text]["count"] += 1
            groups[answer_text]["explanations"].append(ans.get("reasoning", ""))
            groups[answer_text]["confidence_sum"] += confidence

        grouped_str = ""
        for ans_text, data in sorted(groups.items(), key=lambda item: item[1]['count'], reverse=True):
            avg_confidence = data["confidence_sum"] / data["count"] if data["count"] > 0 else 0
            grouped_str += f"Answer Option: '{ans_text.upper()}'\n"
            grouped_str += f"- Supporters: {data['count']}\n"
            grouped_str += f"- Average Confidence: {avg_confidence:.2f}\n"
            grouped_str += "- Supporting Reasons:\n"
            for i, exp in enumerate(data["explanations"]):
                exp_short = (exp[:400] + '...') if len(exp) > 400 else exp
                grouped_str += f"  - Reason {i+1}: {exp_short}\n"
            grouped_str += "\n"
        return grouped_str.strip()

    def _recalibrate(self, confidence: float) -> float:
        # (This method remains unchanged)
        if confidence == 1.0: return 1.0
        elif confidence >= 0.9: return 0.8
        elif confidence >= 0.8: return 0.5
        elif confidence > 0.6: return 0.3
        else: return 0.1

    def _weighted_vote(self, answers: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        # (This method remains unchanged, but benefits from better logging)
        vote_weights = {}
        for ans in answers:
            answer_value = ans.get("answer", "")
            if not isinstance(answer_value, str): answer_value = str(answer_value)
            answer = answer_value.strip()
            if not answer: continue
            
            confidence = ans.get("confidence", 0.0)
            weight = self._recalibrate(confidence)
            key = answer.lower()
            if key not in vote_weights:
                vote_weights[key] = {"weight": 0, "original": answer, "supporters": 0}
            vote_weights[key]["weight"] += weight
            vote_weights[key]["supporters"] += 1

        if not vote_weights:
            return "", {"vote_weights": {}, "winner": "No valid answer", "note": "All agents provided empty answers."}

        winner_key = max(vote_weights, key=lambda k: vote_weights[k]["weight"])
        final_decision = vote_weights[winner_key]["original"]
        voting_log = {"vote_weights": vote_weights, "winner_key": winner_key, "final_decision": final_decision}
        return final_decision, voting_log

    def _check_consensus(self, answers: List[Dict[str, Any]]) -> bool:
        # (This method remains unchanged)
        valid_answers = []
        for ans in answers:
            answer_value = ans.get("answer", "")
            if not isinstance(answer_value, str): answer_value = str(answer_value)
            answer_text = answer_value.strip()
            if answer_text: valid_answers.append(answer_text.lower())
        return len(valid_answers) > 0 and len(set(valid_answers)) == 1

    def run_discussion(self,
                      qid: str,
                      question: str,
                      options: Optional[Dict[str, str]] = None,
                      image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete discussion process with integrated quantitative observation mechanisms.
        """
        print(f"Starting discussion with {len(self.agents)} agents on question: {question}")
        start_time = time.time()

        # MODIFICATION: Initialize audit_trail and other tracking variables
        discussion_history = []
        audit_trail = {
            "keus": {},
            "viewpoints": {agent.agent_id: [] for agent in self.agents},
            "collaboration_audits": {},
            "ccps": {}
        }
        keu_counter = 0
        ccp_counter = 0
        all_unresolved_ccps = []

        # === Phase 1: Initial responses ===
        print("Phase 1: Generating initial responses")
        current_answers = []
        initial_contributions_for_ccp = []
        for agent in self.agents:
            resp, step_log = agent.generate_initial_response(question, options, image_path)
            current_answers.append(resp)
            
            # --- Mechanism 1 & 3: Audit initial response ---
            explanation = resp.get("reasoning", "")
            contribution_audit = self.auditor_agent.audit_domain_agent_contribution(question, agent.agent_id, "Specialist", explanation) # Reconcile has no specialties
            risk_audit = self.auditor_agent.audit_risk_and_quality(agent.agent_id, explanation)
            audit_trail["collaboration_audits"][f"round_0_initial_{agent.agent_id}"] = {**contribution_audit, **risk_audit}

            # --- Mechanism 1: Collect KEUs ---
            for keu_content in resp.get("keus", []):
                keu_id = f"KEU-{keu_counter}"
                new_keu = KEU(keu_id, keu_content, agent.agent_id, 0)
                audit_trail["keus"][keu_id] = new_keu
                keu_counter += 1

            # --- Mechanism 2: Log initial viewpoint ---
            initial_viewpoint = {
                "step": "round_0_initial",
                "viewpoint": resp.get("answer"),
                "viewpoint_changed": False,
                "justification_type": "initial_analysis",
                "cited_references": [f"KEU-{i}" for i in range(keu_counter - len(resp.get("keus", [])), keu_counter)]
            }
            audit_trail["viewpoints"][agent.agent_id].append(initial_viewpoint)
            
            # --- For Mechanism 4 ---
            full_text = explanation + "\nKey Evidential Units:\n- " + "\n- ".join(resp.get("keus", []))
            initial_contributions_for_ccp.append({'agent_id': agent.agent_id, 'text': full_text})

            discussion_history.append({
                "phase": DiscussionPhase.INITIAL.value,
                "agent_id": agent.agent_id, "model_name": agent.model_name,
                "response": resp, "interaction_log": step_log
            })
            print(f"Agent {agent.agent_id} initial answer: {resp.get('answer', '')} (conf: {resp.get('confidence', 0.0):.2f})")

        # --- Mechanism 1 & 4 after all initial responses ---
        all_keus_for_audit = [{"keu_id": k, "content": v.content} for k, v in audit_trail["keus"].items()]
        if all_keus_for_audit:
            key_status_map = self.auditor_agent.identify_key_evidential_units(question, [], self.agents, all_keus_for_audit)
            for keu_id, is_key in key_status_map.items():
                if keu_id in audit_trail["keus"]: audit_trail["keus"][keu_id].is_key = is_key

        initial_ccps = self.auditor_agent.identify_critical_conflicts(initial_contributions_for_ccp, "initial analyses")
        audit_trail["ccps"][0] = []
        for ccp in initial_ccps:
            ccp.update({'ccp_id': f"CCP-{ccp_counter}", 'round_identified': 0, 'status': 'unresolved', 'round_resolved': None})
            audit_trail["ccps"][0].append(ccp)
            ccp_counter += 1
        all_unresolved_ccps.extend(audit_trail["ccps"][0])

        # === Phase 2: Multi-round discussion ===
        round_num = 0
        consensus_reached = False
        while round_num < self.max_rounds and not consensus_reached:
            round_num += 1
            print(f"Phase 2: Discussion round {round_num}/{self.max_rounds}")

            # --- Prepare prompts with KEU and CCP context ---
            discussion_prompt = self._group_answers(current_answers)
            keu_list_text = "\n".join([f"- {k}: '{v.content}' (from {v.source_agent})" for k,v in audit_trail["keus"].items()])
            ccp_text_for_prompt = "\n".join([f"- {c['ccp_id']}: {c['conflict_summary']} (Involves: {', '.join(c['conflicting_agents'])})" for c in all_unresolved_ccps]) if all_unresolved_ccps else ""

            new_answers = []
            discussion_contributions_for_ccp = []
            for agent in self.agents:
                resp, step_log = agent.generate_discussion_response(
                    question, discussion_prompt, keu_list_text, ccp_text_for_prompt, round_num, options, image_path
                )
                new_answers.append(resp)
                reasoning = resp.get("reasoning", "")
                
                # --- Mechanism 1, 2, 3 Audits ---
                contribution_audit = self.auditor_agent.audit_domain_agent_contribution(question, agent.agent_id, "Specialist", reasoning)
                risk_audit = self.auditor_agent.audit_risk_and_quality(agent.agent_id, reasoning)
                audit_trail["collaboration_audits"][f"round_{round_num}_discussion_{agent.agent_id}"] = {**contribution_audit, **risk_audit}
                
                viewpoint_entry = {
                    "step": f"round_{round_num}_discussion", "viewpoint": resp.get("answer"),
                    "viewpoint_changed": resp.get("viewpoint_changed", False),
                    "justification_type": resp.get("justification_type", "unknown"),
                    "cited_references": resp.get("cited_references", [])
                }
                audit_trail["viewpoints"][agent.agent_id].append(viewpoint_entry)
                
                for keu_id, keu in audit_trail["keus"].items():
                    if keu_id in reasoning or keu.content in reasoning:
                        keu.cited_by.append({"agent_id": agent.agent_id, "round": round_num, "action": "discussion"})
                
                discussion_contributions_for_ccp.append({'agent_id': agent.agent_id, 'text': reasoning})

                discussion_history.append({
                    "phase": DiscussionPhase.DISCUSSION.value, "round": round_num,
                    "agent_id": agent.agent_id, "model_name": agent.model_name,
                    "response": resp, "interaction_log": step_log
                })
                print(f"Agent {agent.agent_id} round {round_num} answer: {resp.get('answer', '')} (conf: {resp.get('confidence', 0.0):.2f})")

            # --- Mechanism 3: Pre-vote quality audit for the final round ---
            if round_num == self.max_rounds or self._check_consensus(new_answers):
                 # --- START OF FIX ---
                 # 为每个智能体创建一个占位符“专科”列表
                 dummy_specialties = ["Medical Expert"] * len(new_answers)
                 
                 # 确保传递给审计函数的数据包含agent_id
                 # 注意：new_answers 是一个 response 列表，我们需要 agent_id
                 # 从 discussion_history 的最新一轮中获取 agent_id
                 final_round_reviews = []
                 history_this_round = [h for h in discussion_history if h.get("round") == round_num]
                 for agent_resp in history_this_round:
                     review_data = agent_resp['response']
                     review_data['agent_id'] = agent_resp['agent_id'] # 将 agent_id 注入
                     final_round_reviews.append(review_data)

                 overall_quality_audit = self.auditor_agent.audit_overall_quality_for_decision(
                     question, 
                     final_round_reviews, # <--- 传递包含 agent_id 的 review 列表
                     dummy_specialties    # <--- 传递正确长度的占位符列表
                 )
                 audit_trail["collaboration_audits"][f"round_{round_num}_pre_vote_quality"] = overall_quality_audit

            # --- Mechanism 4: Conflict Resolution Tracking ---
            resolved_this_round_log = []
            still_unresolved = []
            current_discussion_text = self._group_answers(new_answers)
            for ccp in all_unresolved_ccps:
                was_addressed, resolution_reasoning = self.analysis_llm.check_if_conflict_was_addressed(ccp, current_discussion_text)
                if was_addressed:
                    ccp.update({'status': 'resolved', 'round_resolved': round_num, 'resolution_reasoning': resolution_reasoning})
                    resolved_this_round_log.append(ccp)
                else:
                    still_unresolved.append(ccp)
            all_unresolved_ccps = still_unresolved
            
            new_ccps = self.auditor_agent.identify_critical_conflicts(discussion_contributions_for_ccp, f"round {round_num} discussion")
            audit_trail["ccps"][round_num] = []
            for ccp in new_ccps:
                ccp.update({'ccp_id': f"CCP-{ccp_counter}", 'round_identified': round_num, 'status': 'unresolved', 'round_resolved': None})
                audit_trail["ccps"][round_num].append(ccp)
                ccp_counter += 1
            all_unresolved_ccps.extend(new_ccps)
            
            current_answers = new_answers
            consensus_reached = self._check_consensus(current_answers)
            if consensus_reached:
                print("Consensus reached, ending discussion.")
                break

        # === Phase 3: Final team answer via weighted vote ===
        print("Phase 3: Generating final team answer")
        final_decision, voting_log = self._weighted_vote(current_answers)
        
        discussion_history.append({
            "phase": DiscussionPhase.FINAL.value,
            "final_decision": final_decision, "consensus_reached": consensus_reached,
            "rounds_completed": round_num, "final_round_agent_answers": current_answers,
            "voting_details": voting_log
        })

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Discussion completed in {processing_time:.2f} seconds. Final answer: {final_decision}")

        # Finalize and serialize audit_trail for saving
        if "keus" in audit_trail and audit_trail["keus"]:
            audit_trail["keus"] = {keu_id: keu.to_dict() for keu_id, keu in audit_trail["keus"].items()}

        return {
            "final_decision": final_decision,
            "discussion_history": discussion_history,
            "processing_time": processing_time,
            "audit_trail": audit_trail, # MODIFICATION: Add the complete audit trail
        }


def process_item(item: Dict[str, Any],
               agent_configs: List[Dict[str, str]],
               max_rounds: int = 3,
               auditor_model_key: str = "gemini-2.5-pro",
               conflict_analysis_model_key: str = "gemini-2.5-pro") -> Dict[str, Any]:
    """
    Process a single QA item with the Reconcile framework.
    """
    qid = item.get("qid", "unknown")
    print(f"Processing item {qid}")

    coordinator = ReconcileCoordinator(agent_configs, max_rounds, auditor_model_key, conflict_analysis_model_key)
    discussion_result = coordinator.run_discussion(
        qid=qid,
        question=item.get("question", ""),
        options=item.get("options"),
        image_path=item.get("image_path")
    )
    
    result = {
        "qid": qid,
        "timestamp": int(time.time()),
        "question": item.get("question", ""),
        "options": item.get("options"),
        "image_path": item.get("image_path"),
        "ground_truth": item.get("answer"),
        "predicted_answer": discussion_result["final_decision"],
        "case_history": discussion_result,
    }
    return result


def main():
    """
    Main entry point for running the Reconcile framework with full observation logging.
    """
    parser = argparse.ArgumentParser(description="Run the Reconcile framework on medical QA datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--qa_type", type=str, choices=["mc", "ff"], default="mc", help="QA type: multiple-choice (mc) or free-form (ff)")
    parser.add_argument("--agents", nargs='+', required=True, help="List of agent model keys")
    parser.add_argument("--start", type=int, required=True, help="Starting index for samples to run")
    parser.add_argument("--end", type=int, required=True, help="Ending index for samples to run")
    parser.add_argument("--max_rounds", type=int, default=3, help="Maximum number of discussion rounds")
    # MODIFICATION: Add arguments for auditor and conflict models
    parser.add_argument("--auditor_model", type=str, default="gemini-2.5-pro", help="Model for the AuditorAgent.")
    parser.add_argument("--conflict_model", type=str, default="gemini-2.5-pro", help="Model for conflict analysis (AnalysisHelperLLM).")
    args = parser.parse_args()

    method = f"ReConcile_full_log_{time.strftime('%Y%m%d_%H%M%S')}"
    logs_dir = os.path.join("logs", "observation", "ReConcile", args.dataset)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Logs will be saved to: {logs_dir}")

    data_path = f"./my_datasets/processed/medqa/{args.dataset}/medqa_{args.qa_type}_test.json"
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    agent_configs = [{"agent_id": f"agent_{idx}", "model_key": model_key} for idx, model_key in enumerate(args.agents, 1)]
    print(f"Configured {len(agent_configs)} agents: {[cfg['model_key'] for cfg in agent_configs]}")

    for item in tqdm(data[args.start:args.end], desc=f"Processing {args.dataset} ({args.qa_type})"):
        qid = item.get("qid")
        result_path = os.path.join(logs_dir, f"{qid}-result.json")

        if os.path.exists(result_path):
            print(f"Skipping {qid} (already processed)")
            continue

        try:
            result = process_item(
                item,
                agent_configs,
                args.max_rounds,
                auditor_model_key=args.auditor_model,
                conflict_analysis_model_key=args.conflict_model
            )
            save_json(result, result_path)
        except Exception as e:
            # 1. 获取完整的错误堆栈信息字符串
            error_traceback = traceback.format_exc()
            
            # 2. 打印到终端 (使用 flush=True 确保立即显示)
            print(f"--- FATAL ERROR processing item {qid} ---", flush=True)
            print(error_traceback, flush=True)
            print(f"--- END OF ERROR for {qid} ---", flush=True)

            # 3. 将完整的堆栈信息保存到 JSON 文件中，方便事后分析
            error_log = {
                "qid": qid,
                "error_type": str(type(e).__name__), # e.g., "IndexError"
                "error_message": str(e),            # e.g., "list index out of range"
                "traceback": error_traceback        # The full stack trace
            }
            save_json(error_log, os.path.join(logs_dir, f"{qid}-error.json"))

if __name__ == "__main__":
    main()