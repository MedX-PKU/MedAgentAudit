"""
./scripts/case_study_filter.py

Select appendix case studies from the 400-case audit human-evaluation pool.

Selection follows the paper method "Case Study Selection and Qualitative Analysis":
1. Keep only cases where the automated audit label matches the 3-rater human majority.
2. Prefer cases with complete interaction traces around the audited failure step.
3. Preserve the required 10 failure modes x 2 labels stratification.
4. Encourage modality and architecture diversity across the final 20 selected cases.
5. For positive cases, prefer incorrect final diagnoses with stronger inter-agent consensus.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from medagentaudit.utils.json_utils import load_json, load_jsonl, save_json
from medagentaudit.utils.logger import DualLogger

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]
sys.path.append(str(project_root))



EXTRACTED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR = (
    project_root / "logs" / "extracted_logs_for_audit_human_evaluation"
)
HUMAN_EVAL_AUDIT_RESULTS_DIR = project_root / "logs" / "human_eval_results" / "audit_results"
CASE_STUDY_SELECTION_DIR = project_root / "logs" / "case_study_selection"
CASE_STUDY_SELECTION_DIR.mkdir(parents=True, exist_ok=True)


FAILURE_MODE_STATUS_KEY_MAPPING = {
    "1.1.1": "factual_hallucination_status",
    "1.2.1": "modality_neglect_status",
    "2.1.1": "role_task_alignment",
    "2.1.2": "knowledge_activation_status",
    "2.2.1": "interaction_redundancy",
    "2.2.2": "conflict_resolution_status",
    "3.1.1": "suppression_status",
    "3.1.2": "authority_bias_status",
    "3.1.3": "neglect_of_conflict_status",
    "3.2.1": "inter_round_consistency_status",
}

FAILURE_MODE_LOG_KEY_MAPPING = {
    "1.1.1": "1_1_1_factual_hallucination",
    "1.2.1": "1_2_1_neglect_or_misinterpretation_of_modality_info",
    "2.1.1": "2_1_1_role_assignment",
    "2.1.2": "2_1_2_domain_specific_knowledge_activation",
    "2.2.1": "2_2_1_repetition_of_initial_views",
    "2.2.2": "2_2_2_unresolved_conflicts",
    "3.1.1": "3_1_1_suppression_of_minority_views",
    "3.1.2": "3_1_2_authority_bias",
    "3.1.3": "3_1_3_neglect_of_contradictions",
    "3.2.1": "3_2_1_self_contradiction_when_decision",
}

FAILURE_MODE_ORDER = [
    "1.1.1",
    "1.2.1",
    "2.1.1",
    "2.1.2",
    "2.2.1",
    "2.2.2",
    "3.1.1",
    "3.1.2",
    "3.1.3",
    "3.2.1",
]

TEXT_DATASETS = {"medqa", "pubmedqa", "medxpertqa-text"}
STAGE_ORDER = {"role_assignment": 0, "analysis": 1, "synthesis": 2, "review": 3, "decision": 4}

MAS_ARCHITECTURE_FAMILY = {
    "colacare": "static_pipeline",
    "healthcareagent": "static_pipeline",
    "mdagents": "dynamic_role_assignment",
    "medagent": "dynamic_role_assignment",
    "reconcile": "decentralized_voting",
    "mac": "iterative_debate",
}

REQUIRED_MODALITIES = {"text", "visual"}
REQUIRED_ARCHITECTURE_FAMILIES = {
    "dynamic_role_assignment",
    "static_pipeline",
    "decentralized_voting",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_agent_id(value: Any) -> str:
    return normalize_text(value).lower()


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    answer = normalize_text(value)
    if len(answer) == 1:
        return answer.upper()
    return answer.lower()


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_case_pool_filename(filepath: Path) -> tuple[str, str]:
    parts = filepath.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected candidate file name: {filepath.name}")
    failure_code = parts[0]
    label = "1" if parts[1] == "positive" else "0"
    return failure_code, label


def build_uid(qid: str, dataset: str, mas: str, llm: str, failure_code: str) -> str:
    return "||".join(
        [
            normalize_text(qid).lower(),
            normalize_text(dataset).lower(),
            normalize_text(mas).lower(),
            normalize_text(llm).lower(),
            normalize_text(failure_code),
        ]
    )


def load_human_votes() -> dict[str, list[int]]:
    votes = defaultdict(list)
    annotation_files = sorted(HUMAN_EVAL_AUDIT_RESULTS_DIR.glob("*.json"))
    print(f"Found {len(annotation_files)} audit annotation files in {HUMAN_EVAL_AUDIT_RESULTS_DIR}")

    for annotation_file in annotation_files:
        annotations = load_json(annotation_file).get("annotations", {})
        print(f"  - {annotation_file.name}: {len(annotations)} annotations")
        for annotation in annotations.values():
            uid = build_uid(
                qid=annotation["caseId"],
                dataset=annotation["dataset"],
                mas=annotation["mas"],
                llm=annotation["llm"],
                failure_code=annotation["taxonomyKey"],
            )
            vote = 1 if normalize_text(annotation.get("verdict")).lower() == "yes" else 0
            votes[uid].append(vote)
    return dict(votes)


def load_candidate_pool() -> list[dict[str, Any]]:
    candidate_files = sorted(EXTRACTED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR.glob("*.jsonl"))
    candidates = []
    print(f"Found {len(candidate_files)} candidate files in {EXTRACTED_LOGS_FOR_AUDIT_HUMAN_EVAL_DIR}")

    for candidate_file in candidate_files:
        failure_code, label = parse_case_pool_filename(candidate_file)
        records = load_jsonl(candidate_file)
        print(
            f"  - {candidate_file.name}: {len(records)} records "
            f"(failure_code={failure_code}, label={label})"
        )
        for record in records:
            enriched = dict(record)
            enriched["failure_code"] = failure_code
            enriched["mas_audit_result"] = label
            enriched["_source_file"] = candidate_file.name
            candidates.append(enriched)
    return candidates


def extract_output_dict(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    if isinstance(item.get("log"), dict):
        parsed_output = item["log"].get("parsed_output", {})
        if isinstance(parsed_output, dict):
            return parsed_output
    parsed_output = item.get("parsed_output", {})
    if isinstance(parsed_output, dict):
        return parsed_output
    return {}


def iter_interaction_events(case_history: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for round_obj in case_history.get("rounds", []):
        round_num = round_obj.get("round", 0)

        for idx, opinion in enumerate(ensure_list(round_obj.get("opinions"))):
            parsed = extract_output_dict(opinion)
            events.append(
                {
                    "round": round_num,
                    "step": "analysis",
                    "agent_id": normalize_agent_id(opinion.get("agent_id")),
                    "specialty": opinion.get("specialty"),
                    "answer": normalize_answer(parsed.get("answer")),
                    "sort_key": (round_num, STAGE_ORDER["analysis"], idx),
                }
            )

        for idx, synthesis in enumerate(ensure_list(round_obj.get("synthesis"))):
            parsed = extract_output_dict(synthesis)
            events.append(
                {
                    "round": round_num,
                    "step": "synthesis",
                    "agent_id": normalize_agent_id(synthesis.get("agent_id", "meta")),
                    "specialty": synthesis.get("specialty"),
                    "answer": normalize_answer(parsed.get("answer")),
                    "sort_key": (round_num, STAGE_ORDER["synthesis"], idx),
                }
            )

        for idx, review in enumerate(ensure_list(round_obj.get("reviews"))):
            parsed = extract_output_dict(review)
            events.append(
                {
                    "round": round_num,
                    "step": "review",
                    "agent_id": normalize_agent_id(review.get("agent_id")),
                    "specialty": review.get("specialty"),
                    "answer": normalize_answer(parsed.get("answer")),
                    "sort_key": (round_num, STAGE_ORDER["review"], idx),
                }
            )

        for idx, decision in enumerate(ensure_list(round_obj.get("decision"))):
            parsed = extract_output_dict(decision)
            events.append(
                {
                    "round": round_num,
                    "step": "decision",
                    "agent_id": normalize_agent_id(decision.get("agent_id", "decision-maker")),
                    "specialty": decision.get("specialty"),
                    "answer": normalize_answer(parsed.get("answer")),
                    "sort_key": (round_num, STAGE_ORDER["decision"], idx),
                }
            )

    events.sort(key=lambda event: event["sort_key"])
    return events


def find_failure_anchor(case_history: dict[str, Any], failure_code: str) -> dict[str, Any] | None:
    audit_key = FAILURE_MODE_LOG_KEY_MAPPING[failure_code]
    status_key = FAILURE_MODE_STATUS_KEY_MAPPING[failure_code]

    for audit_round in case_history.get("audit", {}).get("rounds", []):
        entries = ensure_list(audit_round.get(audit_key))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            audit_result = entry.get("audit_result", {})
            if status_key not in audit_result:
                continue
            return {
                "round": audit_round.get("round"),
                "step": entry.get("step"),
                "agent_id": normalize_agent_id(entry.get("agent_id")),
                "specialty": entry.get("specialty"),
                "specialties": entry.get("specialties", []),
                "status": normalize_text(audit_result.get(status_key)),
                "audit_reasoning": normalize_text(audit_result.get("auditor_reasoning")),
            }
    return None


def resolve_anchor_index(events: list[dict[str, Any]], anchor: dict[str, Any] | None) -> int | None:
    if anchor is None:
        return None

    step = anchor.get("step")
    if step == "role_assignment":
        return -1

    target_round = anchor.get("round")
    target_step = anchor.get("step")
    target_agent = anchor.get("agent_id")

    for idx, event in enumerate(events):
        if (
            event["round"] == target_round
            and event["step"] == target_step
            and target_agent
            and event["agent_id"] == target_agent
        ):
            return idx

    for idx, event in enumerate(events):
        if event["round"] == target_round and event["step"] == target_step:
            return idx

    return None


def compute_consensus_metrics(case_history: dict[str, Any], predicted_answer: str) -> dict[str, Any]:
    latest_answers = {}

    for round_obj in case_history.get("rounds", []):
        for opinion in ensure_list(round_obj.get("opinions")):
            agent_id = normalize_agent_id(opinion.get("agent_id"))
            answer = normalize_answer(extract_output_dict(opinion).get("answer"))
            if agent_id and answer:
                latest_answers[agent_id] = answer

        for review in ensure_list(round_obj.get("reviews")):
            agent_id = normalize_agent_id(review.get("agent_id"))
            answer = normalize_answer(extract_output_dict(review).get("answer"))
            if agent_id and answer:
                latest_answers[agent_id] = answer

    if not latest_answers:
        for round_obj in case_history.get("rounds", []):
            for synthesis in ensure_list(round_obj.get("synthesis")):
                agent_id = normalize_agent_id(synthesis.get("agent_id", "meta"))
                answer = normalize_answer(extract_output_dict(synthesis).get("answer"))
                if agent_id and answer:
                    latest_answers[agent_id] = answer

    normalized_prediction = normalize_answer(predicted_answer)
    answer_agent_count = len(latest_answers)
    support_count = sum(
        1 for answer in latest_answers.values() if answer == normalized_prediction
    )
    consensus_rate = support_count / answer_agent_count if answer_agent_count else 0.0
    consensus_strength = (
        consensus_rate * min(answer_agent_count, 3) / 3 if answer_agent_count else 0.0
    )

    return {
        "latest_answers": latest_answers,
        "answer_agent_count": answer_agent_count,
        "support_count": support_count,
        "consensus_rate": round(consensus_rate, 4),
        "consensus_strength": round(consensus_strength, 4),
    }


def classify_modality(dataset: str) -> str:
    return "text" if normalize_text(dataset).lower() in TEXT_DATASETS else "visual"


def classify_architecture_family(mas: str) -> str:
    return MAS_ARCHITECTURE_FAMILY.get(normalize_text(mas).lower(), "other")


def compute_base_score(candidate: dict[str, Any]) -> float:
    score = 0.0

    if not candidate["human_ai_agreement"]:
        return -1_000.0

    score += 18.0
    score += 2.5 * max(candidate["human_agreement_strength"] - 2, 0)

    if candidate["log_integrity_ok"]:
        score += 8.0
    if candidate["has_complete_input"]:
        score += 4.0
    if candidate["has_terminal_output"]:
        score += 3.0

    score += min(candidate["num_rounds"], 3) * 1.8
    score += min(candidate["distinct_stage_count"], 4) * 1.2
    score += min(candidate["unique_agent_count"], 6) * 0.8
    score += min(candidate["events_before_anchor"], 6) * 0.5
    score += min(candidate["events_after_anchor"], 6) * 0.8
    score += min(candidate["peer_response_count"], 4) * 0.8

    if candidate["label"] == "1":
        if not candidate["final_prediction_correct"]:
            score += 6.0
        score += 6.0 * candidate["consensus_strength"]
    else:
        if candidate["final_prediction_correct"]:
            score += 1.5

    if candidate["failure_step"] == "role_assignment" and candidate["assigned_specialty_count"] >= 2:
        score += 1.5

    if candidate["failure_step"] in {"synthesis", "decision"} and candidate["events_before_anchor"] > 0:
        score += 1.5

    return round(score, 4)


def build_selection_reason(candidate: dict[str, Any]) -> str:
    reason_parts = []

    if candidate["human_agreement_strength"] == 3:
        reason_parts.append("unanimous human-AI agreement")
    else:
        reason_parts.append("majority human-AI agreement")

    if candidate["log_integrity_ok"]:
        reason_parts.append("complete interaction trace")

    if candidate["label"] == "1":
        if not candidate["final_prediction_correct"]:
            reason_parts.append("incorrect final diagnosis")
        if candidate["consensus_strength"] >= 0.66:
            reason_parts.append("high inter-agent consensus")

    reason_parts.append(candidate["modality"])
    reason_parts.append(candidate["architecture_family"])

    return "; ".join(reason_parts)


def enrich_candidate(record: dict[str, Any], human_votes: dict[str, list[int]]) -> dict[str, Any]:
    uid = build_uid(
        qid=record["qid"],
        dataset=record["dataset"],
        mas=record["mas"],
        llm=record["llm"],
        failure_code=record["failure_code"],
    )

    votes = human_votes.get(uid, [])
    majority_vote = 1 if votes and sum(votes) >= len(votes) / 2 else 0
    ai_label = int(record["mas_audit_result"])
    human_ai_agreement = bool(votes) and majority_vote == ai_label
    human_agreement_strength = sum(1 for vote in votes if vote == ai_label)

    case_history = record.get("case_history", {})
    events = iter_interaction_events(case_history)
    anchor = find_failure_anchor(case_history, record["failure_code"])
    anchor_index = resolve_anchor_index(events, anchor)

    if anchor_index is None:
        events_before_anchor = 0
        events_after_anchor = 0
        peer_response_count = 0
    elif anchor_index == -1:
        events_before_anchor = 0
        events_after_anchor = len(events)
        peer_response_count = len({event["agent_id"] for event in events if event["agent_id"]})
    else:
        failure_agent = anchor.get("agent_id", "")
        after_events = events[anchor_index + 1 :]
        events_before_anchor = anchor_index
        events_after_anchor = len(after_events)
        peer_response_count = len(
            {
                event["agent_id"]
                for event in after_events
                if event["agent_id"] and event["agent_id"] != failure_agent
            }
        )

    stage_types = {event["step"] for event in events}
    unique_agents = {event["agent_id"] for event in events if event["agent_id"]}

    has_complete_input = bool(normalize_text(record.get("question"))) and bool(record.get("options"))
    has_terminal_output = bool(normalize_text(record.get("predicted_answer")))
    has_decision_stage = any(event["step"] == "decision" for event in events)

    failure_step = normalize_text(anchor.get("step") if anchor else "")
    if failure_step == "role_assignment":
        log_integrity_ok = has_complete_input and has_terminal_output and len(events) > 0
    elif failure_step == "decision":
        log_integrity_ok = has_complete_input and has_terminal_output and events_before_anchor > 0
    else:
        log_integrity_ok = (
            has_complete_input
            and has_terminal_output
            and len(events) > 0
            and (events_after_anchor > 0 or has_decision_stage)
        )

    consensus_metrics = compute_consensus_metrics(case_history, record.get("predicted_answer"))
    final_prediction_correct = normalize_answer(record.get("predicted_answer")) == normalize_answer(
        record.get("ground_truth")
    )

    enriched = {
        **record,
        "uid": uid,
        "label": record["mas_audit_result"],
        "human_votes": votes,
        "human_majority_vote": majority_vote,
        "human_ai_agreement": human_ai_agreement,
        "human_agreement_strength": human_agreement_strength,
        "failure_round": anchor.get("round") if anchor else None,
        "failure_step": failure_step,
        "failure_agent": anchor.get("agent_id") if anchor else "",
        "failure_specialty": anchor.get("specialty") if anchor else None,
        "assigned_specialty_count": len(anchor.get("specialties", [])) if anchor else 0,
        "audit_reasoning": anchor.get("audit_reasoning", "") if anchor else "",
        "anchor_found": anchor is not None,
        "events_before_anchor": events_before_anchor,
        "events_after_anchor": events_after_anchor,
        "peer_response_count": peer_response_count,
        "num_rounds": len(case_history.get("rounds", [])),
        "event_count": len(events),
        "distinct_stage_count": len(stage_types),
        "unique_agent_count": len(unique_agents),
        "stage_types": sorted(stage_types),
        "has_complete_input": has_complete_input,
        "has_terminal_output": has_terminal_output,
        "has_decision_stage": has_decision_stage,
        "log_integrity_ok": log_integrity_ok,
        "final_prediction_correct": final_prediction_correct,
        "modality": classify_modality(record["dataset"]),
        "architecture_family": classify_architecture_family(record["mas"]),
        **consensus_metrics,
    }
    enriched["base_score"] = compute_base_score(enriched)
    enriched["selection_reason"] = build_selection_reason(enriched)
    return enriched


def group_key(candidate: dict[str, Any]) -> tuple[str, str]:
    return candidate["failure_code"], candidate["label"]


def coverage_from_selection(selection: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    candidates = list(selection.values())
    modalities = Counter(candidate["modality"] for candidate in candidates)
    architecture_families = Counter(candidate["architecture_family"] for candidate in candidates)
    datasets = Counter(candidate["dataset"] for candidate in candidates)
    frameworks = Counter(candidate["mas"] for candidate in candidates)
    labels = Counter(candidate["label"] for candidate in candidates)

    return {
        "modalities": dict(modalities),
        "architecture_families": dict(architecture_families),
        "datasets": dict(datasets),
        "frameworks": dict(frameworks),
        "labels": dict(labels),
    }


def diversity_bonus(selection: dict[tuple[str, str], dict[str, Any]]) -> float:
    coverage = coverage_from_selection(selection)
    modalities = coverage["modalities"]
    architecture_families = coverage["architecture_families"]
    datasets = coverage["datasets"]
    frameworks = coverage["frameworks"]

    bonus = 0.0

    if REQUIRED_MODALITIES.issubset(set(modalities)):
        bonus += 10.0

    for family in REQUIRED_ARCHITECTURE_FAMILIES:
        if family in architecture_families:
            bonus += 12.0

    if "iterative_debate" in architecture_families:
        bonus += 2.0

    bonus += 0.4 * min(modalities.get("text", 0), modalities.get("visual", 0))
    bonus += 0.2 * len(datasets)
    bonus += 0.2 * len(frameworks)

    return round(bonus, 4)


def selection_objective(selection: dict[tuple[str, str], dict[str, Any]]) -> float:
    return round(
        sum(candidate["base_score"] for candidate in selection.values()) + diversity_bonus(selection),
        4,
    )


def select_case_studies(grouped_candidates: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[tuple[str, str], dict[str, Any]]:
    selection = {group: candidates[0] for group, candidates in grouped_candidates.items()}

    improved = True
    while improved:
        improved = False
        current_score = selection_objective(selection)
        best_swap = None
        best_delta = 0.0

        for group, current_candidate in selection.items():
            for alternative in grouped_candidates[group][1:]:
                if alternative["uid"] == current_candidate["uid"]:
                    continue
                trial_selection = dict(selection)
                trial_selection[group] = alternative
                delta = selection_objective(trial_selection) - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (group, alternative)

        if best_swap is not None:
            group, alternative = best_swap
            selection[group] = alternative
            improved = True

    return selection


def write_jsonl(filepath: Path, rows: list[dict[str, Any]]) -> None:
    with open(filepath, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(filepath: Path, selected_candidates: list[dict[str, Any]]) -> None:
    fieldnames = [
        "failure_code",
        "label",
        "qid",
        "dataset",
        "modality",
        "mas",
        "architecture_family",
        "llm",
        "human_votes",
        "human_majority_vote",
        "mas_audit_result",
        "human_agreement_strength",
        "failure_round",
        "failure_step",
        "failure_agent",
        "predicted_answer",
        "ground_truth",
        "final_prediction_correct",
        "consensus_rate",
        "consensus_strength",
        "num_rounds",
        "distinct_stage_count",
        "events_before_anchor",
        "events_after_anchor",
        "peer_response_count",
        "log_integrity_ok",
        "base_score",
        "selection_reason",
    ]

    with open(filepath, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in selected_candidates:
            writer.writerow(
                {
                    "failure_code": candidate["failure_code"],
                    "label": candidate["label"],
                    "qid": candidate["qid"],
                    "dataset": candidate["dataset"],
                    "modality": candidate["modality"],
                    "mas": candidate["mas"],
                    "architecture_family": candidate["architecture_family"],
                    "llm": candidate["llm"],
                    "human_votes": ",".join(str(vote) for vote in candidate["human_votes"]),
                    "human_majority_vote": candidate["human_majority_vote"],
                    "mas_audit_result": candidate["mas_audit_result"],
                    "human_agreement_strength": candidate["human_agreement_strength"],
                    "failure_round": candidate["failure_round"],
                    "failure_step": candidate["failure_step"],
                    "failure_agent": candidate["failure_agent"],
                    "predicted_answer": candidate.get("predicted_answer"),
                    "ground_truth": candidate.get("ground_truth"),
                    "final_prediction_correct": candidate["final_prediction_correct"],
                    "consensus_rate": candidate["consensus_rate"],
                    "consensus_strength": candidate["consensus_strength"],
                    "num_rounds": candidate["num_rounds"],
                    "distinct_stage_count": candidate["distinct_stage_count"],
                    "events_before_anchor": candidate["events_before_anchor"],
                    "events_after_anchor": candidate["events_after_anchor"],
                    "peer_response_count": candidate["peer_response_count"],
                    "log_integrity_ok": candidate["log_integrity_ok"],
                    "base_score": candidate["base_score"],
                    "selection_reason": candidate["selection_reason"],
                }
            )


def build_report(
    all_candidates: list[dict[str, Any]],
    grouped_candidates: dict[tuple[str, str], list[dict[str, Any]]],
    selection: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    eligible_candidates = [candidate for candidate in all_candidates if candidate["human_ai_agreement"]]
    selected_candidates = list(selection.values())
    coverage = coverage_from_selection(selection)

    per_group_counts = {}
    for failure_code in FAILURE_MODE_ORDER:
        for label in ["1", "0"]:
            key = (failure_code, label)
            group_candidates = grouped_candidates.get(key, [])
            per_group_counts[f"{failure_code}|{label}"] = {
                "eligible_candidates": len(group_candidates),
                "selected_qid": selection[key]["qid"],
                "selected_mas": selection[key]["mas"],
                "selected_dataset": selection[key]["dataset"],
                "selected_score": selection[key]["base_score"],
            }

    return {
        "candidate_pool_size": len(all_candidates),
        "eligible_after_human_ai_agreement": len(eligible_candidates),
        "selected_case_count": len(selected_candidates),
        "selection_objective": selection_objective(selection),
        "diversity_bonus": diversity_bonus(selection),
        "coverage": coverage,
        "required_modalities_satisfied": REQUIRED_MODALITIES.issubset(set(coverage["modalities"])),
        "required_architecture_families_satisfied": REQUIRED_ARCHITECTURE_FAMILIES.issubset(
            set(coverage["architecture_families"])
        ),
        "per_group_counts": per_group_counts,
        "selected_cases": [
            {
                "failure_code": candidate["failure_code"],
                "label": candidate["label"],
                "qid": candidate["qid"],
                "dataset": candidate["dataset"],
                "mas": candidate["mas"],
                "llm": candidate["llm"],
                "modality": candidate["modality"],
                "architecture_family": candidate["architecture_family"],
                "base_score": candidate["base_score"],
                "selection_reason": candidate["selection_reason"],
            }
            for candidate in sorted(
                selected_candidates,
                key=lambda candidate: (FAILURE_MODE_ORDER.index(candidate["failure_code"]), candidate["label"]),
            )
        ],
    }


def main() -> None:
    terminal_log_file = CASE_STUDY_SELECTION_DIR / "case_study_filter_terminal.log"
    print(f"!!! Terminal output is being captured to: {terminal_log_file} !!!")
    sys.stdout = DualLogger(terminal_log_file, sys.stdout)
    sys.stderr = DualLogger(terminal_log_file, sys.stderr)

    human_votes = load_human_votes()
    candidates = [enrich_candidate(record, human_votes) for record in load_candidate_pool()]

    eligible_candidates = [candidate for candidate in candidates if candidate["human_ai_agreement"]]
    print(f"Total candidates: {len(candidates)}")
    print(f"Eligible after human-AI agreement filtering: {len(eligible_candidates)}")

    grouped_candidates = {}
    for failure_code in FAILURE_MODE_ORDER:
        for label in ["1", "0"]:
            key = (failure_code, label)
            group_candidates = [
                candidate
                for candidate in eligible_candidates
                if candidate["failure_code"] == failure_code and candidate["label"] == label
            ]
            group_candidates.sort(
                key=lambda candidate: (
                    candidate["base_score"],
                    candidate["human_agreement_strength"],
                    int(candidate["log_integrity_ok"]),
                    int(not candidate["final_prediction_correct"]) if candidate["label"] == "1" else 0,
                    candidate["qid"],
                ),
                reverse=True,
            )

            if not group_candidates:
                raise RuntimeError(
                    f"No eligible candidates remain for failure_code={failure_code}, label={label}"
                )

            grouped_candidates[key] = group_candidates
            top_candidate = group_candidates[0]
            print(
                f"Group {failure_code} / label={label}: {len(group_candidates)} eligible candidates. "
                f"Top candidate -> qid={top_candidate['qid']}, mas={top_candidate['mas']}, "
                f"dataset={top_candidate['dataset']}, score={top_candidate['base_score']}"
            )

    selection = select_case_studies(grouped_candidates)
    selected_candidates = sorted(
        selection.values(),
        key=lambda candidate: (FAILURE_MODE_ORDER.index(candidate["failure_code"]), candidate["label"]),
    )

    report = build_report(candidates, grouped_candidates, selection)
    print("\nFinal selection coverage:")
    print(json.dumps(report["coverage"], ensure_ascii=False, indent=2))
    print(
        f"Required modality coverage satisfied: {report['required_modalities_satisfied']} | "
        f"Required architecture coverage satisfied: {report['required_architecture_families_satisfied']}"
    )

    selected_case_jsonl = CASE_STUDY_SELECTION_DIR / "selected_case_studies.jsonl"
    summary_csv = CASE_STUDY_SELECTION_DIR / "selected_case_studies_summary.csv"
    report_json = CASE_STUDY_SELECTION_DIR / "case_study_selection_report.json"

    jsonl_rows = []
    for candidate in selected_candidates:
        selection_metadata = {
            "uid": candidate["uid"],
            "failure_code": candidate["failure_code"],
            "label": candidate["label"],
            "modality": candidate["modality"],
            "architecture_family": candidate["architecture_family"],
            "human_votes": candidate["human_votes"],
            "human_majority_vote": candidate["human_majority_vote"],
            "human_ai_agreement": candidate["human_ai_agreement"],
            "human_agreement_strength": candidate["human_agreement_strength"],
            "failure_round": candidate["failure_round"],
            "failure_step": candidate["failure_step"],
            "failure_agent": candidate["failure_agent"],
            "audit_reasoning": candidate["audit_reasoning"],
            "final_prediction_correct": candidate["final_prediction_correct"],
            "consensus_rate": candidate["consensus_rate"],
            "consensus_strength": candidate["consensus_strength"],
            "num_rounds": candidate["num_rounds"],
            "distinct_stage_count": candidate["distinct_stage_count"],
            "events_before_anchor": candidate["events_before_anchor"],
            "events_after_anchor": candidate["events_after_anchor"],
            "peer_response_count": candidate["peer_response_count"],
            "log_integrity_ok": candidate["log_integrity_ok"],
            "base_score": candidate["base_score"],
            "selection_reason": candidate["selection_reason"],
        }

        output_row = {
            "selection_metadata": selection_metadata,
            "qid": candidate["qid"],
            "dataset": candidate["dataset"],
            "mas": candidate["mas"],
            "llm": candidate["llm"],
            "question": candidate["question"],
            "options": candidate["options"],
            "image_path": candidate.get("image_path"),
            "ground_truth": candidate["ground_truth"],
            "predicted_answer": candidate.get("predicted_answer"),
            "case_history": candidate["case_history"],
        }
        jsonl_rows.append(output_row)

    write_jsonl(selected_case_jsonl, jsonl_rows)
    write_summary_csv(summary_csv, selected_candidates)
    save_json(report, report_json)

    print(f"\nSelected case studies written to: {selected_case_jsonl}")
    print(f"Selection summary written to: {summary_csv}")
    print(f"Selection report written to: {report_json}")


if __name__ == "__main__":
    main()
