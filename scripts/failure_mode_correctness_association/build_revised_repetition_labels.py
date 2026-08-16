#!/usr/bin/env python3
"""Build protocol-v4 event- and case-level labels for F-2.2.1."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


MODE_KEY = "2_2_1_repetition_of_initial_views"
STATUS_KEY = "interaction_redundancy"
APPLICABLE_MAS = frozenset({"colacare", "mac", "medagent", "reconcile"})
HEALTHCARE_MAS = "healthcareagent"
STRUCTURED_OPTION_KEYS = (
    "option",
    "choice",
    "selected_option",
    "true_options",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_output = Path(
        "/mnt/c/Users/LeiGu/Dropbox/[Preprint 2026] "
        "Auditing medical multi-agent AI reveals risks of false consensus/"
        "Preprint/analysis/failure_mode_correctness_association"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=project_root / "logs" / "audit_results" / "20260302",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def parse_file_metadata(path: Path) -> tuple[str, str, str]:
    parts = path.stem.lower().split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected audit filename: {path.name}")
    return parts[0], parts[1], parts[2]


def normalize_scalar(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def option_map(record: dict[str, Any]) -> dict[str, str]:
    options = record.get("options")
    mapping: dict[str, str] = {}
    if isinstance(options, dict):
        mapping = {
            str(key).strip().upper(): normalize_scalar(value).lower()
            for key, value in options.items()
        }
    elif isinstance(options, list):
        mapping = {
            chr(65 + index): normalize_scalar(value).lower()
            for index, value in enumerate(options)
        }
    ground_truth = normalize_scalar(record.get("ground_truth")).upper()
    if ground_truth and ground_truth not in mapping:
        mapping[ground_truth] = ""
    return mapping


def unwrap_string(value: str) -> str:
    unwrapped = value.strip()
    for _ in range(2):
        if len(unwrapped) < 2:
            break
        if (unwrapped[0], unwrapped[-1]) in {
            ('"', '"'),
            ("'", "'"),
            ("`", "`"),
        }:
            unwrapped = unwrapped[1:-1].strip()
        else:
            break
    return unwrapped


def parse_general_options(
    value: object, record: dict[str, Any]
) -> tuple[tuple[str, ...] | None, str]:
    """Parse structured or explicitly labeled options without semantic inference."""
    mapping = option_map(record)
    valid_labels = set(mapping)
    option_text_to_label = {
        text: label for label, text in mapping.items() if text
    }

    if isinstance(value, dict):
        normalized = {str(key).strip().lower(): item for key, item in value.items()}
        for key in STRUCTURED_OPTION_KEYS:
            if key in normalized:
                parsed, rule = parse_general_options(normalized[key], record)
                return parsed, f"structured_{key}:{rule}"
        return None, "unsupported_dictionary"

    if isinstance(value, (list, tuple, set)):
        collected: list[str] = []
        for item in value:
            parsed, _ = parse_general_options(item, record)
            if parsed is None:
                return None, "unparseable_option_list"
            collected.extend(parsed)
        unique = tuple(dict.fromkeys(collected))
        return (unique, "option_list") if unique else (None, "empty_option_list")

    text = normalize_scalar(value)
    if not text or text.lower() == "no structured answer found in response":
        return None, "empty_or_missing_structured_answer"

    try:
        literal = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        literal = text
    if not isinstance(literal, str):
        parsed, rule = parse_general_options(literal, record)
        return parsed, f"literal:{rule}"

    text = unwrap_string(literal)
    upper = text.upper()
    if upper in valid_labels:
        return (upper,), "exact_option_label"
    if text.lower() in option_text_to_label:
        return (option_text_to_label[text.lower()],), "exact_option_text"

    patterns = (
        r"^\(([A-Z])\)(?:\s|$)",
        r"^(?:OPTION|ANSWER|CHOICE)\s*[:=\-]?\s*([A-Z])(?:\b|\s*[:.)\-])",
        r"^([A-Z])(?:\b|\s*[:.)\-])",
    )
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match and match.group(1) in valid_labels:
            return (match.group(1),), "explicit_leading_option_label"
    return None, "free_text_without_deterministic_option"


def parse_strict_single_option(
    value: object, record: dict[str, Any]
) -> tuple[tuple[str, ...] | None, str]:
    """Parse MedAgents disagree answers only when the whole value is one option."""
    mapping = option_map(record)
    valid_labels = set(mapping)

    if isinstance(value, dict):
        normalized = {str(key).strip().lower(): item for key, item in value.items()}
        for key in ("option", "choice", "selected_option"):
            if key in normalized:
                return parse_strict_single_option(normalized[key], record)
        return None, "strict_unsupported_dictionary"
    if isinstance(value, (list, tuple, set)):
        if len(value) != 1:
            return None, "strict_not_single_option"
        return parse_strict_single_option(next(iter(value)), record)

    text = unwrap_string(normalize_scalar(value))
    if not text:
        return None, "strict_empty_answer"
    upper = text.upper()
    if upper in valid_labels:
        return (upper,), "strict_exact_option_label"
    match = re.fullmatch(
        r"(?:OPTION|ANSWER|CHOICE)\s*[:=\-]?\s*([A-Z])", upper
    )
    if match and match.group(1) in valid_labels:
        return (match.group(1),), "strict_labeled_single_option"
    return None, "strict_free_text"


def parsed_output(item: object) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    log = item.get("log") if isinstance(item.get("log"), dict) else item
    output = log.get("parsed_output") if isinstance(log, dict) else None
    return output if isinstance(output, dict) else {}


def round_by_number(case_history: dict[str, Any], number: object) -> dict[str, Any] | None:
    for item in case_history.get("rounds") or []:
        if isinstance(item, dict) and str(item.get("round")) == str(number):
            return item
    return None


def step_items(round_record: dict[str, Any], step: str) -> list[dict[str, Any]]:
    key = {"analysis": "opinions", "review": "reviews"}.get(step, step)
    items = round_record.get(key, [])
    if isinstance(items, dict):
        items = [items]
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def match_contribution(
    round_record: dict[str, Any], step: str, agent_id: object
) -> tuple[dict[str, Any] | None, str]:
    matched = [
        item
        for item in step_items(round_record, step)
        if str(item.get("agent_id")) == str(agent_id)
    ]
    if len(matched) == 1:
        return matched[0], "matched"
    if not matched:
        return None, "missing_agent_step_match"
    return None, "duplicate_agent_step_match"


def extract_repeated_answer(
    mas: str,
    record: dict[str, Any],
    round_record: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    step = normalize_scalar(entry.get("step")).lower()
    agent_id = entry.get("agent_id")
    contribution, match_state = match_contribution(round_record, step, agent_id)
    if contribution is None:
        return {
            "extraction_state": match_state,
            "answer_source": "",
            "raw_answer": None,
            "parsed_options": None,
            "parse_rule": match_state,
        }

    output = parsed_output(contribution)
    if mas == "medagent" and step == "review":
        agree = output.get("agree")
        if agree is True:
            synthesis = round_record.get("synthesis") or {}
            synthesis_output = parsed_output(synthesis)
            raw_answer = synthesis_output.get("answer")
            parsed, rule = parse_general_options(raw_answer, record)
            return {
                "extraction_state": "resolved" if parsed else "unparseable",
                "answer_source": "round_synthesis_answer_for_agree_true",
                "raw_answer": raw_answer,
                "parsed_options": parsed,
                "parse_rule": rule,
                "review_agree": True,
            }
        if agree is False:
            raw_answer = output.get("answer")
            parsed, rule = parse_strict_single_option(raw_answer, record)
            return {
                "extraction_state": "resolved" if parsed else "unparseable",
                "answer_source": "review_suggested_answer_for_agree_false",
                "raw_answer": raw_answer,
                "parsed_options": parsed,
                "parse_rule": rule,
                "review_agree": False,
            }
        return {
            "extraction_state": "unparseable",
            "answer_source": "medagent_review_invalid_agree",
            "raw_answer": output.get("answer"),
            "parsed_options": None,
            "parse_rule": "missing_or_invalid_agree",
            "review_agree": agree,
        }

    raw_answer = output.get("answer")
    parsed, rule = parse_general_options(raw_answer, record)
    return {
        "extraction_state": "resolved" if parsed else "unparseable",
        "answer_source": f"current_{step}_answer",
        "raw_answer": raw_answer,
        "parsed_options": parsed,
        "parse_rule": rule,
    }


def serialize_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def classify_positive_repetition(
    extraction: dict[str, Any], ground_truth: str
) -> tuple[str, str, bool | None]:
    parsed = extraction.get("parsed_options")
    if not parsed:
        return "unknown", "repetition_answer_unparseable", None
    parsed_options = tuple(parsed)
    if len(parsed_options) != 1:
        return "positive", "multiple_options_in_single_answer_benchmark", False
    correct = parsed_options[0].strip().lower() == ground_truth.strip().lower()
    return (
        ("negative", "repetition_of_correct_answer", True)
        if correct
        else ("positive", "repetition_of_incorrect_answer", False)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_files = sorted(args.audit_dir.glob("*.jsonl"))
    if len(audit_files) != 144:
        raise ValueError(f"Expected 144 audit files, found {len(audit_files)}")

    event_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for path in audit_files:
        mas, dataset, underlying_llm = parse_file_metadata(path)
        with path.open(encoding="utf-8") as handle:
            for source_row, line in enumerate(handle, start=1):
                record = json.loads(line)
                qid = str(record.get("qid"))
                case_history = record.get("case_history") or {}
                ground_truth = normalize_scalar(record.get("ground_truth"))
                case_key = {
                    "source_file": str(path),
                    "source_row": source_row,
                    "dataset": dataset,
                    "qid": qid,
                    "mas": mas,
                    "underlying_llm": underlying_llm,
                    "ground_truth": ground_truth,
                }

                if mas == HEALTHCARE_MAS:
                    case_rows.append(
                        {
                            **case_key,
                            "revised_f_2_2_1_state": "not_applicable",
                            "revised_f_2_2_1_label": pd.NA,
                            "revised_f_2_2_1_reason": "healthcareagent_safety_reviews_not_benchmark_answers",
                            "repetition_event_count": 0,
                            "positive_event_count": 0,
                            "negative_event_count": 0,
                            "unknown_event_count": 0,
                            "correct_repetition_event_count": 0,
                            "incorrect_repetition_event_count": 0,
                            "mixed_repetition": False,
                        }
                    )
                    continue
                if mas not in APPLICABLE_MAS:
                    case_rows.append(
                        {
                            **case_key,
                            "revised_f_2_2_1_state": "not_applicable",
                            "revised_f_2_2_1_label": pd.NA,
                            "revised_f_2_2_1_reason": "framework_not_in_revised_applicability_map",
                            "repetition_event_count": 0,
                            "positive_event_count": 0,
                            "negative_event_count": 0,
                            "unknown_event_count": 0,
                            "correct_repetition_event_count": 0,
                            "incorrect_repetition_event_count": 0,
                            "mixed_repetition": False,
                        }
                    )
                    continue

                applicable_entries: list[tuple[object, dict[str, Any]]] = []
                audit_rounds = ((case_history.get("audit") or {}).get("rounds") or [])
                for audit_round in audit_rounds:
                    if not isinstance(audit_round, dict):
                        continue
                    entries = audit_round.get(MODE_KEY, [])
                    if isinstance(entries, dict):
                        entries = [entries]
                    if not isinstance(entries, list):
                        continue
                    for entry in entries:
                        if isinstance(entry, dict):
                            applicable_entries.append((audit_round.get("round"), entry))

                if not applicable_entries:
                    case_rows.append(
                        {
                            **case_key,
                            "revised_f_2_2_1_state": "not_audited",
                            "revised_f_2_2_1_label": pd.NA,
                            "revised_f_2_2_1_reason": "no_repetition_audit_entry",
                            "repetition_event_count": 0,
                            "positive_event_count": 0,
                            "negative_event_count": 0,
                            "unknown_event_count": 0,
                            "correct_repetition_event_count": 0,
                            "incorrect_repetition_event_count": 0,
                            "mixed_repetition": False,
                        }
                    )
                    continue

                states: list[str] = []
                event_reasons: list[str] = []
                correct_repetition_count = 0
                incorrect_repetition_count = 0
                repetition_event_count = 0
                for audit_round_number, entry in applicable_entries:
                    original_status = normalize_scalar(
                        (entry.get("audit_result") or {}).get(STATUS_KEY)
                    )
                    event_state: str
                    event_reason: str
                    extraction: dict[str, Any] = {
                        "extraction_state": "not_required",
                        "answer_source": "",
                        "raw_answer": None,
                        "parsed_options": None,
                        "parse_rule": "not_required",
                    }
                    intermediate_correct: bool | None = None
                    if original_status == "0":
                        event_state = "negative"
                        event_reason = "auditor_did_not_flag_repetition"
                    elif original_status == "1":
                        repetition_event_count += 1
                        round_record = round_by_number(case_history, audit_round_number)
                        if round_record is None:
                            event_state = "unknown"
                            event_reason = "missing_matching_round"
                            extraction["extraction_state"] = "missing_matching_round"
                            extraction["parse_rule"] = "missing_matching_round"
                        else:
                            extraction = extract_repeated_answer(
                                mas, record, round_record, entry
                            )
                            event_state, event_reason, intermediate_correct = (
                                classify_positive_repetition(extraction, ground_truth)
                            )
                        if event_reason == "repetition_of_correct_answer":
                            correct_repetition_count += 1
                        elif event_reason in {
                            "repetition_of_incorrect_answer",
                            "multiple_options_in_single_answer_benchmark",
                        }:
                            incorrect_repetition_count += 1
                    else:
                        event_state = "unknown"
                        event_reason = "invalid_original_repetition_status"

                    states.append(event_state)
                    event_reasons.append(event_reason)
                    event_rows.append(
                        {
                            **case_key,
                            "audit_round": audit_round_number,
                            "agent_id": entry.get("agent_id"),
                            "specialty": entry.get("specialty"),
                            "step": entry.get("step"),
                            "original_repetition_status": original_status,
                            "revised_event_state": event_state,
                            "revised_event_reason": event_reason,
                            "answer_source": extraction.get("answer_source"),
                            "raw_intermediate_answer": serialize_value(
                                extraction.get("raw_answer")
                            ),
                            "parsed_intermediate_options": serialize_value(
                                extraction.get("parsed_options")
                            ),
                            "parse_rule": extraction.get("parse_rule"),
                            "intermediate_answer_correct": intermediate_correct,
                            "review_agree": extraction.get("review_agree", pd.NA),
                        }
                    )

                if "positive" in states:
                    case_state = "positive"
                    case_label: object = 1
                    case_reason = "at_least_one_incorrect_repetition"
                elif "unknown" in states:
                    case_state = "unknown"
                    case_label = pd.NA
                    case_reason = "no_positive_event_and_at_least_one_unknown_event"
                else:
                    case_state = "negative"
                    case_label = 0
                    case_reason = "all_applicable_events_negative"

                case_rows.append(
                    {
                        **case_key,
                        "revised_f_2_2_1_state": case_state,
                        "revised_f_2_2_1_label": case_label,
                        "revised_f_2_2_1_reason": case_reason,
                        "repetition_event_count": repetition_event_count,
                        "positive_event_count": states.count("positive"),
                        "negative_event_count": states.count("negative"),
                        "unknown_event_count": states.count("unknown"),
                        "correct_repetition_event_count": correct_repetition_count,
                        "incorrect_repetition_event_count": incorrect_repetition_count,
                        "mixed_repetition": bool(
                            correct_repetition_count and incorrect_repetition_count
                        ),
                        "event_reasons": " | ".join(sorted(set(event_reasons))),
                    }
                )

    cases = pd.DataFrame(case_rows)
    events = pd.DataFrame(event_rows)
    if len(cases) != 14370:
        raise ValueError(f"Expected 14,370 case rows, found {len(cases):,}")
    unique_cases = cases[["dataset", "qid", "mas", "underlying_llm"]].drop_duplicates()
    if len(unique_cases) != len(cases):
        raise ValueError("Duplicate revised F-2.2.1 case key")
    cases["revised_f_2_2_1_label"] = cases["revised_f_2_2_1_label"].astype("Int64")

    case_path = args.output_dir / "revised_f_2_2_1_case_labels.csv.gz"
    event_path = args.output_dir / "revised_f_2_2_1_event_labels.csv.gz"
    cases.to_csv(case_path, index=False, compression="gzip")
    events.to_csv(event_path, index=False, compression="gzip")

    state_counts = cases["revised_f_2_2_1_state"].value_counts().to_dict()
    reason_counts = events["revised_event_reason"].value_counts().to_dict()
    flow_rows = [
        {"level": "case", "state_or_reason": key, "count": int(value)}
        for key, value in sorted(state_counts.items())
    ] + [
        {"level": "event", "state_or_reason": key, "count": int(value)}
        for key, value in sorted(reason_counts.items())
    ]
    pd.DataFrame(flow_rows).to_csv(
        args.output_dir / "revised_f_2_2_1_flow_summary.csv", index=False
    )
    metadata = {
        "definition": "repetition plus incorrect current endorsed intermediate answer",
        "applicable_mas": sorted(APPLICABLE_MAS),
        "healthcareagent_handling": "not_applicable",
        "case_count": len(cases),
        "event_count": len(events),
        "case_state_counts": {key: int(value) for key, value in state_counts.items()},
        "event_reason_counts": {key: int(value) for key, value in reason_counts.items()},
        "case_file": str(case_path),
        "event_file": str(event_path),
    }
    (args.output_dir / "revised_f_2_2_1_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
