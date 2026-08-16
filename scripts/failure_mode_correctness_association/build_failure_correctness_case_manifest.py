#!/usr/bin/env python3
"""Build a case-level manifest from the frozen March 2 automated-audit logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from failure_mode_schema import (
    DATASET_DISPLAY,
    FAILURE_MODES,
    LLM_DISPLAY,
    MAS_DISPLAY,
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
    parser.add_argument(
        "--mas-dir",
        type=Path,
        default=project_root / "logs" / "mas_collaboration_results_audit",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def parse_file_metadata(path: Path) -> tuple[str, str, str]:
    parts = path.stem.lower().split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected audit filename: {path.name}")
    return parts[0], parts[1], parts[2]


def normalize_answer(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def iter_entries(rounds: list[dict], mode) -> list[dict]:
    collected: list[dict] = []
    for audit_round in rounds:
        if not isinstance(audit_round, dict):
            continue
        entries = audit_round.get(mode.log_key, [])
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            step = str(entry.get("step", "")).lower()
            if mode.allowed_steps is not None and step not in mode.allowed_steps:
                continue
            collected.append(entry)
    return collected


def classify_case_mode(mas: str, rounds: list[dict], mode) -> dict[str, object]:
    if mode.applicable_mas is not None and mas not in mode.applicable_mas:
        return {
            "state": "not_applicable",
            "label": pd.NA,
            "audit_count": 0,
            "valid_audit_count": 0,
            "invalid_audit_count": 0,
            "first_step": "",
            "last_step": "",
        }

    entries = iter_entries(rounds, mode)
    if not entries:
        return {
            "state": "not_audited",
            "label": pd.NA,
            "audit_count": 0,
            "valid_audit_count": 0,
            "invalid_audit_count": 0,
            "first_step": "",
            "last_step": "",
        }

    statuses = [
        str((entry.get("audit_result") or {}).get(mode.status_key))
        for entry in entries
    ]
    valid = [value for value in statuses if value in {"0", "1"}]
    invalid_count = len(statuses) - len(valid)
    steps = [str(entry.get("step", "")) for entry in entries]
    if not valid:
        state = "unknown"
        label = pd.NA
    elif "1" in valid:
        state = "positive"
        label = 1
    else:
        state = "negative"
        label = 0
    return {
        "state": state,
        "label": label,
        "audit_count": len(entries),
        "valid_audit_count": len(valid),
        "invalid_audit_count": invalid_count,
        "first_step": steps[0] if steps else "",
        "last_step": steps[-1] if steps else "",
    }


def load_reference_answers(path: Path) -> dict[str, tuple[str, str]]:
    answers: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            qid = str(record.get("qid"))
            if qid in answers:
                raise ValueError(f"Duplicate qid {qid} in {path.name}")
            answers[qid] = (
                normalize_answer(record.get("predicted_answer")),
                normalize_answer(record.get("ground_truth")),
            )
    return answers


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_files = sorted(args.audit_dir.glob("*.jsonl"))
    if len(audit_files) != 144:
        raise ValueError(f"Expected 144 audit files, found {len(audit_files)}")

    rows: list[dict[str, object]] = []
    file_counts: Counter[int] = Counter()
    for audit_file in audit_files:
        mas, dataset, llm = parse_file_metadata(audit_file)
        source_file = args.mas_dir / audit_file.name
        if not source_file.exists():
            raise FileNotFoundError(source_file)
        reference_answers = load_reference_answers(source_file)
        seen_qids: set[str] = set()
        current_file_count = 0
        with audit_file.open(encoding="utf-8") as handle:
            for source_row, line in enumerate(handle, start=1):
                record = json.loads(line)
                qid = str(record.get("qid"))
                if qid in seen_qids:
                    raise ValueError(f"Duplicate qid {qid} in {audit_file.name}")
                seen_qids.add(qid)
                current_file_count += 1
                predicted = normalize_answer(record.get("predicted_answer"))
                ground_truth = normalize_answer(record.get("ground_truth"))
                if qid not in reference_answers:
                    raise ValueError(f"qid {qid} absent from {source_file.name}")
                if (predicted, ground_truth) != reference_answers[qid]:
                    raise ValueError(
                        f"Answer mismatch for {audit_file.name}, qid={qid}"
                    )
                answer_parse_valid = bool(predicted and ground_truth)
                base: dict[str, object] = {
                    "source_file": str(audit_file),
                    "source_row": source_row,
                    "qid": qid,
                    "question_cluster": f"{dataset}:{qid}",
                    "dataset": dataset,
                    "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
                    "modality": "VQA"
                    if dataset in {"pathvqa", "vqa-rad", "slake"}
                    else "QA",
                    "mas": mas,
                    "mas_display": MAS_DISPLAY.get(mas, mas),
                    "underlying_llm": llm,
                    "underlying_llm_display": LLM_DISPLAY.get(llm, llm),
                    "predicted_answer": predicted,
                    "ground_truth": ground_truth,
                    "answer_parse_valid": answer_parse_valid,
                    "correctness": int(predicted == ground_truth)
                    if answer_parse_valid
                    else pd.NA,
                }
                rounds = (
                    ((record.get("case_history") or {}).get("audit") or {}).get(
                        "rounds"
                    )
                    or []
                )
                for mode in FAILURE_MODES:
                    result = classify_case_mode(mas, rounds, mode)
                    prefix = f"f_{mode.code.replace('.', '_')}"
                    for key, value in result.items():
                        base[f"{prefix}_{key}"] = value
                rows.append(base)
        if set(reference_answers) != seen_qids:
            raise ValueError(f"qid set mismatch for {audit_file.name}")
        file_counts[current_file_count] += 1

    manifest = pd.DataFrame(rows)
    manifest["correctness"] = manifest["correctness"].astype("Int64")
    for mode in FAILURE_MODES:
        column = f"f_{mode.code.replace('.', '_')}_label"
        manifest[column] = manifest[column].astype("Int64")

    manifest_path = args.output_dir / "failure_correctness_case_manifest.csv.gz"
    manifest.to_csv(manifest_path, index=False, compression="gzip")

    flow_rows: list[dict[str, object]] = [
        {"item": "planned_records", "count": 14400, "note": "6 MAS × 24 dataset–LLM settings × 100"},
        {"item": "observed_records", "count": len(manifest), "note": "frozen 20260302 audit logs"},
        {"item": "valid_correctness", "count": int(manifest["correctness"].notna().sum()), "note": "parsed predicted_answer and ground_truth"},
        {"item": "unique_question_clusters", "count": int(manifest["question_cluster"].nunique()), "note": "dataset:qid"},
    ]
    for mode in FAILURE_MODES:
        prefix = f"f_{mode.code.replace('.', '_')}"
        counts = manifest[f"{prefix}_state"].value_counts(dropna=False)
        for state in (
            "positive",
            "negative",
            "unknown",
            "not_applicable",
            "not_audited",
        ):
            flow_rows.append(
                {
                    "item": f"F-{mode.code}_{state}",
                    "count": int(counts.get(state, 0)),
                    "note": mode.short_name,
                }
            )
    pd.DataFrame(flow_rows).to_csv(
        args.output_dir / "failure_correctness_manifest_flow_summary.csv",
        index=False,
    )

    metadata = {
        "audit_directory": str(args.audit_dir),
        "mas_directory": str(args.mas_dir),
        "audit_file_count": len(audit_files),
        "record_count": len(manifest),
        "records_per_file_distribution": dict(file_counts),
        "manifest_file": str(manifest_path),
    }
    (args.output_dir / "failure_correctness_manifest_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
