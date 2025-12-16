# -*- coding: utf-8 -*-
"""
Mechanism-3: Collaboration Quality Signals by Stage (Full Version)

- Robust stage presence extraction from multiple log fields:
  rounds / discussion_history / steps (HealthcareAgent) / process_log (MDAgents) /
  viewpoints / audit_trail.collaboration_audits
- Shows ALL stages in the plot (0% for stages without data instead of hiding them)
- Accepts directories (walk) AND individual files via --files
"""

import os
import re
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- If you use your own loader, replace here ----
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# Global Config / Mappings
# -----------------------
FRAMEWORK_NAMES = (
    "ColaCare",
    "MedAgent",
    "HealthcareAgent",
    "MAC",
    "MDAgents",
    "ReConcile",
)

HEALTHCARE_STEP_TO_STAGE = {
    "1_Planner": "Propose",
    "2_Inquiry": "Propose",
    "3_Preliminary_Analysis": "Synthesis",
    "4_Safety_Review": "Review",
    "5_Final_Modification": "Conclusion",
}

MDA_STEP_TO_STAGE = {
    "determine_complexity": "Propose",
    "basic_initial": "Propose",
    "intermediate_initial": "Propose",
    "advanced_initial": "Propose",
    "recruit_experts": "Conclusion",
}

RECONCILE_PHASE_TO_STAGE = {
    "initial": "Propose",
    "discussion": "Review",
    "final": "Conclusion",
}

STAGE_SEQUENCE = ["Propose", "Synthesis", "Review", "Conclusion"]
STAGE_ORDER = {name: idx for idx, name in enumerate(STAGE_SEQUENCE)}
STAGE_LABEL_PATTERN = re.compile(
    r"^R(?P<round>\d+)\s+(?P<stage>Propose|Synthesis|Review|Conclusion)$"
)

CATEGORY_ORDER = [
    "Clinical Priority Mismatch Rate",
    "Domain-Specific Activation Rate",
    "Decision Quality Shortfall Rate",
]
CATEGORY_COLORS = {
    "Clinical Priority Mismatch Rate": "#D55E00",
    "Domain-Specific Activation Rate": "#009E73",
    "Decision Quality Shortfall Rate": "#0072B2",
}

DIAGNOSTIC_LEVEL_ORDER = {
    "Immediate (STAT)": 0,
    "Standard (Routine)": 1,
    "Delayed (Deferrable)": 2,
}
OVERALL_QUALITY_ORDER = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}
SPECIALIZED_LEVEL_ORDER = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}


# -----------------------
# Helpers
# -----------------------
def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _stage_sort_key(stage_label: str) -> Tuple[int, int, str]:
    m = STAGE_LABEL_PATTERN.match(stage_label)
    if m:
        round_num = int(m.group("round"))
        stage = m.group("stage")
        return (round_num, STAGE_ORDER.get(stage, 99), stage_label)
    return (10**9, 10**9, stage_label)


def _find_stage_round(stage_presence: Dict[int, Set[str]], preferred_round: int, stage_name: str) -> Optional[int]:
    """Find the closest round in which the given stage exists (>= preferred first, then <=)."""
    if not stage_presence:
        return None
    if stage_name in stage_presence.get(preferred_round, set()):
        return preferred_round
    rounds = sorted(stage_presence.keys())
    for r in rounds:
        if r >= preferred_round and stage_name in stage_presence.get(r, set()):
            return r
    for r in reversed(rounds):
        if r <= preferred_round and stage_name in stage_presence.get(r, set()):
            return r
    return None


def _detect_framework(file_path: str, data: Dict[str, Any]) -> str:
    """Directory-based hint + content sniffing fallback."""
    parts = set(os.path.normpath(file_path).split(os.sep))
    for name in FRAMEWORK_NAMES:
        if name in parts:
            return name

    ch = data.get("case_history") or {}
    if ch.get("discussion_history"):
        return "ReConcile"
    if ch.get("steps"):
        return "HealthcareAgent"
    if data.get("process_log"):
        return "MDAgents"
    if ch.get("rounds"):
        # fall back to ColaCare / MedAgent / MAC based on fields inside rounds
        for rnd in ch.get("rounds") or []:
            if "analyses" in rnd:
                return "MedAgent"
            if "doctor_responses" in rnd or "supervisor_response" in rnd:
                return "MAC"
        return "ColaCare"

    return "ColaCare"


def _audit_stage_hint(key: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse audit key to (round, stage)."""
    if not isinstance(key, str):
        return None, None
    lowered = key.lower()
    m = re.search(r"round_(\d+)", lowered)
    round_idx = int(m.group(1)) if m else None
    if round_idx == 0:
        round_idx = 1

    if any(t in lowered for t in ["synthesis", "summary", "aggregation", "group_synthesis"]):
        stage = "Synthesis"
    elif any(t in lowered for t in ["analysis", "initial", "preliminary"]):
        stage = "Propose"
    elif any(t in lowered for t in ["review", "discussion", "assist", "check"]):
        stage = "Review"
    elif any(t in lowered for t in ["final", "decision", "pre_decision", "vote"]):
        stage = "Conclusion"
    else:
        stage = None
    return round_idx, stage


def _is_priority_mismatch(level: Optional[str]) -> bool:
    if not level:
        return False
    v = DIAGNOSTIC_LEVEL_ORDER.get(level)
    if v is None:
        return False
    return v != min(DIAGNOSTIC_LEVEL_ORDER.values())


def _is_specialized_activated(level: Optional[str]) -> bool:
    if not level:
        return False
    v = SPECIALIZED_LEVEL_ORDER.get(level)
    if v is None:
        return False
    max_v = max(SPECIALIZED_LEVEL_ORDER.values())
    return v >= max_v - 1  # High 或次高


def _decision_quality_shortfall(decision_category: Optional[str], doctor_categories: List[Optional[str]]) -> bool:
    """True if decision quality < best doctor pre-decision quality."""
    if not decision_category:
        return True
    dscore = OVERALL_QUALITY_ORDER.get(decision_category)
    if dscore is None:
        return True
    dscores = [OVERALL_QUALITY_ORDER.get(c) for c in doctor_categories if c in OVERALL_QUALITY_ORDER]
    if not dscores:
        return dscore != max(OVERALL_QUALITY_ORDER.values())
    return any(s is not None and s > dscore for s in dscores)


# -----------------------
# Stage Presence Extraction (Robust)
# -----------------------
def _extract_stage_presence(data: Dict[str, Any], framework: str) -> Tuple[Dict[int, Set[str]], int]:
    """
    Build {round: {stage}} from:
    - case_history.rounds (opinions/analyses/supervisor_response/synthesis/reviews/decision)
    - case_history.discussion_history (ReConcile)
    - case_history.steps (HealthcareAgent)
    - process_log (MDAgents)
    - case_history.viewpoints[*].step
    - audit_trail.collaboration_audits (keys)
    """
    stage_presence: Dict[int, Set[str]] = {}
    ch = data.get("case_history") or {}

    def add_stage(r: Optional[int], stage: Optional[str]) -> None:
        if not stage:
            return
        rr = _safe_int(r, 1)
        if rr <= 0:
            rr = 1
        stage_presence.setdefault(rr, set()).add(stage)

    # 1) rounds-based (works for ColaCare/MedAgent/MAC/Generic)
    for rnd in (ch.get("rounds") or []):
        r = _safe_int(rnd.get("round"), 1)
        if rnd.get("opinions") or rnd.get("analyses"):
            add_stage(r, "Propose")
        if rnd.get("synthesis") or rnd.get("supervisor_response"):
            add_stage(r, "Synthesis")
        if rnd.get("reviews"):
            add_stage(r, "Review")
        if rnd.get("decision"):
            add_stage(r, "Conclusion")

    # 2) ReConcile: discussion_history
    for item in (ch.get("discussion_history") or []):
        phase = (item.get("phase") or "").lower()
        if phase in RECONCILE_PHASE_TO_STAGE:
            add_stage(1, RECONCILE_PHASE_TO_STAGE[phase])

    # 3) HealthcareAgent: steps
    for step in (ch.get("steps") or []):
        stg = HEALTHCARE_STEP_TO_STAGE.get(step.get("step"))
        if stg:
            add_stage(1, stg)

    # 4) MDAgents: process_log
    for entry in (data.get("process_log") or []):
        step_name = entry.get("step_name") or entry.get("step")
        if not isinstance(step_name, str):
            continue
        base = step_name.split(":", 1)[0]
        if base in MDA_STEP_TO_STAGE:
            add_stage(1, MDA_STEP_TO_STAGE[base])
        elif base.endswith("_initial") or "initial" in base:
            add_stage(1, "Propose")

    # 5) viewpoints[*].step -> infer round + stage keywords
    for lst in (ch.get("viewpoints") or {}).values():
        if not isinstance(lst, list):
            continue
        for step_obj in lst:
            step_name = step_obj.get("step") if isinstance(step_obj, dict) else None
            if not isinstance(step_name, str):
                continue
            m = re.search(r"round_(\d+)", step_name.lower())
            r = int(m.group(1)) if m else 1
            r = 1 if r <= 0 else r
            _, stg = _audit_stage_hint(step_name)  # reuse keyword mapping
            if stg:
                add_stage(r, stg)

    # 6) audits keys
    audit_trail = ch.get("audit_trail") or data.get("audit_trail") or {}
    audits = audit_trail.get("collaboration_audits") or {}
    for key in audits.keys():
        r, stg = _audit_stage_hint(key)
        add_stage(r, stg)

    # 7) final decision hints
    if ch.get("final_decision") or ch.get("final_decision_log") or data.get("predicted_answer"):
        # place Conclusion at max round seen, else R1
        max_r = max(stage_presence.keys(), default=1)
        add_stage(max_r, "Conclusion")

    final_round = max(stage_presence.keys(), default=0)
    return stage_presence, final_round


# -----------------------
# Collect per-case metrics
# -----------------------
def _collect_stage_counts_for_case(
    case_history: Dict[str, Any],
    audits: Dict[str, Any],
    stage_presence: Dict[int, Set[str]],
    final_round: int,
) -> Tuple[int, Dict[str, Dict[str, Dict[str, int]]]]:
    """Return (final_round, { 'Rk Stage': {Metric: {numerator, denominator}}})"""
    if not stage_presence:
        return 0, {}

    stage_metrics: Dict[str, Dict[str, Dict[str, int]]] = {}

    def ensure_stage(label: str) -> Dict[str, Dict[str, int]]:
        return stage_metrics.setdefault(
            label,
            {
                "Clinical Priority Mismatch Rate": {"numerator": 0, "denominator": 0},
                "Domain-Specific Activation Rate": {"numerator": 0, "denominator": 0},
                "Decision Quality Shortfall Rate": {"numerator": 0, "denominator": 0},
            },
        )

    # Gather pre-decision quality (doctors) by round for Decision comparison
    pre_decision_candidates: Dict[int, List[Optional[str]]] = defaultdict(list)
    for key, entry in (audits or {}).items():
        lowered = key.lower() if isinstance(key, str) else ""
        if "pre_decision_quality" not in lowered and "pre_vote" not in lowered:
            continue
        raw_round, _ = _audit_stage_hint(key)
        round_hint = raw_round if raw_round is not None else final_round
        values: List[Optional[str]] = []
        if isinstance(entry, list):
            for obj in entry:
                if isinstance(obj, dict):
                    values.append(obj.get("overall_quality_category"))
        elif isinstance(entry, dict):
            values.append(entry.get("overall_quality_category"))
        # map to actual Conclusion round (nearest)
        resolved_round = _find_stage_round(stage_presence, round_hint, "Conclusion")
        if resolved_round is not None and values:
            pre_decision_candidates[resolved_round].extend(values)

    # Walk all audit entries for metrics
    for key, entry in (audits or {}).items():
        raw_round, stage_name = _audit_stage_hint(key)
        if stage_name is None:
            continue
        round_hint = raw_round if raw_round is not None else final_round
        resolved_round = _find_stage_round(stage_presence, round_hint, stage_name)
        if resolved_round is None:
            continue
        if stage_name not in stage_presence.get(resolved_round, set()):
            continue

        label = f"R{resolved_round} {stage_name}"
        metrics = ensure_stage(label)

        entries: List[Dict[str, Any]]
        if isinstance(entry, list):
            entries = [obj for obj in entry if isinstance(obj, dict)]
        elif isinstance(entry, dict):
            entries = [entry]
        else:
            entries = []

        for obj in entries:
            # Clinical Priority
            level = obj.get("diagnostic_urgency_level")
            if level is not None:
                metrics["Clinical Priority Mismatch Rate"]["denominator"] += 1
                if _is_priority_mismatch(level):
                    metrics["Clinical Priority Mismatch Rate"]["numerator"] += 1

            # Domain-Specific
            specialized = obj.get("specialized_insight_emergence")
            if specialized is not None:
                metrics["Domain-Specific Activation Rate"]["denominator"] += 1
                if _is_specialized_activated(specialized):
                    metrics["Domain-Specific Activation Rate"]["numerator"] += 1

            # Decision Quality
            if stage_name == "Conclusion":
                decision_category = obj.get("overall_quality_category")
                if decision_category is not None:
                    metrics["Decision Quality Shortfall Rate"]["denominator"] += 1
                    doctor_categories = pre_decision_candidates.get(resolved_round, [])
                    if _decision_quality_shortfall(decision_category, doctor_categories):
                        metrics["Decision Quality Shortfall Rate"]["numerator"] += 1

    # Ensure every detected stage in stage_presence appears in the metrics map,
    # even when no audit entries existed for that stage (renders as 0%).
    for r, stages in (stage_presence or {}).items():
        for stg in stages:
            label = f"R{r} {stg}"
            ensure_stage(label)

    return final_round, stage_metrics


# -----------------------
# Aggregate across files/roots
# -----------------------
def _iter_json_files(roots: Iterable[str], files: Iterable[str]) -> Iterable[str]:
    """Yield JSON result files from dir roots and explicit file list."""
    seen = set()
    for f in files or []:
        if f and os.path.isfile(f) and f.endswith(".json") and f not in seen:
            seen.add(f)
            yield f
    for root in roots or []:
        if not os.path.exists(root):
            continue
        if os.path.isfile(root) and root.endswith(".json"):
            if root not in seen:
                seen.add(root)
                yield root
            continue
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if not fn.endswith("-result.json") and not fn.endswith(".json"):
                        continue
                    full = os.path.join(dirpath, fn)
                    if full not in seen:
                        seen.add(full)
                        yield full


def aggregate_stage_metrics(
    observation_roots: List[str],
    files: Optional[List[str]] = None,
    max_round: int = 3,
    debug: bool = False,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    stage_totals: Dict[str, Dict[str, Dict[str, int]]] = {}

    for file_path in _iter_json_files(observation_roots, files or []):
        try:
            data = load_json(file_path)
        except Exception as exc:
            print(f"Failed to load {file_path}: {exc}")
            continue

        framework = _detect_framework(file_path, data)
        stage_presence, detected_final_round = _extract_stage_presence(data, framework)
        if not stage_presence or detected_final_round <= 0:
            if debug:
                print(f"[Skip] No stage presence in {os.path.basename(file_path)}")
            continue

        final_round = min(detected_final_round, max_round) if max_round else detected_final_round
        effective_stage_presence = {r: s for r, s in stage_presence.items() if r <= final_round}
        if not effective_stage_presence:
            if debug:
                print(f"[Skip] Empty effective stage presence in {os.path.basename(file_path)}")
            continue

        case_history = data.get("case_history") or {}
        audit_trail = case_history.get("audit_trail") or data.get("audit_trail") or {}
        audits = audit_trail.get("collaboration_audits") or {}

        if debug:
            prs = ", ".join(f"R{r}:" + "/".join(sorted(list(stages))) for r, stages in sorted(effective_stage_presence.items()))
            print(f"[Stages] {os.path.basename(file_path)} -> {prs}")

        final_round_for_case, stage_counts = _collect_stage_counts_for_case(
            case_history,
            audits,
            effective_stage_presence,
            final_round,
        )
        if final_round_for_case == 0:
            if debug:
                print(f"[Skip] No final round for {os.path.basename(file_path)}")
            continue

        # aggregate
        for stage_label, metrics in stage_counts.items():
            agg_metrics = stage_totals.setdefault(stage_label, {})
            for metric_name, stats in metrics.items():
                agg_stats = agg_metrics.setdefault(metric_name, {"numerator": 0, "denominator": 0})
                agg_stats["numerator"] += stats.get("numerator", 0)
                agg_stats["denominator"] += stats.get("denominator", 0)

    return stage_totals


# -----------------------
# Plotting
# -----------------------
def plot_stage_metrics(
    stage_counts: Dict[str, Dict[str, Dict[str, int]]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    if not stage_counts:
        print("No data to plot; skipping figure.")
        return

    # Show ALL stages (even if denominators are zero) to avoid hiding valid stages.
    stages = [stage for stage in sorted(stage_counts.keys(), key=_stage_sort_key)]
    if not stages:
        print("No stages to plot; skipping figure.")
        return

    x = np.arange(len(stages))
    bar_width = 0.4
    offsets = np.linspace(
        -bar_width * (len(CATEGORY_ORDER) - 1) / 2,
        bar_width * (len(CATEGORY_ORDER) - 1) / 2,
        len(CATEGORY_ORDER),
    )

    fig_width = max(12, len(stages) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    bar_containers = []
    colors = sns.color_palette("pastel")

    for offset, category, category_index in zip(offsets, CATEGORY_ORDER, range(len(CATEGORY_ORDER))):
        heights = []
        for stage in stages:
            stats = stage_counts.get(stage, {}).get(category, {"numerator": 0, "denominator": 0})
            denom = stats.get("denominator", 0)
            value = (stats.get("numerator", 0) / denom * 100) if denom else 0
            heights.append(value)
        bars = ax.bar(
            x + offset,
            heights,
            width=bar_width,
            label=category if category != "Decision Quality Shortfall Rate" else "Voting Decision Rate",
            color=colors[category_index],
        )
        bar_containers.append((bars, heights))

    # add value labels
    for bars, heights in bar_containers:
        for bar, height in zip(bars, heights):
            if height == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=16,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=45, ha="right", fontsize=16)
    ax.set_xlabel("Stage", fontsize=32)
    ax.set_ylabel("Percentage (%)", fontsize=32)
    ax.tick_params(axis="x", labelsize=16)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=12, loc="upper left")
    plt.tight_layout(rect=[0, 0.05, 1, 0.88])

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=500, format='pdf', bbox_inches='tight')
        print(f"Saved mechanism-3 chart to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------
# CLI
# -----------------------
def _print_stage_table(stage_counts: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    print("Collaboration quality percentages by stage:")
    report_lines = ["Collaboration quality percentages by stage:"]
    for stage in sorted(stage_counts.keys(), key=_stage_sort_key):
        stats_map = stage_counts[stage]
        details = []
        for category in CATEGORY_ORDER:
            st = stats_map.get(category, {})
            num = st.get("numerator", 0)
            den = st.get("denominator", 0)
            if den:
                details.append(f"{category}: {num/den*100:.2f}% ({num}/{den})")
            else:
                details.append(f"{category}: n/a (0/0)")
        line = f"  {stage}: " + ", ".join(details)
        print(line)
        report_lines.append(line)
    # Persist a text report next to the figure directory if possible
    # This function doesn't know save_dir; handled in __main__ below.
    return report_lines


if __name__ == "__main__":
    import argparse

    default_roots = [
        os.path.join("observation", "ColaCare"),
        os.path.join("observation", "MedAgent"),
        os.path.join("observation", "HealthcareAgent"),
        os.path.join("observation", "MAC"),
        os.path.join("observation", "MDAgents"),
        os.path.join("observation", "ReConcile"),
    ]

    parser = argparse.ArgumentParser(
        description="Aggregate mechanism-3 quality signals per stage across multiple observation sources."
    )
    parser.add_argument(
        "--observation-roots",
        nargs="+",
        default=None,
        help="One or more directories OR JSON files. Directories will be searched recursively for *-result.json / *.json.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit JSON files to include (in addition to roots).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join("figures", "mechanism3", "combined"),
        help="Directory to save the grouped bar chart.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure; only save it.",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=3,
        help="Cap the maximum round considered (default: 3).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print stage presence per file for debugging.",
    )
    args = parser.parse_args()

    observation_roots = args.observation_roots or default_roots
    stage_counts = aggregate_stage_metrics(
        observation_roots=observation_roots,
        files=args.files,
        max_round=args.max_round,
        debug=args.debug,
    )

    if not stage_counts:
        print("No stages with collaboration quality data; nothing to report.")
        raise SystemExit(0)

    report_lines = _print_stage_table(stage_counts)

    save_dir = args.save_dir
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "stage_collaboration_quality.pdf")

    # Save text report alongside the figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, 'mechanism3_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f'Saved text report to: {report_path}')

    plot_stage_metrics(
        stage_counts,
        title="Collaboration Quality Signals by Stage",
        save_path=save_path,
        show=False,
    )
