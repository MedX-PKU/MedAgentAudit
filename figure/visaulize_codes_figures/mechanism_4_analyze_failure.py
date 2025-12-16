import os
import re
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: When file doesn't exist
        json.JSONDecodeError: When JSON format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

FRAMEWORK_NAMES = (
    'ColaCare',
    'MedAgent',
    'HealthcareAgent',
    'MAC',
    'MDAgents',
    'ReConcile',
)

HEALTHCARE_STEP_TO_STAGE = {
    '1_Planner': 'Propose',
    '2_Inquiry': 'Propose',
    '3_Preliminary_Analysis': 'Synthesis',
    '4_Safety_Review': 'Review',
    '5_Final_Modification': 'Conclusion',
}

MDA_STEP_TO_STAGE = {
    'determine_complexity': 'Propose',
    'basic_initial': 'Propose',
    'intermediate_initial': 'Propose',
    'advanced_initial': 'Propose',
    'recruit_experts': 'Conclusion',
}

RECONCILE_PHASE_TO_STAGE = {
    'initial': 'Propose',
    'discussion': 'Review',
    'final': 'Conclusion',
}


STAGE_SEQUENCE = ["Propose", "Synthesis", "Review", "Conclusion"]
STAGE_ORDER = {name: idx for idx, name in enumerate(STAGE_SEQUENCE)}
STAGE_LABEL_PATTERN = re.compile(
    r"^R(?P<round>\d+)\s+(?P<stage>Propose|Synthesis|Review|Conclusion)$"
)


def _case_has_stage(stage_presence: Dict[int, Set[str]], round_num: int, stage_name: str) -> bool:
    return stage_name in stage_presence.get(round_num, set())


def _stage_sort_key(stage_label: str) -> Tuple[int, int, str]:
    match = STAGE_LABEL_PATTERN.match(stage_label)
    if match:
        round_num = int(match.group("round"))
        stage = match.group("stage")
        return (round_num, STAGE_ORDER.get(stage, 99), stage_label)
    return (10**9, 10**9, stage_label)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0



def _detect_framework(file_path: str) -> str:
    parts = set(os.path.normpath(file_path).split(os.sep))
    for name in FRAMEWORK_NAMES:
        if name in parts:
            return name
    return 'ColaCare'



def _extract_stage_presence(data: Dict[str, Any], framework: str) -> Tuple[Dict[int, Set[str]], int]:
    stage_presence: Dict[int, Set[str]] = {}
    case_history = data.get('case_history', {}) or {}

    def add_stage(round_num: int, stage_name: str) -> None:
        if round_num <= 0:
            return
        stage_presence.setdefault(round_num, set()).add(stage_name)

    if framework == 'ColaCare':
        rounds = case_history.get('rounds', []) or []
        for round_entry in rounds:
            round_num = _safe_int(round_entry.get('round'))
            if round_num <= 0:
                continue
            if round_entry.get('opinions'):
                add_stage(round_num, 'Propose')
            if round_entry.get('synthesis'):
                add_stage(round_num, 'Synthesis')
            if round_entry.get('reviews'):
                add_stage(round_num, 'Review')
            if round_entry.get('decision'):
                add_stage(round_num, 'Conclusion')
        final_round = max(stage_presence.keys(), default=0)
        return stage_presence, final_round

    if framework == 'MedAgent':
        rounds = case_history.get('rounds', []) or []
        for round_entry in rounds:
            round_num = _safe_int(round_entry.get('round'))
            if round_num <= 0:
                continue
            if round_entry.get('analyses'):
                add_stage(round_num, 'Propose')
            if round_entry.get('synthesis'):
                add_stage(round_num, 'Synthesis')
            if round_entry.get('reviews'):
                add_stage(round_num, 'Review')
        total_rounds = _safe_int(case_history.get('total_rounds'))
        final_round = max(stage_presence.keys(), default=0)
        if total_rounds > 0:
            final_round = max(final_round, total_rounds)
        if case_history.get('final_decision_log'):
            decision_round = final_round if final_round > 0 else (total_rounds if total_rounds > 0 else 1)
            add_stage(decision_round, 'Conclusion')
            final_round = max(final_round, decision_round)
        return stage_presence, final_round

    if framework == 'MAC':
        rounds = case_history.get('rounds', []) or []
        for round_entry in rounds:
            round_num = _safe_int(round_entry.get('round'))
            if round_num <= 0:
                continue
            if round_entry.get('doctor_responses'):
                add_stage(round_num, 'Propose')
            if round_entry.get('supervisor_response'):
                add_stage(round_num, 'Synthesis')
                add_stage(round_num, 'Conclusion')
        final_round = max(stage_presence.keys(), default=0)
        return stage_presence, final_round

    if framework == 'HealthcareAgent':
        steps = case_history.get('steps', []) or []
        for step in steps:
            stage_name = HEALTHCARE_STEP_TO_STAGE.get(step.get('step'))
            if stage_name:
                add_stage(1, stage_name)
        final_round = 1 if stage_presence.get(1) else 0
        return stage_presence, final_round

    if framework == 'MDAgents':
        process_log = data.get('process_log', []) or []
        if process_log:
            for entry in process_log:
                step_name = entry.get('step_name') or entry.get('step')
                if not isinstance(step_name, str):
                    continue
                base_name = step_name.split(':', 1)[0]
                if base_name in MDA_STEP_TO_STAGE:
                    add_stage(1, MDA_STEP_TO_STAGE[base_name])
                elif base_name.endswith('_initial') or 'initial' in base_name:
                    add_stage(1, 'Propose')
                elif base_name == 'recruit_experts':
                    add_stage(1, 'Conclusion')
            if data.get('predicted_answer'):
                add_stage(1, 'Conclusion')
            return stage_presence, 1 if stage_presence.get(1) else 0
        return stage_presence, 0

    if framework == 'ReConcile':
        history = case_history.get('discussion_history', []) or []
        for item in history:
            stage_name = RECONCILE_PHASE_TO_STAGE.get(item.get('phase'))
            if stage_name:
                add_stage(1, stage_name)
        if case_history.get('final_decision'):
            add_stage(1, 'Conclusion')
        return stage_presence, 1 if stage_presence.get(1) else 0

    rounds = case_history.get('rounds', []) or []
    for round_entry in rounds:
        round_num = _safe_int(round_entry.get('round'))
        if round_num <= 0:
            continue
        if round_entry.get('opinions'):
            add_stage(round_num, 'Propose')
        if round_entry.get('synthesis'):
            add_stage(round_num, 'Synthesis')
        if round_entry.get('reviews'):
            add_stage(round_num, 'Review')
        if round_entry.get('decision'):
            add_stage(round_num, 'Conclusion')
    final_round = max(stage_presence.keys(), default=0)
    return stage_presence, final_round



def _find_stage_round(stage_presence: Dict[int, Set[str]], preferred_round: int, stage_name: str) -> Optional[int]:
    if not stage_presence:
        return None
    if stage_name in stage_presence.get(preferred_round, set()):
        return preferred_round
    sorted_rounds = sorted(stage_presence.keys())
    for round_num in sorted_rounds:
        if round_num >= preferred_round and stage_name in stage_presence.get(round_num, set()):
            return round_num
    for round_num in reversed(sorted_rounds):
        if round_num <= preferred_round and stage_name in stage_presence.get(round_num, set()):
            return round_num
    return None



def _determine_intro_stage(conflicting_agents: Any) -> str:
    if not isinstance(conflicting_agents, (list, tuple)):
        return "Propose"
    for agent in conflicting_agents:
        if isinstance(agent, str) and 'meta' in agent.strip().lower():
            return "Synthesis"
    return "Propose"


def collect_case_ccp_stage_sets(
    ccps: Dict[str, Any],
    final_round: int,
) -> Dict[str, Set[str]]:
    stage_sets: Dict[str, Set[str]] = {}
    if not isinstance(ccps, dict):
        return stage_sets

    final_round = max(final_round, 1)
    final_time = (final_round, STAGE_ORDER["Conclusion"])

    for round_key, entries in ccps.items():
        if not entries:
            continue
        round_hint = _safe_int(round_key)
        for ccp in entries:
            if not isinstance(ccp, dict):
                continue

            ccp_id = str(ccp.get("ccp_id", f"{round_key}"))
            intro_round = _safe_int(ccp.get("round_identified")) or round_hint or 1
            intro_round = max(1, min(intro_round, final_round))

            intro_stage = _determine_intro_stage(ccp.get("conflicting_agents"))
            intro_stage_idx = STAGE_ORDER[intro_stage]
            intro_time = (intro_round, intro_stage_idx)

            status = (ccp.get("status") or "").lower()
            if status == "resolved":
                res_round = _safe_int(ccp.get("round_resolved")) or intro_round
                res_round = max(intro_round, min(res_round, final_round))
                res_time: Optional[Tuple[int, int]] = (res_round, STAGE_ORDER["Review"])
            else:
                res_time = None

            for round_num in range(intro_round, final_round + 1):
                if res_time and round_num > res_time[0]:
                    break

                for stage_idx, stage_name in enumerate(STAGE_SEQUENCE):
                    stage_time = (round_num, stage_idx)
                    if stage_time < intro_time:
                        continue
                    if res_time and stage_time >= res_time:
                        break
                    if stage_time > final_time:
                        break

                    label = f"R{round_num} {stage_name}"
                    stage_sets.setdefault(label, set()).add(ccp_id)

                if res_time and round_num >= res_time[0]:
                    break

    return stage_sets


def aggregate_ccp_counts_by_final_round(
    observation_roots: List[str],
    max_round: int = 3,
) -> Dict[str, Dict[str, int]]:
    stage_totals: Dict[str, Dict[str, int]] = {}

    for observation_root in observation_roots:
        if not os.path.isdir(observation_root):
            print(f"Observation root '{observation_root}' does not exist; skipping.")
            continue

        for dirpath, _, filenames in os.walk(observation_root):
            for filename in filenames:
                if not filename.endswith('-result.json'):
                    continue
                file_path = os.path.join(dirpath, filename)
                try:
                    data = load_json(file_path)
                except Exception as exc:
                    print(f"Failed to load {file_path}: {exc}")
                    continue

                framework = _detect_framework(file_path)
                case_history = data.get('case_history', {}) or {}
                audit_trail = (case_history.get('audit_trail') or data.get('audit_trail') or {})
                ccps = audit_trail.get('ccps', {}) or {}

                ccp_ids = set()
                for entries in ccps.values():
                    if isinstance(entries, list):
                        for ccp in entries:
                            if isinstance(ccp, dict) and ccp.get('ccp_id'):
                                ccp_ids.add(ccp['ccp_id'])
                if not ccp_ids:
                    continue

                stage_presence, detected_final_round = _extract_stage_presence(data, framework)
                if not stage_presence or detected_final_round <= 0:
                    continue

                final_round = detected_final_round
                if max_round and final_round > max_round:
                    final_round = max_round

                effective_stage_presence = {
                    round_num: stages
                    for round_num, stages in stage_presence.items()
                    if round_num <= final_round
                }
                if not effective_stage_presence:
                    continue

                total_ccp = len(ccp_ids)
                stage_sets = collect_case_ccp_stage_sets(ccps, final_round)

                for round_num in range(1, final_round + 1):
                    for stage_name in STAGE_SEQUENCE:
                        if not _case_has_stage(effective_stage_presence, round_num, stage_name):
                            continue
                        label = f"R{round_num} {stage_name}"
                        stats = stage_totals.setdefault(label, {'numerator': 0, 'denominator': 0})
                        active_ccps = stage_sets.get(label, set()) if isinstance(stage_sets, dict) else set()
                        stats['numerator'] += len(active_ccps)
                        stats['denominator'] += total_ccp

    return stage_totals




def plot_stage_counts(
    stage_counts: Dict[str, Dict[str, int]],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    if not stage_counts:
        print("No stage data to plot; skipping figure.")
        return

    stages = [
        stage
        for stage in sorted(stage_counts.keys(), key=_stage_sort_key)
        if stage_counts[stage].get('denominator')
    ]
    if not stages:
        print("No stages with valid denominators; skipping figure.")
        return

    percentages = [
        (stage_counts[stage]['numerator'] / stage_counts[stage]['denominator']) * 100
        for stage in stages
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(stages) * 0.8), 6))

    # 1. 定义一个颜色列表，用于区分不同的聚合阶段
    colors = sns.color_palette("pastel")

    # 2. 自动为每个聚合阶段（如 'R1', 'R2'）分配一个颜色
    stage_prefixes = sorted(list(set(stage.split()[0] for stage in stages)))
    color_map = {prefix: colors[i % len(colors)] for i, prefix in enumerate(stage_prefixes)}

    # 3. 为每个柱子生成对应的颜色列表
    bar_colors = [color_map[stage.split()[0]] for stage in stages]

    fig, ax = plt.subplots(figsize=(max(8, len(stages) * 0.9), 5.5))

    # 4. 使用生成的颜色列表进行绘图
    bars = ax.bar(stages, percentages, color=bar_colors)

    ax.set_xlabel('Stage', fontsize=32)
    ax.set_ylabel('Percentage (%)', fontsize=32)

    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{pct:.1f}",
            ha='center',
            va='bottom',
            fontsize=20,
        )

    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='x', labelrotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=500, format='pdf', bbox_inches='tight')
        print(f'Saved stage percentage chart to: {save_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)


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
        description="Aggregate unresolved CCP counts per stage across multiple observation sources."
    )
    parser.add_argument(
        "--observation-roots",
        nargs="+",
        default=None,
        help="One or more directories containing observation result JSON files.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join("figures", "mechanism4", "combined"),
        help="Directory to save the generated bar charts (one per final round).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figures; only save them.",
    )
    args = parser.parse_args()

    observation_roots = args.observation_roots or default_roots
    stage_counts = aggregate_ccp_counts_by_final_round(
        observation_roots,
        max_round=3,
    )

    print("Aggregating unresolved CCP percentages from roots:")
    for root in observation_roots:
        print(f"  - {root}")

    stages = [stage for stage in sorted(stage_counts.keys(), key=_stage_sort_key) if stage_counts[stage].get('denominator')]
    if not stages:
        print("No stages with unresolved CCP data; nothing to report.")
        raise SystemExit(0)

    print("Unresolved CCP percentages by stage:")
    report_lines = ["Unresolved CCP percentages by stage:"]
    for stage in stages:
        stats = stage_counts[stage]
        denom = stats.get('denominator', 0)
        if not denom:
            continue
        pct = stats['numerator'] / denom * 100
        line = f"  {stage}: {pct:.2f}% ({stats['numerator']}/{denom})"
        print(line)
        report_lines.append(line)

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "stage_ccp_dropout.pdf")
    else:
        save_path = None

    # Save text report alongside the figure
    report_path = os.path.join(save_dir, 'mechanism4_report.txt') if save_dir else None
    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f'Saved text report to: {report_path}')

    plot_stage_counts(
        stage_counts,
        title="Cases with Unresolved CCPs per Stage",
        save_path=save_path,
        show=False,
    )
