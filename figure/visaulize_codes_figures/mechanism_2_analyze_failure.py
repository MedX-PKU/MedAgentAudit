import os
import re
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
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

STAGE_SEQUENCE = ['Propose', 'Synthesis', 'Review', 'Conclusion']
STAGE_ORDER = {name: idx for idx, name in enumerate(STAGE_SEQUENCE)}
STAGE_LABEL_PATTERN = re.compile(
    r'^R(?P<round>\d+)\s+(?P<stage>Propose|Synthesis|Review|Conclusion)$'
)
STAGE_KEY_TO_NAME = {
    'analysis': 'Propose',
    'review': 'Review',
    'initial': 'Propose',
    'discussion': 'Review',
}
CATEGORY_ORDER = [
    'Rate of Successful Minority Correction',
    'Rate of Negative Majority Assimilation',
    'Rate of Robust Majority Resilience',
    'Rate of Minority-Induced Derailment',
]
CATEGORY_COLORS = {
    'Rate of Successful Minority Correction': '#0072B2',
    'Rate of Negative Majority Assimilation': '#D55E00',
    'Rate of Robust Majority Resilience': '#009E73',
    'Rate of Minority-Induced Derailment': '#CC79A7',
}


def _case_has_stage(stage_presence: Dict[int, Set[str]], round_num: int, stage_name: str) -> bool:
    return stage_name in stage_presence.get(round_num, set())


def _stage_sort_key(stage_label: str) -> Tuple[int, int, str]:
    match = STAGE_LABEL_PATTERN.match(stage_label)
    if match:
        round_num = int(match.group('round'))
        stage = match.group('stage')
        return round_num, STAGE_ORDER.get(stage, 99), stage_label
    return 10**9, 10**9, stage_label


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
        for round_entry in case_history.get('rounds', []) or []:
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
        for round_entry in case_history.get('rounds', []) or []:
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
        for round_entry in case_history.get('rounds', []) or []:
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
        for step in case_history.get('steps', []) or []:
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
        for item in (case_history.get('discussion_history') or []):
            stage_name = RECONCILE_PHASE_TO_STAGE.get(item.get('phase'))
            if stage_name:
                add_stage(1, stage_name)
        if case_history.get('final_decision'):
            add_stage(1, 'Conclusion')
        return stage_presence, 1 if stage_presence.get(1) else 0

    for round_entry in case_history.get('rounds', []) or []:
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


def _parse_step(step_name: str) -> Tuple[Optional[int], Optional[str]]:
    if not step_name:
        return None, None
    normalized = (step_name or '').strip()
    match = re.match(r'round_(\d+)_(.+)', normalized)
    if match:
        round_idx = int(match.group(1))
        raw_stage = match.group(2).strip().lower()
        if raw_stage.endswith('_initial'):
            stage_key = 'initial'
        elif raw_stage.endswith('_analysis'):
            stage_key = 'analysis'
        elif raw_stage.endswith('_review'):
            stage_key = 'review'
        elif raw_stage.endswith('_discussion'):
            stage_key = 'discussion'
        else:
            stage_key = raw_stage
        return round_idx, stage_key

    lowered = normalized.lower()
    if lowered.endswith('_initial') or 'initial' in lowered:
        return 1, 'initial'
    if lowered.endswith('_analysis'):
        return 1, 'analysis'
    if lowered.endswith('_review'):
        return 1, 'review'
    if lowered.endswith('_discussion'):
        return 1, 'discussion'
    return None, None

def _collect_doctor_answers(viewpoints: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[int, Dict[str, str]]]:
    doctor_answers: Dict[str, Dict[int, Dict[str, str]]] = {}
    for agent_id, history in (viewpoints or {}).items():
        if not isinstance(history, list):
            continue
        agent_rounds: Dict[int, Dict[str, str]] = defaultdict(dict)
        for entry in history:
            if not isinstance(entry, dict):
                continue
            round_idx, stage_key = _parse_step(entry.get('step', ''))
            if round_idx is None or stage_key is None:
                continue
            stage_name = STAGE_KEY_TO_NAME.get(stage_key)
            if not stage_name:
                continue
            answer = entry.get('viewpoint')
            if answer:
                agent_rounds[round_idx][stage_name] = answer
        if agent_rounds:
            doctor_answers[agent_id] = dict(agent_rounds)
    return doctor_answers


def _collect_meta_answers(
    data: Dict[str, Any],
    framework: str,
    stage_presence: Dict[int, Set[str]],
) -> Dict[int, Dict[str, str]]:
    meta_answers: Dict[int, Dict[str, str]] = {}
    case_history = data.get('case_history', {}) or {}

    def add_meta(round_hint: int, stage_name: str, value: Optional[str]) -> None:
        if not value:
            return
        resolved_round = _find_stage_round(stage_presence, round_hint, stage_name)
        if resolved_round is None:
            return
        meta_answers.setdefault(resolved_round, {})[stage_name] = value

    if framework == 'ColaCare':
        for round_entry in case_history.get('rounds', []) or []:
            round_idx = _safe_int(round_entry.get('round'))
            if round_idx <= 0:
                continue
            synthesis_answer = (
                round_entry.get('synthesis', {})
                .get('parsed_output', {})
                .get('answer')
            )
            add_meta(round_idx, 'Synthesis', synthesis_answer)
            decision_answer = (
                round_entry.get('decision', {})
                .get('parsed_output', {})
                .get('answer')
            )
            add_meta(round_idx, 'Conclusion', decision_answer)
        return meta_answers

    if framework == 'MedAgent':
        for round_entry in case_history.get('rounds', []) or []:
            round_idx = _safe_int(round_entry.get('round'))
            if round_idx <= 0:
                continue
            synthesis_answer = (
                round_entry.get('synthesis', {})
                .get('parsed_output', {})
                .get('answer')
            )
            add_meta(round_idx, 'Synthesis', synthesis_answer)
        final_decision = (
            case_history.get('final_decision_log', {})
            .get('parsed_output', {})
            .get('answer')
        )
        if final_decision:
            target_round = max(stage_presence.keys(), default=1)
            add_meta(target_round, 'Conclusion', final_decision)
        return meta_answers

    if framework == 'MAC':
        for round_entry in case_history.get('rounds', []) or []:
            round_idx = _safe_int(round_entry.get('round'))
            if round_idx <= 0:
                continue
            supervisor = round_entry.get('supervisor_response', {}) or {}
            parsed_output = supervisor.get('parsed_output', {}) or {}
            summary = parsed_output.get('summary')
            final_info = parsed_output.get('final_answer') or {}
            final_answer = final_info.get('answer') if isinstance(final_info, dict) else None
            add_meta(round_idx, 'Synthesis', summary)
            add_meta(round_idx, 'Conclusion', final_answer)
        return meta_answers

    if framework == 'HealthcareAgent':
        for step in case_history.get('steps', []) or []:
            step_name = step.get('step')
            log = step.get('log') or {}
            parsed_output_str = log.get('parsed_output_str')
            parsed_obj: Optional[Dict[str, Any]] = None
            if isinstance(parsed_output_str, str):
                try:
                    parsed_obj = json.loads(parsed_output_str)
                except json.JSONDecodeError:
                    parsed_obj = None
            answer: Optional[str] = None
            if isinstance(parsed_obj, dict):
                answer = parsed_obj.get('answer') or parsed_obj.get('final_answer')
            if step_name == '3_Preliminary_Analysis':
                add_meta(1, 'Synthesis', answer)
            if step_name == '5_Final_Modification':
                add_meta(1, 'Conclusion', answer)
        return meta_answers

    if framework == 'MDAgents':
        predicted = data.get('predicted_answer')
        if predicted:
            add_meta(1, 'Conclusion', predicted)
        return meta_answers

    if framework == 'ReConcile':
        for item in case_history.get('discussion_history', []) or []:
            if item.get('phase') == 'final':
                add_meta(1, 'Conclusion', item.get('response'))
        final_decision = case_history.get('final_decision')
        if final_decision:
            add_meta(1, 'Conclusion', final_decision)
        return meta_answers

    for round_entry in case_history.get('rounds', []) or []:
        round_idx = _safe_int(round_entry.get('round'))
        if round_idx <= 0:
            continue
        synthesis_answer = (
            round_entry.get('synthesis', {})
            .get('parsed_output', {})
            .get('answer')
        )
        add_meta(round_idx, 'Synthesis', synthesis_answer)
        decision_answer = (
            round_entry.get('decision', {})
            .get('parsed_output', {})
            .get('answer')
        )
        add_meta(round_idx, 'Conclusion', decision_answer)
    return meta_answers

def _compute_stage_categories_for_case(
    doctor_answers: Dict[str, Dict[int, Dict[str, str]]],
    meta_answers: Dict[int, Dict[str, str]],
    stage_presence: Dict[int, Set[str]],
    ground_truth: str,
    final_round: int,
) -> Tuple[int, Dict[str, str]]:
    agent_state: Dict[str, Optional[str]] = {agent_id: None for agent_id in doctor_answers.keys()}
    agent_state['meta'] = None
    stage_categories: Dict[str, str] = {}
    initial_majority_correct: Optional[bool] = None
    last_round_with_stage = 0

    def evaluate_current_majority() -> Optional[bool]:
        answers = [ans for ans in agent_state.values() if ans]
        if not answers:
            return None
        correct = sum(1 for ans in answers if ans == ground_truth)
        return correct > len(answers) / 2

    for round_idx in range(1, final_round + 1):
        stages_available = stage_presence.get(round_idx, set())

        if 'Propose' in stages_available:
            for agent_id, rounds_map in doctor_answers.items():
                answer = rounds_map.get(round_idx, {}).get('Propose')
                if answer:
                    agent_state[agent_id] = answer
            current_majority = evaluate_current_majority()
            if initial_majority_correct is None and current_majority is not None:
                initial_majority_correct = current_majority
            if current_majority is not None and initial_majority_correct is not None:
                stage_categories[f'R{round_idx} Propose'] = _classify(initial_majority_correct, current_majority)
                last_round_with_stage = max(last_round_with_stage, round_idx)

        if 'Synthesis' in stages_available:
            synthesis_answer = meta_answers.get(round_idx, {}).get('Synthesis')
            if synthesis_answer:
                agent_state['meta'] = synthesis_answer
            current_majority = evaluate_current_majority()
            if current_majority is not None and initial_majority_correct is not None:
                stage_categories[f'R{round_idx} Synthesis'] = _classify(initial_majority_correct, current_majority)
                last_round_with_stage = max(last_round_with_stage, round_idx)

        if 'Review' in stages_available:
            for agent_id, rounds_map in doctor_answers.items():
                answer = rounds_map.get(round_idx, {}).get('Review')
                if answer:
                    agent_state[agent_id] = answer
            current_majority = evaluate_current_majority()
            if current_majority is not None and initial_majority_correct is not None:
                stage_categories[f'R{round_idx} Review'] = _classify(initial_majority_correct, current_majority)
                last_round_with_stage = max(last_round_with_stage, round_idx)

        if 'Conclusion' in stages_available:
            conclusion_answer = meta_answers.get(round_idx, {}).get('Conclusion')
            if conclusion_answer:
                agent_state['meta'] = conclusion_answer
            current_majority = evaluate_current_majority()
            if current_majority is not None and initial_majority_correct is not None:
                stage_categories[f'R{round_idx} Conclusion'] = _classify(initial_majority_correct, current_majority)
                last_round_with_stage = max(last_round_with_stage, round_idx)

    return last_round_with_stage, stage_categories


def _classify(initial_majority_correct: bool, current_majority_correct: bool) -> str:
    if not initial_majority_correct:
        if current_majority_correct:
            return 'Rate of Successful Minority Correction'
        return 'Rate of Negative Majority Assimilation'
    if current_majority_correct:
        return 'Rate of Robust Majority Resilience'
    return 'Rate of Minority-Induced Derailment'


def aggregate_viewpoint_categories_by_stage(
    observation_roots: List[str],
    max_round: int = 3,
) -> Dict[str, Counter]:
    stage_totals: Dict[str, Counter] = defaultdict(Counter)

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

                audit_source = (data.get('case_history', {}) or {}).get('audit_trail')
                if audit_source is None:
                    audit_source = data.get('audit_trail', {}) or {}
                viewpoints = audit_source.get('viewpoints') or {}
                doctor_answers = _collect_doctor_answers(viewpoints)
                if not doctor_answers:
                    continue

                ground_truth = data.get('ground_truth')
                if not ground_truth:
                    continue

                meta_answers = _collect_meta_answers(data, framework, effective_stage_presence)

                final_round_for_case, stage_categories = _compute_stage_categories_for_case(
                    doctor_answers,
                    meta_answers,
                    effective_stage_presence,
                    ground_truth,
                    final_round,
                )
                if final_round_for_case == 0:
                    continue

                stage_labels_for_case: Set[str] = set()
                for round_num in range(1, final_round_for_case + 1):
                    for stage_name in STAGE_SEQUENCE:
                        if _case_has_stage(effective_stage_presence, round_num, stage_name):
                            stage_labels_for_case.add(f'R{round_num} {stage_name}')

                if not stage_labels_for_case:
                    continue

                for label in stage_labels_for_case:
                    stage_totals[label]['__total__'] += 1

                for stage_label, category in stage_categories.items():
                    if stage_label not in stage_labels_for_case:
                        continue
                    stage_totals[stage_label][category] += 1

    return stage_totals

def plot_stage_category_counts(
    stage_counts: Dict[str, Counter],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    if not stage_counts:
        print('No data to plot; skipping figure.')
        return

    stages = [
        stage
        for stage in sorted(stage_counts.keys(), key=_stage_sort_key)
        if stage_counts[stage].get('__total__')
    ]
    if not stages:
        print('No stages with valid totals; skipping figure.')
        return

    # --- 代码修改开始 ---

    n_categories = len(CATEGORY_ORDER)
    # 您现在可以安全地增加 bar_width，而不会导致条形图重叠
    bar_width = 0.5

    # 定义不同组（例如 R1 Propose 和 R1 Synthesis）之间的间隙大小
    group_gap = 0.3

    # 计算一个组内所有条形的总宽度
    total_group_width = n_categories * bar_width

    # 计算每个组中心点之间的步长（距离）
    # 这是解决重叠问题的关键：确保组间距足以容纳所有条形加上一个额外的间隙
    group_step = total_group_width + group_gap

    # 根据新的步长计算每个组在 X 轴上的中心位置
    x = np.arange(len(stages)) * group_step

    # 组内每个条形相对于其组中心的偏移量计算保持不变，
    # 因为它能正确地将条形并排排列。
    offsets = np.linspace(
        -bar_width * (n_categories - 1) / 2,
        bar_width * (n_categories - 1) / 2,
        n_categories,
    )

    # --- 代码修改结束 ---

    fig_width = max(16, len(stages) * group_step * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    stage_annotations: Dict[int, List[Tuple[float, plt.Rectangle, int]]] = defaultdict(list)
    max_height = 0.0

    colors = sns.color_palette("pastel")

    for offset, category, category_index in zip(offsets, CATEGORY_ORDER, range(len(CATEGORY_ORDER))):
        heights = []
        for stage_idx, stage in enumerate(stages):
            counter = stage_counts[stage]
            total = counter.get('__total__', 0)
            value = (counter.get(category, 0) / total * 100) if total else 0
            heights.append(value)
        # 在新的 X 轴位置 (x + offset) 处绘制条形图
        bars = ax.bar(
            x + offset,
            heights,
            width=bar_width,
            label=category,
            color=colors[category_index],
        )
        category_index = CATEGORY_ORDER.index(category)
        for stage_idx, (bar, height) in enumerate(zip(bars, heights)):
            stage_annotations[stage_idx].append((height, bar, category_index))
            max_height = max(max_height, height)

    base_offset = max(max_height * 0.02, 0.5)
    for stage_idx, items in stage_annotations.items():
        sorted_items = sorted(items, key=lambda item: item[0], reverse=True)
        stage_name = stages[stage_idx]
        for position, (height, bar, category_idx) in enumerate(sorted_items):
            if height == 0:
                continue
            text_y = bar.get_height() + 0.1
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=12,
            )

    ax.set_ylim(0, max(105, max_height + base_offset * 5))

    # 使用新的 x 值来设置刻度线位置
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=45, ha='right', fontsize=16)
    ax.set_xlabel('Stage', fontsize=32)
    ax.set_ylabel('Percentage (%)', fontsize=32)
    ax.tick_params(axis="x", labelsize=16)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, 105)
    ax.legend(ncol=2, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.88])

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=500, format='pdf', bbox_inches='tight')
        print(f'Saved stage category chart to: {save_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    import argparse

    default_roots = [
        os.path.join('observation', 'ColaCare'),
        os.path.join('observation', 'MedAgent'),
        os.path.join('observation', 'HealthcareAgent'),
        os.path.join('observation', 'MAC'),
        os.path.join('observation', 'MDAgents'),
        os.path.join('observation', 'ReConcile'),
    ]

    parser = argparse.ArgumentParser(
        description='Aggregate viewpoint dynamics (Mechanism 2) per stage across multiple observation sources.'
    )
    parser.add_argument(
        '--observation-roots',
        nargs='+',
        default=None,
        help='One or more directories containing observation result JSON files.',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=os.path.join('figures', 'mechanism2', 'combined'),
        help='Directory to save grouped bar charts (one per termination round).',
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the figures; only save them.',
    )
    args = parser.parse_args()

    observation_roots = args.observation_roots or default_roots
    stage_counts = aggregate_viewpoint_categories_by_stage(
        observation_roots,
        max_round=3,
    )

    print('Aggregating viewpoint dynamics from roots:')
    for root in observation_roots:
        print(f'  - {root}')

    stages = [stage for stage in sorted(stage_counts.keys(), key=_stage_sort_key) if stage_counts[stage].get('__total__')]
    if not stages:
        print('No stages with viewpoint data; nothing to report.')
        raise SystemExit(0)

    print('Viewpoint dynamics percentages by stage:')
    report_lines = ['Viewpoint dynamics percentages by stage:']
    for stage in stages:
        counter = stage_counts[stage]
        total = counter.get('__total__', 0)
        if not total:
            continue
        pct_details = ', '.join(
            f"{category}: {counter.get(category, 0) / total * 100:.2f}%"
            for category in CATEGORY_ORDER
        )
        line = f'  {stage}: {pct_details} (total cases={total})'
        print(line)
        report_lines.append(line)

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'stage_viewpoint_shift.pdf')
    else:
        save_path = None

    # Save text report alongside the figure
    report_path = os.path.join(save_dir, 'mechanism2_report.txt') if save_dir else None
    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines) + '\n')
        print(f'Saved text report to: {report_path}')

    plot_stage_category_counts(
        stage_counts,
        title='Viewpoint Dynamics by Stage',
        save_path=save_path,
        show=False,
    )
