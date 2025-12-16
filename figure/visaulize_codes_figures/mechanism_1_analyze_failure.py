import os
import json
import re
from typing import Any, Dict, Optional, Tuple, List, Set
from textwrap import fill

import matplotlib.pyplot as plt
import seaborn as sns

# from openai import OpenAI  # LLM 用于反驳判断

# from llm_configs import LLM_MODELS_SETTINGS

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

CITATION_ACTION_STAGE_MAP = {
    'review': 'Review',
    'discussion': 'Review',
    'synthesis': 'Synthesis',
    'group_synthesis': 'Synthesis',
    'summary': 'Synthesis',
}


# class AnalysisHelperLLM:
#     """
#     一个使用 LLM 来对日志进行深度语义分析的辅助类。
#     目前仅用于：判断某个 KEU 是否在讨论中被“有效反驳”。
#     """

#     def __init__(self, model_key: str = "gemini-2.5-pro"):  # 建议使用较强模型进行分析
#         if model_key not in LLM_MODELS_SETTINGS:
#             raise ValueError(f"Model key '{model_key}' not found.")

#         model_settings = LLM_MODELS_SETTINGS[model_key]
#         self.client = OpenAI(
#             api_key=model_settings["api_key"],
#             base_url=model_settings["base_url"],
#         )
#         self.model_name = model_settings["model_name"]
#         print(f"Analysis Helper LLM initialized with model: {self.model_name}")

#     def check_if_rebutted(self, keu_to_check: Dict[str, Any], case_history: Dict[str, Any]) -> bool:
#         """
#         判断一个 KEU 是否在讨论中被有效反驳。

#         返回 True/False。
#         """
#         system_message = {
#             "role": "system",
#             "content": """You are an expert debate judge and a sharp medical logician. Your task is to determine if a specific 'Claim' (a Key Evidential Unit from a minority opinion) was effectively rebutted during a medical discussion.

# Definition of 'Effectively Rebutted':
# - A simple disagreement ('I don't agree with KEU-5') is NOT a rebuttal.
# - A rebuttal requires providing specific counter-evidence or a logical argument that directly invalidates or seriously challenges the claim.
# - Example:
#   - Claim: 'The image shows a clear fracture in the tibia.'
#   - Effective Rebuttal: 'The line identified as a fracture is actually a nutrient canal, which is a normal anatomical feature. This is confirmed by its oblique orientation and sclerotic borders, unlike a typical fracture line.'
#   - Ineffective Rebuttal: 'I have checked the image and I do not see a fracture.'

# You MUST respond with a single JSON object with one key: `is_rebutted` (boolean: true or false).
# """,
#         }

#         # 构建完整的讨论上下文
#         discussion_text = ""
#         for i, round_data in enumerate(case_history.get("rounds", [])):
#             discussion_text += f"--- ROUND {i+1} ---\n"
#             synthesis = round_data.get("synthesis", {}).get("log", {}).get("parsed_output", {})
#             if synthesis:
#                 discussion_text += f"MetaAgent Synthesis: {synthesis.get('explanation', 'N/A')}\n\n"

#             for review in round_data.get("reviews", []) or []:
#                 doctor_id = review.get("doctor_id")
#                 review_output = review.get("log", {}).get("parsed_output", {})
#                 if review_output:
#                     discussion_text += f"{doctor_id} Review: {review_output.get('reason', 'N/A')}\n"
#             discussion_text += "\n"

#         user_message = {
#             "role": "user",
#             "content": (
#                 f"Here is the claim from a minority doctor:\n"
#                 f"Claim (from {keu_to_check.get('source_agent', '?')}): \"{keu_to_check.get('content', '')}\"\n\n"
#                 f"Here is the full discussion that followed:\n"
#                 f"--- DISCUSSION START ---\n"
#                 f"{discussion_text}"
#                 f"--- DISCUSSION END ---\n\n"
#                 f"Was this specific claim effectively rebutted during the discussion? Provide your judgment as a JSON object."
#             ),
#         }

#         # 调用 LLM
#         try:
#             completion = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[system_message, user_message],
#                 response_format={"type": "json_object"},
#             )
#             response_text = completion.choices[0].message.content
#             parsed_response = json.loads(preprocess_response_string(response_text))
#             return bool(parsed_response.get("is_rebutted", False))
#         except Exception as e:
#             print(f"LLM call for rebuttal check failed: {e}")
#             return False  # 保守起见，如果检查失败，则默认该 KEU 未被反驳


# AnalysisResultM1 = Tuple[bool, Optional[Dict[str, Any]]]


# def analyze_information_loss(
#     case_history: Dict[str, Any],
#     audit_trail: Dict[str, Any],
#     analysis_llm: AnalysisHelperLLM,
# ) -> AnalysisResultM1:
#     """
#     分析机制一：关键证据单元（KEU）是否在最终决策中丢失，且未被有效反驳。

#     仅当存在被审计判定为 is_key 的 KEU 时适用；
#     仅统计同时满足：is_key=True 且 present_in_final_decision=False 且 未被有效反驳 的 KEU。
#     """
#     all_keus = audit_trail.get("keus", {})

#     # 前提：日志中必须有 KEU
#     if not all_keus:
#         return False, None

#     # 1. 筛选出所有被 Auditor 标记为“关键”的 KEU
#     key_keus = {keu_id: keu for keu_id, keu in all_keus.items() if keu.get("is_key", False)}

#     # 分母：只要有关键 KEU，本分析就适用
#     if not key_keus:
#         return False, None

#     is_applicable = True
#     truly_lost_key_keus: List[Dict[str, Any]] = []

#     for keu_id, keu in key_keus.items():
#         # 2. 检查关键 KEU 是否在最终决策中缺失（直接依据 present_in_final_decision）
#         is_missing = not keu.get("present_in_final_decision", False)

#         if is_missing:
#             # 3. 如果缺失，调用 LLM 检查其是否在讨论中被有效反驳
#             print(f"  Checking if missing KEY KEU '{keu_id}' was effectively rebutted...")
#             was_rebutted = analysis_llm.check_if_rebutted(keu, case_history)

#             # 可在此更新 KEU 临时状态（仅分析阶段使用）
#             # keu['is_rebutted_by_analysis'] = was_rebutted

#             if not was_rebutted:
#                 # 仅当“关键”、“缺失”且“未被反驳”时，计为关键信息丢失
#                 truly_lost_key_keus.append(
#                     {
#                         "keu_id": keu_id,
#                         "content": keu.get("content", ""),
#                         "source_agent": keu.get("source_agent", ""),
#                         "status": "Ignored Critical Evidence",
#                     }
#                 )
#             else:
#                 print(
#                     f"  --> KEU '{keu_id}' was missing but deemed effectively rebutted. Not counted as loss."
#                 )

#     if truly_lost_key_keus:
#         failure_details = {
#             "failure_type": "Key Information Loss",
#             "details": truly_lost_key_keus,
#         }
#         print(
#             f"[FAILURE DETECTED]: Key Information Loss. {len(truly_lost_key_keus)} critical, unrebutted KEU(s) were ignored."
#         )
#         return is_applicable, failure_details

#     return is_applicable, None


# # ==========================
# # 可视化（机制一）
# # ==========================
# import matplotlib.pyplot as plt
# import numpy as np






STAGE_SEQUENCE = ["Propose", "Synthesis", "Review", "Conclusion"]
STAGE_ORDER = {
    "Propose": 0,
    "Synthesis": 1,
    "Review": 2,
    "Conclusion": 3,
}

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



def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False



def _add_stage(stage_sets: Dict[str, Set[str]], round_num: int, stage_type: str, keu_id: str) -> None:
    if round_num <= 0:
        return
    label = f"R{round_num} {stage_type}"
    stage_sets.setdefault(label, set()).add(keu_id)



def collect_case_stage_sets(
    keus: Dict[str, Dict[str, Any]],
    stage_presence: Dict[int, Set[str]],
    final_round: int,
) -> Dict[str, Set[str]]:
    stage_sets: Dict[str, Set[str]] = {}
    if not isinstance(keus, dict) or not stage_presence:
        return stage_sets

    available_rounds = sorted(stage_presence.keys())
    if not available_rounds:
        return stage_sets

    min_round = available_rounds[0]
    max_round = max(final_round, available_rounds[-1])

    for keu_id, keu_data in keus.items():
        if not isinstance(keu_data, dict) or not keu_data.get('is_key', False):
            continue

        local_id = str(keu_id)

        introduce_round = _safe_int(keu_data.get('round_introduced'))
        if introduce_round <= 0:
            introduce_round = min_round
        introduce_round = max(min_round, min(max_round, introduce_round))

        propose_round = _find_stage_round(stage_presence, introduce_round, 'Propose')
        if propose_round is None:
            propose_round = _find_stage_round(stage_presence, min_round, 'Propose')
        if propose_round is not None:
            _add_stage(stage_sets, propose_round, 'Propose', local_id)

        synthesis_info = keu_data.get('present_in_synthesis') or {}
        for round_key, present in synthesis_info.items():
            if not _is_truthy(present):
                continue
            round_hint = _safe_int(round_key)
            if round_hint <= 0:
                round_hint = introduce_round
            round_hint = max(min_round, min(max_round, round_hint))
            synthesis_round = _find_stage_round(stage_presence, round_hint, 'Synthesis')
            if synthesis_round is not None:
                _add_stage(stage_sets, synthesis_round, 'Synthesis', local_id)

        for citation in keu_data.get('cited_by') or []:
            if not isinstance(citation, dict):
                continue
            action = (citation.get('action') or '').lower()
            stage_name = CITATION_ACTION_STAGE_MAP.get(action)
            if not stage_name:
                continue
            round_hint = _safe_int(citation.get('round'))
            if round_hint <= 0:
                round_hint = introduce_round
            round_hint = max(min_round, min(max_round, round_hint))
            stage_round = _find_stage_round(stage_presence, round_hint, stage_name)
            if stage_round is not None:
                _add_stage(stage_sets, stage_round, stage_name, local_id)

        if _is_truthy(keu_data.get('present_in_final_decision')):
            target_round = max(final_round, available_rounds[-1])
            conclusion_round = _find_stage_round(stage_presence, target_round, 'Conclusion')
            if conclusion_round is None:
                conclusion_round = _find_stage_round(stage_presence, available_rounds[-1], 'Conclusion')
            if conclusion_round is not None:
                _add_stage(stage_sets, conclusion_round, 'Conclusion', local_id)

    return stage_sets


def aggregate_stage_counts_by_final_round(
    observation_roots: List[str], max_round: int = 3
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
                keus = audit_trail.get('keus', {}) or {}
                if not keus:
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

                key_keus = {
                    k: v
                    for k, v in keus.items()
                    if isinstance(v, dict) and v.get('is_key')
                }
                total_key_count = len(key_keus)
                if total_key_count == 0:
                    continue

                stage_sets = collect_case_stage_sets(
                    keus,
                    effective_stage_presence,
                    final_round,
                )

                # Build per-case denominators by stage label
                # - Propose (Rr): denominator = KEUs introduced at round r (size of Rr Propose set)
                # - Synthesis/Review (Rr): denominator = KEUs introduced up to and including round r (cumulative proposed)
                # - Conclusion (Rr): denominator = KEUs introduced up to and including final_round

                # Collect proposed KEUs per round and cumulative
                proposed_by_round: Dict[int, Set[str]] = {}
                for label_key, keu_ids in stage_sets.items():
                    if not isinstance(label_key, str) or not label_key.startswith('R'):
                        continue
                    parts = label_key.split(' ', 1)
                    if len(parts) != 2:
                        continue
                    try:
                        r_val = int(parts[0][1:])
                    except Exception:
                        continue
                    stage_val = parts[1]
                    if stage_val == 'Propose':
                        proposed_by_round.setdefault(r_val, set()).update(keu_ids)

                cumulative_proposed_by_round: Dict[int, Set[str]] = {}
                seen: Set[str] = set()
                for r_val in range(1, final_round + 1):
                    seen.update(proposed_by_round.get(r_val, set()))
                    cumulative_proposed_by_round[r_val] = set(seen)

                for round_num in range(1, final_round + 1):
                    for stage_name in STAGE_SEQUENCE:
                        if not _case_has_stage(effective_stage_presence, round_num, stage_name):
                            continue
                        label = f"R{round_num} {stage_name}"
                        # Numerator: KEUs detected for this stage in this round
                        numerator_count = len(stage_sets.get(label, set()))
                        # Denominator per-stage
                        if stage_name == 'Propose':
                            # Propose at Rr uses cumulative KEUs introduced up to and including Rr
                            denom_count = len(cumulative_proposed_by_round.get(round_num, set()))
                        elif stage_name in {'Synthesis', 'Review'}:
                            denom_count = len(cumulative_proposed_by_round.get(round_num, set()))
                        else:  # Conclusion
                            denom_count = len(cumulative_proposed_by_round.get(final_round, set()))

                        # Fallback: if denominator ends up zero but stage exists, avoid division by zero later
                        # by using numerator_count (so 100% if any mapped), else 0.
                        if denom_count == 0:
                            denom_count = numerator_count

                        stats = stage_totals.setdefault(label, {'numerator': 0, 'denominator': 0})
                        stats['numerator'] += numerator_count
                        stats['denominator'] += denom_count

    return stage_totals



def plot_stage_counts(
    stage_stats: Dict[str, Dict[str, int]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    根据每个阶段的统计数据绘制柱状图，并为不同的轮次（如 R1, R2）使用不同的颜色。

    Args:
        stage_stats: 包含阶段统计数据的字典。
        title: 图表标题。
        save_path: 保存图表的路径。
        show: 是否显示图表。
    """
    if not stage_stats:
        print("No stage data to plot; skipping figure.")
        return

    entries = [
        (stage, stats)
        for stage, stats in stage_stats.items()
        if stats.get('denominator')
    ]
    if not entries:
        print("No stages with valid denominators; skipping figure.")
        return

    entries.sort(key=lambda item: _stage_sort_key(item[0]))
    stages = [stage for stage, _ in entries]
    percentages = [
        (stats['numerator'] / stats['denominator']) * 100
        for _, stats in entries
    ]

    # --- 代码修改开始 ---

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

    # --- 代码修改结束 ---

    ax.set_xlabel('Stage', fontsize=32)
    ax.set_ylabel('Percentage (%)', fontsize=32)

    offset = max(percentages) * 0.03 if percentages else 0.5
    for bar, pct in zip(bars, percentages):
        text_y = bar.get_height() + 0.1
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{pct:.1f}",
            ha='center',
            va='bottom',
            fontsize=20,
        )

    ax.set_ylim(0, max(105, max(percentages) + offset * 4))
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
        description="Aggregate key KEU counts per stage across multiple observation sources."
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
        default=os.path.join("figures", "mechanism1", "combined"),
        help="Directory to save the generated bar charts (one per final round).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figures; only save them.",
    )
    args = parser.parse_args()

    observation_roots = args.observation_roots or default_roots
    stage_stats = aggregate_stage_counts_by_final_round(
        observation_roots, max_round=3
    )

    print("Aggregating stage data from roots:")
    for root in observation_roots:
        print(f"  - {root}")

    # stages = [stage for stage, stats in sorted(stage_stats.items(), key=lambda item: _stage_sort_key(item[0])) if stats.get('denominator')]
    # if not stages:
    #     print("No stages with key KEU data; nothing to report.")
    #     raise SystemExit(0)

    # print("Stage KEU retention percentages:")
    # report_lines = ["Stage KEU retention percentages:"]
    # for stage in stages:
    #     stats = stage_stats[stage]
    #     denom = stats.get('denominator', 0)
    #     if not denom:
    #         continue
    #     pct = stats['numerator'] / denom * 100
    #     line = f"  {stage}: {pct:.2f}% ({stats['numerator']}/{denom})"
    #     print(line)
    #     report_lines.append(line)

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "stage_keu_retention.pdf")
    else:
        save_path = None

    # Save text report alongside the figure
    # report_path = os.path.join(save_dir, "mechanism1_report.txt") if save_dir else None
    # if report_path:
    #     with open(report_path, 'w', encoding='utf-8') as f:
    #         f.write("\n".join(report_lines) + "\n")
    #     print(f"Saved text report to: {report_path}")

    plot_stage_counts(
        stage_stats,
        save_path=save_path,
        show=False,
    )
