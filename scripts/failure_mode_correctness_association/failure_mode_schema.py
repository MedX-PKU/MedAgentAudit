"""Shared schema for the transfer-revision failure–correctness analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureModeDefinition:
    code: str
    short_name: str
    log_key: str
    status_key: str
    allowed_steps: frozenset[str] | None = None
    applicable_mas: frozenset[str] | None = None


FAILURE_MODES: tuple[FailureModeDefinition, ...] = (
    FailureModeDefinition(
        "1.1.1",
        "factual hallucinations",
        "1_1_1_factual_hallucination",
        "factual_hallucination_status",
    ),
    FailureModeDefinition(
        "1.2.1",
        "modality neglect",
        "1_2_1_neglect_or_misinterpretation_of_modality_info",
        "modality_neglect_status",
    ),
    FailureModeDefinition(
        "2.1.1",
        "role-task mismatch",
        "2_1_1_role_assignment",
        "role_task_alignment",
        applicable_mas=frozenset({"mdagents", "medagent"}),
    ),
    FailureModeDefinition(
        "2.1.2",
        "failure to activate specialist knowledge",
        "2_1_2_domain_specific_knowledge_activation",
        "knowledge_activation_status",
    ),
    FailureModeDefinition(
        "2.2.1",
        "repetition of initial views",
        "2_2_1_repetition_of_initial_views",
        "interaction_redundancy",
        applicable_mas=frozenset(
            {"colacare", "healthcareagent", "mac", "medagent", "reconcile"}
        ),
    ),
    FailureModeDefinition(
        "2.2.2",
        "unresolved conflicts",
        "2_2_2_unresolved_conflicts",
        "conflict_resolution_status",
        allowed_steps=frozenset({"analysis", "review"}),
        applicable_mas=frozenset(
            {"colacare", "healthcareagent", "mac", "medagent", "reconcile"}
        ),
    ),
    FailureModeDefinition(
        "3.1.1",
        "minority suppression",
        "3_1_1_suppression_of_minority_views",
        "suppression_status",
        applicable_mas=frozenset(
            {"colacare", "healthcareagent", "mac", "mdagents", "medagent"}
        ),
    ),
    FailureModeDefinition(
        "3.1.2",
        "authority bias",
        "3_1_2_authority_bias",
        "authority_bias_status",
        applicable_mas=frozenset(
            {"colacare", "healthcareagent", "mac", "mdagents", "medagent"}
        ),
    ),
    FailureModeDefinition(
        "3.1.3",
        "contradiction neglect",
        "3_1_3_neglect_of_contradictions",
        "neglect_of_conflict_status",
        applicable_mas=frozenset(
            {"colacare", "healthcareagent", "mac", "mdagents", "medagent"}
        ),
    ),
    FailureModeDefinition(
        "3.2.1",
        "self-contradiction across rounds",
        "3_2_1_self_contradiction_when_decision",
        "inter_round_consistency_status",
        applicable_mas=frozenset({"colacare", "mac", "medagent"}),
    ),
)


FAILURE_MODE_BY_CODE = {item.code: item for item in FAILURE_MODES}


MAS_DISPLAY = {
    "colacare": "ColaCare",
    "healthcareagent": "HealthcareAgent",
    "mac": "MAC",
    "mdagents": "MDAgents",
    "medagent": "MedAgents",
    "reconcile": "ReConcile",
}

DATASET_DISPLAY = {
    "medqa": "MedQA",
    "pubmedqa": "PubMedQA",
    "medxpertqa-text": "MedXpertQA",
    "pathvqa": "PathVQA",
    "vqa-rad": "VQA-RAD",
    "slake": "SLAKE",
}

LLM_DISPLAY = {
    "deepseek-reasoner": "DeepSeek-V3.2-Thinking",
    "gemini-3-flash-preview": "Gemini-3-Flash",
    "gpt-5.2": "GPT-5.2",
    "qwen3-8b": "Qwen-3",
    "glm-4.6v": "GLM-4.6V",
    "qwen3-vl-8b-thinking": "Qwen-3VL",
}
