# Failure Mode–Correctness Association

This directory contains the reproducible analysis of case-level failure-mode labels and final benchmark-answer correctness. It reads frozen MAS and automated-auditor logs; it does not call an MAS, underlying LLM, or auditor.

## Scripts

- `failure_mode_schema.py`: defines the ten failure modes, log fields, display names, and framework applicability.
- `build_failure_correctness_case_manifest.py`: builds and validates one row per dataset–question–MAS–LLM case, final correctness, and the original case-level failure labels.
- `build_revised_repetition_labels.py`: deterministically rebuilds repetition of initial views (F-2.2.1) from auditor repetition status and framework-specific intermediate answers.
- `run_failure_correctness_analysis.py`: merges the revised F-2.2.1 labels and runs descriptive estimates, question-cluster bootstrap intervals, maximum-likelihood random-intercept logistic models, question-clustered GEE sensitivity models, stratified summaries, and prespecified interaction checks.
- `run_all_failure_correctness_analysis.sh`: runs the complete workflow with 2,000 bootstrap replicates and 9-/15-node adaptive Gauss–Hermite quadrature.

## Statistical Model

Each failure mode is fitted separately:

`correctness ~ failure_positive + dataset + MAS + underlying_LLM + (1 | dataset:question_ID)`

The primary mixed-effects logistic regression is estimated by maximum likelihood with adaptive Gauss–Hermite quadrature around each question-specific random-intercept posterior mode. The script reports optimization status, score norm, observed-information diagnostics, random-intercept standard deviation, and the change in the target coefficient when the quadrature node count is increased from 9 to 15. A question-clustered GEE with robust standard errors is the sensitivity model.

The model for modality neglect (F-1.2.1) additionally checks a failure-by-modality interaction and QA/VQA strata. The model for role-task mismatch (F-2.1.1) checks a failure-by-MAS interaction and MDAgents/MedAgents strata.

All estimates are observational associations. They do not establish that a failure label caused the final answer.

## Run

From the repository root:

```bash
bash scripts/failure_mode_correctness_association/run_all_failure_correctness_analysis.sh
```

Results are written to the `Preprint/analysis/failure_mode_correctness_association` directory specified by the project README.
