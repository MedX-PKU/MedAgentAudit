# Failure Mode–Correctness Association Analysis

This directory contains reproducible scripts for the preliminary case-level association analysis.

## Scope

- Uses frozen existing MAS and automated-auditor outputs.
- Does not rerun any MAS, LLM, or auditor.
- Builds one row per `dataset × qid × MAS × underlying LLM` case.
- Keeps `positive`, `negative`, `unknown`, `not_applicable`, and `not_audited` separate.
- Fits one model per failure mode.
- Excludes the original near-constant F-2.2.1 label from adjusted models until the revised intermediate-answer-correctness label is available.

## Scripts

- `failure_mode_schema.py`: failure-mode keys, valid steps, and conceptual MAS applicability.
- `build_failure_correctness_case_manifest.py`: validates the frozen logs and writes the case manifest and flow summary.
- `run_preliminary_failure_correctness_models.py`: writes descriptive results, a preliminary random-intercept Bayesian GLMM approximation, question-clustered GEE sensitivity results using an independence working correlation and robust standard errors, and a compact report. Use `--reuse-existing-glmm` to rerun summaries and GEE without refitting completed GLMMs.
- `summarize_preliminary_failure_correctness_results.py`: writes data-quality checks, complete stratified descriptive results, and a Chinese interpretation report.
- `run_all_preliminary_failure_correctness_analysis.sh`: runs the complete preliminary workflow.

## Output Directory

All generated results are written to:

`Preprint/analysis/failure_mode_correctness_association/`

The GLMM uses `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` with a question random intercept and variational Bayes. This is a preliminary approximation for inspecting direction and magnitude. Manuscript-ready estimates must be confirmed using the frequentist GLMM specified in the revision protocol.
