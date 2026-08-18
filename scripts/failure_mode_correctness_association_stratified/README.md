# Failure Mode–Correctness Stratified Analyses

`run_stratified_analysis.py` fits separate GLMMs and question-clustered GEEs for all ten failure modes within:

- QA and VQA modalities;
- datasets;
- MAS methods;
- underlying LLMs.

`build_stratified_analysis.py` applies the requirement of at least 10 failure-positive and 10 failure-negative cases, compares stratum-specific directions with the pooled GLMM, and records GEE fallbacks when a stratum GLMM does not meet the convergence criterion.

Outputs are written to `Preprint/analysis/failure_mode_correctness_association/stratified_analysis`.

Separate stratum P values are descriptive. Cross-stratum differences require interaction tests.

## Run

From the code repository root:

```bash
bash scripts/failure_mode_correctness_association_stratified/run_analysis.sh
```
