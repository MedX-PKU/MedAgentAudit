# Failure Mode–Correctness Stratified Analyses

`run_stratified_analysis.py` fits separate GLMMs and question-clustered GEEs for all ten failure modes within:

- QA and VQA modalities;
- datasets;
- MAS methods;
- underlying LLMs.

It also fits `failure-positive status × modality` GLMM and GEE interaction models for all ten failure modes. QA is the reference modality. GLMM and GEE interaction P values are separately corrected across the ten modes using the Benjamini–Hochberg procedure.

`build_stratified_analysis.py` applies the requirement of at least 10 failure-positive and 10 failure-negative cases, compares stratum-specific directions with the pooled GLMM, and records GEE fallbacks when a stratum GLMM does not meet the convergence criterion.

Outputs are written to `Preprint/analysis/failure_mode_correctness_association/stratified_analysis`.

Separate stratum P values do not test whether association estimates differ between strata. The QA/VQA interaction models directly test whether the failure-positive OR differs between QA and VQA.

## Run

From the code repository root:

```bash
bash scripts/failure_mode_correctness_association_stratified/run_analysis.sh
```
