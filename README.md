# MedAgentAudit

MedAgentAudit is a lightweight research codebase for running and auditing multi-agent medical QA pipelines. It provides several agentic frameworks (single-model baselines and multi-agent coordinators), unified configuration for multiple LLM providers, and structured logging to JSONL for later quantitative analysis.

## Key Features

- **Multiple frameworks**: run different coordination strategies (e.g., `medagent`, `mac`, `mdagents`, `reconcile`, `colacare`, `healthcareagent`).
- **Auditing vs. open-coding modes**: frameworks can be executed in either *audit* mode or *open_coding* mode (where supported).
- **Structured outputs and logs**: results/errors are appended as **JSONL** (one record per line), which is friendly to streaming runs and post-hoc analytics.
- **Provider-agnostic LLM config**: models are defined in `config.toml` and loaded at runtime.
- **Vision support**: optional image inputs via `encode_image` for datasets that include `image_path`.

## Repository Layout

- `medagentaudit/framework/`: runnable frameworks and CLI entrypoints
  - `single_llm.py`: single-model baseline runner
  - `medagent.py`: MedAgent-style multi-expert workflow
  - `mac.py`: multi-agent collaboration runner
  - `mdagents.py`: multi-doctor style coordinator
  - `reconcile.py`: Reconcile-style discussion coordinator
  - `colacare.py`, `healthcareagent.py`: additional multi-agent pipelines
- `medagentaudit/core/`: base abstractions (e.g., `BaseAgent`)
- `medagentaudit/auditor/`: auditor agent implementation
- `medagentaudit/utils/`: shared utilities (JSON/JSONL IO, logging, config loading, image encoding)
- `datasets/`: input datasets (typically prepared under `datasets/processed/...`)
- `logs/`: run outputs (JSONL results, JSONL errors, and terminal logs)

## Installation

This project targets **Python 3.12+**.

If you use `uv` (recommended in this repo), install dependencies from `pyproject.toml`:

```bash
uv sync
```

Or install with `pip`:

```bash
pip install -e .
```

## Configuration

Model endpoints are configured in `config.toml`. API keys are expected via environment variables.

Example (shell):

```bash
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export GEMINI_API_KEY=...
export GLM_API_KEY=...
export DASHSCOPE_API_KEY=...
```

Notes:

- The mapping from `model_key` (CLI argument) to provider settings is defined in `config.toml` under `[llm."..."]`.
- Some models support additional options like `reasoning_effort` or `enable_thinking`.

## Data Format

Runners expect dataset items shaped roughly like:

```json
{
  "qid": "...",
  "question": "...",
  "options": {"A": "...", "B": "..."},
  "answer": "A",
  "image_path": "path/to/image.jpg"
}
```

Depending on the dataset, `options` and/or `image_path` can be omitted.

## Running Frameworks

All frameworks under `medagentaudit/framework/` are intended to be runnable as scripts.

### Reconcile

```bash
python3 medagentaudit/framework/reconcile.py \
  --dataset MedQA \
  --agents gpt-5.2 deepseek-reasoner \
  --auditor_model gpt-5.2 \
  --num_samples 100 \
  --max_rounds 3 \
  --time_stamp 20260202 \
  --task audit
```

`--task` supports:

- `audit` (default)
- `open_coding`

### Other frameworks

Other runners follow similar conventions: a dataset selection, one or more model keys, a sample limit, and optional auditing configuration.

Browse the corresponding `main()` in:

- `medagentaudit/framework/medagent.py`
- `medagentaudit/framework/mac.py`
- `medagentaudit/framework/mdagents.py`
- `medagentaudit/framework/colacare.py`
- `medagentaudit/framework/healthcareagent.py`
- `medagentaudit/framework/single_llm.py`

## Output & Logging

Runs write structured artifacts under `logs/`.

Common patterns:

- **Results**: JSONL with one record per processed item (`qid`, model metadata, predicted answer, and per-case history).
- **Errors**: JSONL with exception type/message and full traceback for failed items.
- **Terminal logs**: stdout/stderr are captured for reproducibility (including tqdm progress).

## Development Notes

- Utility code lives in `medagentaudit/utils/` (e.g., `json_utils.py`, `logger.py`, `config_loader.py`).
- Agents extend `medagentaudit/core/base_agent.py` and typically call provider APIs via the OpenAI-compatible client.

## Disclaimer

This repository is for research/engineering purposes only and is **not** medical advice. Always consult qualified clinicians for real-world clinical decisions.

