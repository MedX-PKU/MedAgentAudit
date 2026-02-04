# MedAgentAudit: A Large-Scale Empirical Study & Unified Taxonomy of Medical Multi-Agent Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Dataset](https://img.shields.io/badge/Dataset-MedQA%7CPubMedQA%7CMedXpertQA%7CPathVQA%7CVQA--RAD%7CSlake-green)](./datasets)

## 📖 Abstract

While large language model (LLM)-based multi-agent systems show promise in simulating medical consultations, their evaluation is often confined to final-answer accuracy. This practice treats their internal collaborative processes as opaque "black boxes" and overlooks a critical question: **is a diagnostic conclusion reached through a sound and verifiable reasoning pathway?**

As these systems are increasingly deployed on web-based platforms, their inscrutable nature poses a significant risk of generating and disseminating high-stakes medical misinformation. To address this, we conduct a large-scale empirical study of **3,600 cases** (Open-Coding) and **14,400 cases** (Audit) from six medical datasets and six representative multi-agent frameworks. Through a rigorous, mixed-methods approach combining qualitative analysis with quantitative auditing, we develop a comprehensive taxonomy of collaborative failure modes. Our quantitative audit reveals 10 dominant failure patterns, from the task comprehension stage to the final decision stage.

## 🧬 Unified Taxonomy of Collaborative Failures

We propose a **Unified Taxonomy of Dynamic Processes in Multi-Agent Collaboration for Healthcare**, categorizing failures into three logical phases of the agent workflow.

<details>
<summary><b>Phase I: Task Comprehension Failures (Click to expand)</b></summary>

*Failures occurring before collaboration begins, determining input quality.*

*   **1.1 Perceptual and Knowledge Deficits (Input Processing Failures)**
    *   **1.1.1 Factual Hallucinations during Input Interpretation**: Agents report non-existent lesions, miss obvious abnormalities, or confuse anatomical structures (e.g., swapping left/right) at the perception level.
*   **1.2 Misalignment with Clinical Intent (Instruction-Level Failures)**
    *   **1.2.1 Neglect or Misinterpretation of Modality Information**: Agents ignore image inputs in VQA tasks (treating them as text-only) or fail to answer the specific clinical question (e.g., describing image metadata instead of pathology).

</details>

<details>
<summary><b>Phase II: Collaboration Process Failures (Click to expand)</b></summary>

*Failures occurring during the dynamic interaction and information exchange.*

*   **2.1 Failure in Agent Role Assignment and Execution**
    *   **2.1.1 Mismatch Between Assigned Roles and Clinical Tasks**: Incorrect specialists assigned (e.g., a Dermatologist for a Chest X-ray).
    *   **2.1.2 Failure to Activate Specialist Knowledge**: Agents display "Role Homogeneity" (generic responses regardless of title) or "Restrictive Refusal" (refusing tasks based on rigid role definitions).
*   **2.2 Failures in Agent Discussion**
    *   **2.2.1 Repetition of Initial Views**: Discussion fails to introduce new information; agents simply repeat initial conclusions or blindly agree ("Lazy Agreement").
    *   **2.2.2 Unresolved Conflicts**: Agents ignore mutually exclusive factual claims (e.g., one sees a mass, one does not) and proceed without resolution.

</details>

<details>
<summary><b>Phase III: Decision-Making Failures (Click to expand)</b></summary>

*Failures in the synthesis and final output generation.*

*   **3.1 Failures in Scrutinizing Reasoning Processes**
    *   **3.1.1 Suppression of Correct Minority Views**: The system adopts an incorrect consensus, discarding a correct diagnosis held by a minority.
    *   **3.1.2 Reasoning Distorted by Authority Bias**: Decisions are based on agent "rank" or response length rather than medical evidence.
    *   **3.1.3 Neglect of Contradictions in Reasoning**: The system aggregates answers that have the same final label but contradictory medical justifications (e.g., "Pneumonia on Left" vs "Pneumonia on Right").
*   **3.2 Failures in the Final Aggregation Stage**
    *   **3.2.1 Self-Contradiction Across Rounds**: The synthesizer/decision-maker flips their diagnosis in later rounds without new evidence.

</details>

## 📊 Datasets & Distribution

We evaluate our method on the test sets of 6 distinct medical datasets, covering both Text QA and Visual QA (VQA).

| Dataset        | Modality   | Task Type     | Open-Coding Samples | Audit Samples | Source                                                       |
| :------------- | :--------- | :------------ | :------------------ | :------------ | :----------------------------------------------------------- |
| **MedQA** (US) | Text       | Single Choice | 600                 | 2400          | [Github](https://github.com/jind11/MedQA)                      |
| **PubMedQA**   | Text       | Yes/No/Maybe  | 600                 | 2400          | [Github](https://github.com/pubmedqa/pubmedqa)                 |
| **MedXpertQA** | Text       | Single Choice | 600                 | 2400          | [HuggingFace](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) |
| **PathVQA**    | Image+Text | Yes/No        | 600                 | 2400          | [Github](https://github.com/KaveeshaSilva/PathVQA)             |
| **VQA-RAD**    | Image+Text | Yes/No & Open | 600                 | 2400          | [HuggingFace](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) |
| **Slake**      | Image+Text | Yes/No & Open | 600                 | 2400          | [HuggingFace](https://huggingface.co/datasets/BoKelvin/SLAKE) |
| **Total**      |            |               | **3,600**           | **14,400**    |                                                              |

*   **Open-Coding Set**: Used for qualitative analysis to derive the taxonomy.
    *   *LLM Setup*: QA tasks use `deepseek-reasoner`; VQA tasks use `gemini-3-flash-preview`.
*   **Audit Set**: Used for large-scale quantitative verification (No overlap with Open-Coding set).
    *   *LLM Setup*: 4 different LLMs tested across 6 frameworks.

## 🛠️ Frameworks Audited

The repository includes implementations of the following Multi-Agent Systems (MAS):
1.  **MedAgent**
2.  **ColaCare**
3.  **HealthcareAgent**
4.  **MAC**
5.  **MDAgents**
6.  **ReConcile**

## 📂 Project Structure

```text
├── config.toml                  # Configuration for LLM API keys and endpoints
├── datasets                     # Processed and Raw datasets
├── logs                         # Output logs for audit, open-coding, and metrics
├── medagentaudit
│   ├── auditor                  # The "Auditor" agent logic (performs the taxonomy checks)
│   ├── common                   # Shared Enums (AgentType, MedicalSpecialty)
│   ├── core                     # BaseAgent and LLM calling logic
│   ├── framework                # Implementation of the 6 MAS frameworks (e.g., colacare.py)
│   └── utils                    # Utilities for config, logging, image encoding
├── scripts                      # Execution scripts
│   ├── run_audit.sh             # Script to run the full audit
│   ├── run_open_coding.sh       # Script for open-coding generation
│   └── run_single_llm.sh        # Baseline single-LLM inference
├── pyproject.toml               # Dependency definitions
└── uv.lock                      # Lock file for dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

We use **[uv](https://github.com/astral-sh/uv)** for extremely fast Python package management.

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies from pyproject.toml
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Configuration

Create or modify `config.toml` in the root directory. You need to provide API keys for the models you intend to use (e.g., OpenAI, Google Gemini, DeepSeek).

```toml
# Example config.toml structure
[llm.deepseek-reasoner]
api_key = "YOUR_DEEPSEEK_KEY"
base_url = "https://api.deepseek.com"
model_name = "deepseek-reasoner"

[llm.gemini-3-flash-preview]
api_key = "YOUR_GEMINI_KEY"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model_name = "gemini-3-flash-preview"
# ... other models
```

### 3. Running Experiments

We provide shell scripts to automate the experiments.

#### A. Run Multi-Agent Audit
To run the quantitative audit using the `AuditorAgent` to detect failure modes across the frameworks:

```bash
cd scripts
# Usage: This script iterates through datasets and frameworks defined inside.
# Ensure you check the parameters inside the script before running.
bash run_audit.sh
```
*Logic*: This executes the frameworks located in `medagentaudit/framework/`, engaging the `AuditorAgent` (powered by Gemini-3/Pro) to evaluate every step of the collaboration against our taxonomy.

#### B. Run Open-Coding Generation
To generate the initial set of cases for qualitative analysis (without the automated audit scoring):

```bash
cd scripts
bash run_open_coding.sh
```

#### C. Run Single LLM Baseline
To run simple inference using a single model for baseline comparison:

```bash
cd scripts
bash run_single_llm.sh
```

## 🔍 The Auditor Agent

A core component of this project is the `AuditorAgent` (located in `medagentaudit/auditor/auditor_agent.py`). It is a specialized meta-agent designed to strictly enforce the taxonomy definitions.

*   **Mechanism**: It intervenes at specific hooks (e.g., after an agent speaks, after synthesis) to evaluate the content.
*   **Prompts**: It uses carefully crafted prompts (e.g., `AUDITOR_PROMPTS["Factual_Hallucination_Prompts"]`) to detect specific failures like "1.1.1 Factual Hallucinations" or "3.1.2 Authority Bias".

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.