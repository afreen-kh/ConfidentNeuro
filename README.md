# ConfidentNeuro - Medical LLM Benchmark & Conformal Prediction
Conformal prediction-based hallucination risk scoring for LLM-assisted Alzheimer's clinical decision support

This project evaluates the performance, robustness, and confidence calibration of Large Language Models (LLMs) on medical benchmarks (specifically targeting Alzheimer's Disease contexts). It generates a dataset from PubMed abstracts, evaluates models using BERTScore, runs conformal prediction to estimate confidence boundaries, audits models via perturbations, and performs ablation analyses.

## Scripts Overview

### 1. `generate_benchmark.py`
Retrieves abstracts from PubMed related to Alzheimer's Disease (Diagnosis, Staging, Treatment, Biomarkers, Prognosis) and uses an LLM-assisted extraction pipeline to generate a set of specific clinical decision support questions and ground truth answers.
- **Outputs**: `benchmark_150.json`, `benchmark_150.csv`

### 2. `evaluate_models.py`
Evaluates top-tier LLMs (GPT-4o, Claude Haiku, and open-weight models via Hugging Face) against the generated benchmark. 
- Calculates generation time, grabs model log-probabilities (for GPT-4o), and computes semantic accuracy using **BERTScore**.
- **Outputs**: `evaluation_results.csv`

### 3. `conformal_prediction.py`
Applies conformal prediction over the `evaluation_results.csv` data to interpret model reliability.
- Computes nonconformity scores ($1 - BERTScore F1$) and measures regression/Expected Calibration Error (ECE) for GPT-4o.
- Performs reliability/calibration mapping into High, Medium, and Low risk tiers.
- **Outputs**: `conformal_results.csv`, `coverage_table.csv`, `ece_results.csv`, plotting artifacts.

### 4. `perturbation_audit.py`
Tests the robustness of GPT-4o on synthetic clinical notes. Introduces 8 varied perturbation types (e.g. synonym replacement, value swaps, negations) to notes and has the model generate new summaries.
- Evaluates **semantic disruption** of the summary versus the original baseline, tracking whether model confidence properly aligns with disruption.
- **Outputs**: `clinical_notes.json`, `perturbation_results_v2.csv`, `perturbation_analysis_v2.csv`, perturbation plots.

### 5. `ablation_analysis.py`
Runs systematic ablation runs to observe the sensitivity of the conformal prediction framework.
- Checks varying calibration set sizes.
- Tests multiple nonconformity scoring functions ($1 - \text{F1}$, $1 - \text{Precision}$, $1 - \text{Logprob}$).
- Tests sensitivity to varying significance thresholds ($\alpha$).
- **Outputs**: Several CSV tables and visualization plots (e.g., `ablation_score_functions.csv`, `.png`).

## Setup & Dependencies

1. **Clone the repository.**
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Setup:**
   Copy the provided `.env.example` file to a new file named `.env` and fill it with your API keys:
   ```bash
   cp .env.example .env
   ```
   *Required Keys:*
   - `OPENAI_API_KEY`: Required for GPT-4o evaluation and summarization auditing.
   - `ANTHROPIC_API_KEY`: Required for Claude model evaluation.
   - `GEMINI_API_KEY`: Required for benchmark question generation.
   - `NCBI_API_KEY`: Required for fetching PubMed abstracts via Entrez.
   - `HF_TOKEN`: Required for HuggingFace inference (BioMistral, Mistral, LLaMa, etc).
   - ENTREZ_EMAIL: Required by NCBI Entrez API (any valid email address).

## How to Run
Run the scripts sequentially to reproduce the pipeline from end to end:

```bash
# 1. Generate the Q&A dataset
python generate_benchmark.py

# 2. Evaluate LLMs on the generated dataset
python evaluate_models.py

# 3. Assess model confidence and calibration
python conformal_prediction.py

# 4. Run the robustness and perturbation test
python perturbation_audit.py

# 5. Run ablation sensitivity checks
python ablation_analysis.py
```
