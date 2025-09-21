<div align="center">

# Multi-level Diagnosis and Evaluation for Robust Tabular Feature Engineering with Large Language Models

**A comprehensive framework for evaluating LLMs' ability to perform feature engineering tasks**

</div>

## 📋 Overview

Recent advancements in large language models (LLMs) have shown promise in feature engineering for tabular data, but concerns about their reliability persist, especially due to variability in generated outputs. We introduce a multi-level diagnosis and evaluation framework to assess the robustness of LLMs in feature engineering across diverse domains, focusing on the three main factors: key variables, relationships, and decision boundary values for predicting target classes. We demonstrate that the robustness of LLMs varies significantly over different datasets, and that high-quality LLM-generated features can improve few-shot prediction performance by up to 10.52%. This work opens a new direction for assessing and enhancing the reliability of LLM-driven feature engineering in various domains.

## 📁 Repository Structure

```
RobustnessEval/
├── main.py                 # Main evaluation pipeline
├── config.py              # Configuration management
├── data_utils.py          # Data processing and utilities
├── llm_utils.py           # LLM client implementations
├── prompt_generator.py    # Prompt generation for different tasks
├── evaluator.py           # Evaluation metrics and scoring
├── data/                  # Dataset storage
│   ├── adult.csv
│   ├── bank.csv
│   ├── blood.csv
│   ├── credit-g.csv
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── cultivar.csv
│   └── myocardial.csv
├── templates/             # Prompt templates
├── diagnose/             # Diagnosis mode outputs
├── evaluate/             # Evaluation mode outputs
└── logs/                 # Experiment logs
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/DohaLim/Robustness-eval
cd RobustnessEval
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Basic Usage

#### Single Experiment
```bash
# Run a single diagnosis experiment
python main.py --data adult --mode diagnose --level lv1 --shot 4 --llm-engine openai --llm-name gpt-3.5-turbo

# Run a single evaluation experiment
python main.py --data adult --mode evaluate --level lv1 --shot 4 --llm-engine together --llm-name meta-llama/Llama-3.1-8B-Instruct
```

#### Batch Experiments
```bash
# Run comprehensive evaluation grid
./run_feature_evaluation.sh "meta-llama/Llama-3.1-8B-Instruct" "adult" "diagnose" "vllm"

# Run single configuration
./run_feature_single.sh "gpt-3.5-turbo" "adult" "diagnose" "lv1" "4" "random" "1" "openai"
```

## 📊 Multi-level Tasks

- **Level 1 (Identifying Key Variables)**: LLMs are tested on their ability to recognize the most important variables for a given task.
- **Level 2 (Understanding Variable-Class Relationships)**: This level evaluates whether LLMs can correctly determine the causal relationship between variables and target classes.
- **Level 3 (Setting Decision Boundaries)**: We assess whether LLMs can provide stable decision boundaries for variables.

### 🔧 Custom Datasets

To add a new dataset:

1. Add CSV file to `data/` directory
2. Create corresponding metadata JSON file
3. Update `TASK_DICT` in `prompt_generator.py`
4. Add dataset configuration in `data_utils.py`

## ✉️ Contact

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: yebinuni@korea.ac.kr

---
