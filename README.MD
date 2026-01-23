# Multi-LLM Collaborative Debate System

A sophisticated debate system where multiple LLMs collaborate to solve challenging problems through independent solution generation, peer review, and refinement.

## Overview

This system implements a multi-agent debate framework where:
- **3 Solvers** independently solve problems
- Each solver **critiques** the other solutions through structured peer review
- Solvers **refine** their solutions based on feedback
- **1 Judge** evaluates all refined solutions and selects the best final answer

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd LLM_final

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Set your API keys (required)
export GROQ_API_KEY=your_groq_api_key_here
export MISTRAL_API_KEY=your_mistral_api_key_here

# For Windows PowerShell:
$env:GROQ_API_KEY="your_groq_api_key_here"
$env:MISTRAL_API_KEY="your_mistral_api_key_here"
```

### 3. Prepare Dataset

```bash
# The system includes a default dataset of 25 mathematical problems
# You can view/edit it at: data/problems.json
```

### 4. Run Basic Experiment

```bash
# Test with one problem
python scripts/run_experiments.py --problem-id MATH_01

# Test with multiple problems
python scripts/run_experiments.py --problem-list MATH_01,MATH_02,MATH_03

# Run all problems
python scripts/run_experiments.py --all
```

## Output Structure

After running experiments, results are saved to:
```
data/results/
├── debate_MATH_01_20250123_225315.json  # Individual debate results
├── baselines_20250123_225323.json       # Baseline experiment results
├── detailed_evaluation_report.json      # Detailed metrics
└── final_report_20250123_225324.json   # Final summary report
```

## Configuration

Modify `config.yaml` to adjust model settings:

```yaml
models:
  groq_llama_instant:
    provider: "groq"
    model_name: "llama-3.1-8b-instant"
    max_tokens: 2000
    temperature: 0.7
  
  mistral_7b:
    provider: "mistral"
    model_name: "open-mistral-7b"
    max_tokens: 2000
    temperature: 0.7
```

## Key Commands

### Running Experiments
```bash
# Single problem
python scripts/run_experiments.py --problem-id MATH_01

# Multiple problems
python scripts/run_experiments.py --problem-list MATH_01,MATH_02,MATH_03

# All problems
python scripts/run_experiments.py --all

# With debug output
python scripts/run_experiments.py --problem-id MATH_01 --debug
```

### Diagnosis
```bash
# Analyze why a debate failed
python scripts/diagnose_debate.py

# Test answer extraction
python tests/test_answer_extraction.py
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_integration.py -v

# Run quick validation
python -c "import src; src.validate_environment()"
```

## Project Structure

```
LLM_final/
├── src/                          # Main source code
│   ├── __init__.py              # Package exports
│   ├── agent_prompts.py         # All prompt templates
│   ├── debate_orchestrator.py   # Main orchestrator
│   ├── dataset_manager.py       # Problem dataset
│   ├── evaluation_metrics.py    # Performance metrics
│   ├── model_clients.py         # LLM API clients
│   ├── parsing_utils.py         # LLM output parsing
│   ├── role_assigner.py         # Role assignment
│   └── utils.py                 # Utilities
├── scripts/                     # Execution scripts
│   ├── run_experiments.py      # Main experiment runner
│   └── diagnose_debate.py      # Debugging tool
├── tests/                       # Test suite
├── data/                        # Data directory
│   ├── problems.json           # Problem dataset
│   └── results/                # Experiment results
├── config.yaml                  # Configuration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Example Workflow

### 1. Validate Setup
```bash
python -c "
import src
print('System version:', src.__version__)
print('Environment valid:', src.validate_environment())
"
```

### 2. Run Quick Test
```bash
# Test a simple problem
python -c "
import asyncio
from src import quick_start
result = quick_start('What is 15 × 15?')
print(f'Answer: {result.final_answer}, Correct: {result.is_correct}')
"
```

### 3. Analyze Results
```bash
# Generate metrics report
python -c "
from src import MetricsCalculator
calculator = MetricsCalculator('data/results')
calculator.load_results()
metrics = calculator.calculate_system_metrics()
print(f'Accuracy: {metrics.overall_accuracy:.2%}')
print(f'Consensus Rate: {metrics.consensus_rate:.2%}')
"
```

## Troubleshooting

### Common Issues:

1. **API Key Errors**
   ```
   Error: GROQ_API_KEY not set in environment
   ```
   **Fix:** Set environment variables correctly

2. **Rate Limiting**
   ```
   Rate limit exceeded
   ```
   **Fix:** Reduce request frequency in config.yaml

3. **No Results Found**
   ```
   No debate results found!
   ```
   **Fix:** Run experiments first with `python scripts/run_experiments.py`

### Debug Mode:
```bash
python scripts/run_experiments.py --problem-id MATH_01 --debug
```

## Metrics Calculated

The system calculates:
- **Overall Accuracy**: % of problems solved correctly
- **Improvement Rate**: % of problems where refinement improved answers
- **Consensus Rate**: % of problems where all solvers initially agreed
- **Judge Accuracy**: When solvers disagree, does judge pick the correct one?

## Requirements

- Python 3.8+
- API keys for Groq and Mistral
- Internet connection for API calls

## License

This project is for educational purposes as part of a final project assignment.

---

**Ready to start?** Run:
```bash
python scripts/run_experiments.py --problem-id MATH_01
```