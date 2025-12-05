# LLM Dementia Identification

This repository contains scripts for using Large Language Models (LLMs) to classify dementia status from patient survey data. The project tests multiple LLM models, variable sets, and prompts to evaluate their performance on dementia classification tasks using data from the Health and Retirement Study (HRS).

## Overview

The project evaluates how well various LLM models can classify dementia status based on patient survey data including:
- Sociodemographic variables (age, gender, race, education)
- Cognitive health measures (recall scores, serial 7s test, orientation questions)
- Physical health measures (ADL, IADL, self-reported health)
- Proxy-reported measures (memory, wandering, hallucinations)

The scripts test different variable sets and prompts across multiple LLM models to assess classification performance.

## Prerequisites

- Python 3.7 or higher
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))
- Input CSV file: `bm227_final_data_1000_sampled.csv` with patient data

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd be_m227_llm_dementia_identification
```

2. Install required Python packages:
```bash
pip install pandas numpy requests scikit-learn matplotlib
```

Or create a `requirements.txt` file and install:
```bash
pip install -r requirements.txt
```

## Configuration

### API Keys

**For `openrouter_prompt_runner.py`:**
- Edit line 17 in the script and replace `"YOUR_OPEN_ROUTER_API_KEY_HERE"` with your actual API key
- **Warning**: This script has a hardcoded API key. Be careful not to commit this file with your key exposed.

**For `openrouter_prompt_runner_reasoning.py`:**
- Set the environment variable `OPENROUTER_API_KEY`:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### Input Data

Ensure you have the input CSV file `bm227_final_data_1000_sampled.csv` in the repository root directory. This file should contain patient data with columns matching the variable sets defined in the scripts.

### Model Configuration

Both runner scripts test the following LLM models:
- `openai/gpt-5.1`
- `anthropic/claude-sonnet-4.5`
- `google/gemini-2.5-pro`
- `openai/gpt-5-mini`
- `anthropic/claude-haiku-4.5`
- `google/gemini-2.5-flash`

You can modify the `MODELS` list in each script to test different models.

### Variable Sets

The scripts test three predefined variable sets:

1. **`langa_weir_vars_only`**: Minimal cognitive variables
   - Immediate recall, delayed recall, backwards count, serial 7s, proxy memory, interviewer proxy cognitive rating, IADL

2. **`expert_model`**: Expert-selected variables
   - Includes demographics, cognitive tests, proxy reports, health measures, and activities

3. **`everything`**: All available variables
   - Comprehensive set including all variables from the expert model plus additional health and lifestyle variables

## Usage

### 1. Create Balanced Test Patient Set

Before running the reasoning script, you may want to create a balanced test set of patient IDs:

```bash
python create_test_patient_set.py \
    --csv-path bm227_final_data_1000_sampled.csv \
    --dementia-column expert_dem \
    --n-patients 100 \
    --random-seed 42 \
    --output test_patient_ids.json
```

**Arguments:**
- `--csv-path`: Path to input CSV file (default: `bm227_final_data.csv`)
- `--dementia-column`: Column name for dementia status (default: `expert_dem`)
- `--n-patients`: Total number of patients to select (must be even, default: 10)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--output`: Output JSON file path (default: `test_patient_ids.json`)
- `--no-demographic-balance`: Disable demographic balancing (default: balances by gender and age)

**Output:**
Creates a JSON file with:
- Patient IDs (50% with dementia, 50% without)
- Metadata including demographic distributions
- Separate lists for each group

**Example:**
```bash
# Create a balanced set of 100 patients
python create_test_patient_set.py --n-patients 100 --output test_patient_ids_100.json

# Create without demographic balancing
python create_test_patient_set.py --n-patients 50 --no-demographic-balance
```

### 2. Run Main LLM Prompt Runner

This script runs LLM prompts on patient data and saves results:

```bash
python openrouter_prompt_runner.py
```

**Configuration (edit in script):**
- `CSV_PATH`: Input CSV file path (default: `"bm227_final_data_1000_sampled.csv"`)
- `API_KEY`: Your OpenRouter API key (line 17)
- `PATIENT_LIMIT`: Number of patients to process (default: `1`, set to `None` for all)
- `ACTIVE_VARIABLE_SETS`: Which variable sets to test (default: `["langa_weir_vars_only", "expert_model", "everything"]`)
- `RUNS_PER_PROMPT`: Number of runs per prompt (default: `1`)
- `PROMPTS`: List of prompts to test (default: single dementia classification prompt)

**Output:**
- Creates `llm_results/` directory if it doesn't exist
- Saves results to `llm_results/results_<timestamp>.csv`

**Features:**
- Parallel processing (up to 50 concurrent requests)
- Rate limiting with automatic retry on 429 errors
- Progress tracking with ETA
- Combines results from multiple variable sets into single CSV

### 3. Run Reasoning-Enabled LLM Prompt Runner

This script is similar to the main runner but captures reasoning traces from models:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
python openrouter_prompt_runner_reasoning.py
```

**Configuration (edit in script):**
- `CSV_PATH`: Input CSV file path (default: `"bm227_final_data_1000_sampled.csv"`)
- `PATIENT_ID_JSON`: JSON file with patient IDs to filter (default: `"test_patient_ids_100_from_1000.json"`)
- `PATIENT_LIMIT`: Number of patients to process (default: `None` for all in JSON file)
- `ACTIVE_VARIABLE_SETS`: Which variable sets to test
- `RUNS_PER_PROMPT`: Number of runs per prompt (default: `2`)
- `PROMPTS`: List of prompts to test

**Differences from main runner:**
- Uses environment variable for API key (more secure)
- Filters patients using IDs from JSON file
- Enables reasoning output from models
- Saves reasoning traces in separate column
- Creates separate rows for each variable set (one row per variable_set per patient)
- Outputs to `llm_reasoning_results/results_<timestamp>.csv`

**Output:**
- Creates `llm_reasoning_results/` directory
- CSV includes `reasoning_trace` column with model reasoning
- Includes `ablation_type` and `ablated_variables` columns

## Output Format

### Main Runner Output (`llm_results/results_<timestamp>.csv`)

Columns:
- `patient_id`: Patient identifier
- `csv_row`: Row number from input CSV
- `prompt_id`: Index of the prompt used
- `run`: Run number (1 to RUNS_PER_PROMPT)
- `model`: LLM model name
- `prompt_langa_weir`, `prompt_expert`, `prompt_everything`: Full prompts for each variable set
- `response_langa_weir`, `response_expert`, `response_everything`: LLM responses for each variable set
- `lasso_dem`, `expert_dem`, `hurd_dem`, `langa_weir_2cat`, `cog_impair_2cat`: Ground truth labels

### Reasoning Runner Output (`llm_reasoning_results/results_<timestamp>.csv`)

Columns:
- `patient_id`, `csv_row`, `prompt_id`, `run`: Same as main runner
- `variable_set`: Which variable set was used (one row per variable set)
- `model`: LLM model name
- `prompt`: Full prompt text
- `response`: LLM response content
- `reasoning_trace`: Model's reasoning process (if available)
- `ablation_type`: Type of variable set used
- `ablated_variables`: Variables included in this set
- Ground truth columns: Same as main runner

## Analysis

### Using helper.py Functions

The `helper.py` module provides utility functions for analyzing results:

**`process_response(df, col_name)`:**
- Parses LLM responses to extract binary labels and probability scores
- Handles various response formats (comma-separated, newline-separated)
- Returns DataFrame with `{col_name}_label` and `{col_name}_score` columns

**`sensitivity_specificity(y_true, y_pred)`:**
- Calculates classification metrics: sensitivity, specificity, precision, accuracy, F1
- Handles multiple variable sets (lasso, expert, everything)
- Returns dictionary with metrics for each variable set

### Using analysis.ipynb

The Jupyter notebook `analysis.ipynb` demonstrates how to:
1. Load result CSV files
2. Process LLM responses using `process_response()`
3. Calculate performance metrics using `sensitivity_specificity()`
4. Compare models and variable sets

**Example workflow:**
```python
import pandas as pd
from helper import process_response, sensitivity_specificity

# Load results
df = pd.read_csv('llm_results/results_1234567890.csv')

# Process responses
df_processed = process_response(df, 'response_expert')

# Calculate metrics
metrics = sensitivity_specificity(df['expert_dem'], df_processed)
```

## File Structure

```
be_m227_llm_dementia_identification/
├── README.md                          # This file
├── openrouter_prompt_runner.py        # Main LLM runner script
├── openrouter_prompt_runner_reasoning.py  # Reasoning-enabled runner
├── create_test_patient_set.py         # Test set creation utility
├── helper.py                          # Analysis utility functions
├── analysis.ipynb                     # Jupyter notebook for analysis
├── test_patient_ids_100_from_1000.json  # Pre-generated test set
├── bm227_final_data_1000_sampled.csv  # Input patient data (required)
├── llm_results/                       # Output directory (created by main runner)
│   └── results_<timestamp>.csv
└── llm_reasoning_results/             # Output directory (created by reasoning runner)
    └── results_<timestamp>.csv
```

## Common Workflows

### Workflow 1: Quick Test Run
1. Set API key in `openrouter_prompt_runner.py` (line 17)
2. Ensure `PATIENT_LIMIT = 1` for testing
3. Run: `python openrouter_prompt_runner.py`
4. Check `llm_results/results_<timestamp>.csv`

### Workflow 2: Full Evaluation with Reasoning
1. Create balanced test set:
   ```bash
   python create_test_patient_set.py --n-patients 100 --output test_patient_ids.json
   ```
2. Set environment variable:
   ```bash
   export OPENROUTER_API_KEY="your-key"
   ```
3. Update `PATIENT_ID_JSON` in `openrouter_prompt_runner_reasoning.py` if needed
4. Run: `python openrouter_prompt_runner_reasoning.py`
5. Analyze results in `analysis.ipynb`

### Workflow 3: Custom Variable Set Testing
1. Edit `VARIABLE_SETS` dictionary in runner script to add custom sets
2. Add set name to `ACTIVE_VARIABLE_SETS`
3. Update `response_column_lookup` and `prompt_column_lookup` if needed
4. Run script and analyze results

## Notes

- **Rate Limiting**: The main runner handles up to 50 parallel requests with automatic retry on rate limits. The reasoning runner uses up to 10 workers.
- **Cost**: Running these scripts will incur costs through OpenRouter API. Monitor your usage.
- **Data Privacy**: Ensure patient data is handled according to your institution's data use agreements.
- **Reproducibility**: Use fixed random seeds when creating test sets for reproducible results.

## Troubleshooting

**Error: "OPENROUTER_API_KEY is not set"**
- For reasoning runner: Set the environment variable before running
- For main runner: Update the API_KEY variable in the script

**Error: "FileNotFoundError: bm227_final_data_1000_sampled.csv"**
- Ensure the CSV file exists in the repository root directory
- Or update `CSV_PATH` in the script to point to the correct location

**Error: "429 (rate limit)"**
- The script will automatically retry after 30 seconds
- Consider reducing `max_parallel_requests` in the rate_limit_state dictionary

**No reasoning traces in output**
- Not all models support reasoning. Check OpenRouter documentation for model capabilities.
- Ensure `reasoning.enabled: True` is set in the API payload (already configured in script)

## License

[Add your license information here]

## Contact

[Add contact information here]

