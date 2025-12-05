import concurrent.futures
import csv
import json
import os
import time
from typing import List, Optional

import requests

# Data CSV Input/Output
CSV_PATH = "bm227_final_data_1000_sampled.csv"
RESULTS_DIR = "llm_reasoning_results"
OUTPUT_CSV = os.path.join(RESULTS_DIR, f"results_{int(time.time())}.csv")
# JSON file with pre-selected balanced patient IDs to run reasoning on
PATIENT_ID_JSON = "test_patient_ids_100_from_1000.json"


# OpenRouter API Information
# API key is expected to be provided via environment variable OPENROUTER_API_KEY
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# OpenRouter model IDs
MODELS = [
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-pro",
    "openai/gpt-5-mini",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
]

# Prompt configuration
RUNS_PER_PROMPT = 2
PROMPTS: List[str] = [
    "You are an expert epidemiologist and clinician specializing in cognitive disorders. You will be given survey data for an individual, including sociodemographic variables and physical and cognitive health measures. Some data may be missing; ignore missingness and base your decision only on what is available. Do not assume dementia from missingness alone. Your task is to classify dementia status in two ways (Binary, and as a probability). Output 1: 0 = without dementia, 1 = with dementia. Output 2: A probability ranging from 0 meaning the probability of not having dementia to 1 meaning having dementia. Respond only with output 1, output 2. ",
]

# Patient data configuration
# Use all patients in the ID list by default
PATIENT_LIMIT: Optional[int] = None  # None for all
# Use the same default variable sets as the main prompt runner, but keep one row per variable_set
ACTIVE_VARIABLE_SETS: Optional[List[Optional[str]]] = [
    "langa_weir_vars_only",
    "expert_model",
    "everything",
]  # None here would mean "all columns"
VARIABLE_SETS = {
    "langa_weir_vars_only": [
        "imrc",
        "dlrc",
        "bwc20",
        "ser7",
        "prmem",
        "int_proxy_cog_rating",
        "iadl5h",
    ],
    "expert_model": [
        "age_cont",
        "male",
        "race",
        "hispanic",
        "ed_cont",
        "proxy",
        "imrc",
        "dlrc",
        "mo",
        "dy",
        "yr",
        "dw",
        "ser7",
        "cact",
        "scis",
        "pres",
        "vp",
        "prmem",
        "lost",
        "wander",
        "alone",
        "haluc",
        "iqcode_mean",
        "adl5h",
        "iadl5h",
        "shlt",
        "diabe",
        "mstat",
        "volunteer_children_young",
        "volunteer_other",
    ],
    "everything": [
        "age_cont",
        "male",
        "race",
        "hispanic",
        "ed_cont",
        "proxy",
        "imrc",
        "dlrc",
        "mo",
        "dy",
        "yr",
        "dw",
        "bwc20",
        "ser7",
        "cact",
        "scis",
        "pres",
        "vp",
        "prmem",
        "lost",
        "wander",
        "alone",
        "haluc",
        "iqcode_mean",
        "int_proxy_cog_rating",
        "adl5h",
        "iadl5h",
        "shlt",
        "diabe",
        "hibpe",
        "bmi_cont",
        "smoke_3cat",
        "drink",
        "mstat",
        "volunteer_children_young",
        "volunteer_other",
        "sayret",
    ],
}
VARIABLE_MAPPING = {
    "hhidpn": "HRS person identifier",
    "year": "Calender Year of HRS wave",
    "wave": "HRS wave number",
    "age_cont": "Age (Years)",
    "male": "Whether Male",
    "race": "Race",
    "hispanic": "Whether Hispanic",
    "ed_cont": "Participant Years of Education",
    "proxy": "Whether Proxy Interview",
    "imrc": "Recall Score Immediate",
    "dlrc": "Delayed Recall Score",
    "mo": "Can recall: Month",
    "dy": "Can recall: Day of the month",
    "yr": "Can recall: Year",
    "dw": "Can recall: Day of the week",
    "bwc20": "Backwards count from 20",
    "ser7": "Number of correct subtractions in the serial 7s test",
    "cact": "Can correctly name: Cactus",
    "scis": "Can correctly name: Scissors",
    "pres": "Can correctly name: President",
    "vp": "Can correctly name: Vice President",
    "prmem": "Proxy-rated: Memory",
    "lost": "Ever gets lost in familiar environments",
    "wander": "Ever wanders off and doesn't return on their own",
    "alone": "Can be left alone for an hour or so",
    "haluc": "Ever sees or hears things that aren't really there",
    "iqcode_mean": "Informant Questionnaire on Cognitive Decline (IQCODE) average score",
    "adl5h": "Summary measure: Activities of Daily Living (ADL)",
    "iadl5h": "Summary measure: Instrumental Activities of Daily Living (IADL)",
    "shlt": "Self-report of health",
    "diabe": "Ever had: Diabetes",
    "hibpe": "Ever had: High blood pressure",
    "bmi_cont": "Body Mass Index (kg/m^2)",
    "smoke_3cat": "Smoking Status",
    "drink": "Ever drinks any alcohol",
    "mstat": "Merital status",
    "sayret": "Retirement status",
    "volunteer_children_young": "Volunteer with Children/Young People",
    "volunteer_other": "Other Volunteer/Charity Work",
    "cog_impair_2cat": "Herzog-Wallace Cognitive Impairement Classification",
    "langa_weir_2cat": "Lange and Weir Cognition Classification",
    "hurd_dem": "Hurd model dementia classification using race/ethnicity-specific cutoffs",
    "expert_dem": "Expert model dementia classification using race/ethnicity-specific cutoffs",
    "lasso_dem": "LASSO model dementia classification using race/ethnicity-specific cutoffs",
    "int_proxy_cog_rating": "Interviewer proxy cognitive rating",
}


def load_patient_data(
    csv_path: str,
    variable_set: Optional[str],
    diag_columns: List[str],
    max_patients: Optional[int],
    allowed_patient_ids: Optional[set] = None,
) -> List[dict]:
    if variable_set and variable_set not in VARIABLE_SETS:
        raise ValueError(f"Unknown variable set: {variable_set}")

    patients: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        columns = reader.fieldnames
        if variable_set:
            columns = [col for col in VARIABLE_SETS[variable_set] if col in reader.fieldnames]

        for row in reader:
            raw_hhidpn = row.get("hhidpn") or row.get("row") or ""

            # If we have a restricted patient ID set, enforce it
            if allowed_patient_ids:
                try:
                    pid_int = int(float(raw_hhidpn))
                except (TypeError, ValueError):
                    pid_int = None
                if pid_int is None or pid_int not in allowed_patient_ids:
                    continue

            patient_id = raw_hhidpn or str(len(patients) + 1)
            row_number = row.get("row") or ""
            diag_values = {col: row.get(col, "") for col in diag_columns}
            patient_content = "; ".join(
                f"{VARIABLE_MAPPING.get(col, col)}: {row.get(col, '')}"
                for col in columns
                if col in row
            )
            patients.append(
                {
                    "id": patient_id,
                    "row": row_number,
                    "content": patient_content,
                    "diag": diag_values,
                }
            )

            if max_patients is not None and len(patients) >= max_patients:
                break

    return patients


def call_model(prompt: str, model: str) -> dict:
    """Call OpenRouter with reasoning enabled and return separate content and reasoning_trace."""
    if not API_KEY:
        return {
            "content": "ERROR: OPENROUTER_API_KEY is not set in the environment",
            "reasoning_trace": "",
        }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        # Enable reasoning with default effort and include reasoning tokens in the response
        # (per OpenRouter docs, this is equivalent to "medium" effort by default)
        "reasoning": {
            "enabled": True,
            "exclude": False,
        },
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
    except Exception as exc:
        return {
            "content": f"ERROR: request failed: {exc}",
            "reasoning_trace": "",
        }

    # Try to parse JSON and handle common error shapes
    try:
        data = response.json()
    except Exception:
        return {
            "content": f"ERROR: non-JSON response ({response.status_code}): {response.text[:500]}",
            "reasoning_trace": "",
        }

    # Happy path: standard OpenRouter chat response
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        try:
            message = data["choices"][0]["message"]
            content = message.get("content", "")
            # Reasoning may appear either as a flat string or as structured reasoning_details
            reasoning = message.get("reasoning")
            reasoning_details = message.get("reasoning_details")

            reasoning_block = ""
            if reasoning:
                reasoning_block = str(reasoning)
            elif reasoning_details:
                try:
                    reasoning_block = json.dumps(reasoning_details)
                except Exception:
                    reasoning_block = str(reasoning_details)

            return {
                "content": content,
                "reasoning_trace": reasoning_block,
            }
        except Exception as exc:
            return {
                "content": f"ERROR: malformed choices structure: {exc}",
                "reasoning_trace": json.dumps(data)[:500],
            }

    # Error payload from OpenRouter
    if isinstance(data, dict) and "error" in data:
        return {
            "content": f"ERROR from API ({response.status_code}): {json.dumps(data['error'])[:500]}",
            "reasoning_trace": "",
        }

    # Fallback: unknown structure
    return {
        "content": f"ERROR: unexpected response structure ({response.status_code}): {json.dumps(data)[:500]}",
        "reasoning_trace": "",
    }


def main() -> None:
    diag_columns = [
        "lasso_dem",
        "expert_dem",
        "hurd_dem",
        "langa_weir_2cat",
        "cog_impair_2cat",
    ]
    raw_variable_sets = ACTIVE_VARIABLE_SETS if ACTIVE_VARIABLE_SETS is not None else [None]

    # Load restricted patient ID set if JSON exists
    allowed_patient_ids: Optional[set] = None
    if os.path.exists(PATIENT_ID_JSON):
        try:
            with open(PATIENT_ID_JSON, "r", encoding="utf-8") as fh:
                pid_data = json.load(fh)
            ids = pid_data.get("patient_ids", [])
            allowed_patient_ids = {int(pid) for pid in ids}
            print(f"Restricting reasoning run to {len(allowed_patient_ids)} patient IDs from {PATIENT_ID_JSON}")
        except Exception as exc:
            print(f"Warning: failed to load patient IDs from {PATIENT_ID_JSON}: {exc}")
            allowed_patient_ids = None

    variable_set_batches: List[tuple[str, List[dict]]] = []
    for variable_set in raw_variable_sets:
        normalized_set = variable_set
        variable_set_label = normalized_set or "all_columns"
        patient_data = load_patient_data(
            CSV_PATH,
            normalized_set,
            diag_columns,
            PATIENT_LIMIT,
            allowed_patient_ids=allowed_patient_ids,
        )
        variable_set_batches.append((variable_set_label, patient_data))

    rows: List[dict] = []
    futures: dict = {}

    total_patient_runs = sum(len(batch) for _, batch in variable_set_batches)
    total_tasks = len(PROMPTS) * len(MODELS) * total_patient_runs * RUNS_PER_PROMPT
    max_workers = min(10, max(1, total_tasks))

    # Match the legacy results_*.csv schema
    fieldnames = [
        "patient_id",
        "csv_row",
        "prompt_id",
        "run",
        "variable_set",
        "model",
        "prompt",
        "response",
        "reasoning_trace",
        "ablation_type",
        "ablated_variables",
        *diag_columns,
    ]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        fh.flush()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for variable_set_label, patient_data in variable_set_batches:
                for patient in patient_data:
                    for idx, prompt in enumerate(PROMPTS, start=1):
                        prompt_with_data = f"{prompt}\n\nPatient data:\n{patient['content']}"
                        for run_idx in range(1, RUNS_PER_PROMPT + 1):
                            for model in MODELS:
                                future = executor.submit(call_model, prompt_with_data, model)
                                futures[future] = (
                                    patient["id"],
                                    patient["row"],
                                    patient["diag"],
                                    idx,
                                    run_idx,
                                    prompt_with_data,
                                    model,
                                    variable_set_label,
                                )

            for future in concurrent.futures.as_completed(futures):
                (
                    patient_id,
                    row_number,
                    diag_values,
                    prompt_idx,
                    run_idx,
                    prompt_text,
                    model,
                    variable_set_label,
                ) = futures[future]
                result = future.result()
                response_text = result.get("content", "")
                reasoning_trace = result.get("reasoning_trace", "")

                # Map variable_set_label to ablation metadata (simple scheme)
                if variable_set_label == "all_columns":
                    ablation_type = "baseline"
                    ablated_variables = ""
                else:
                    ablation_type = variable_set_label
                    ablated_variables = variable_set_label
                print(
                    f"Finished patient {patient_id} (row {row_number}) "
                    f"vars {variable_set_label} prompt {prompt_idx} run {run_idx} with model {model}"
                )
                row_entry = {
                    "patient_id": patient_id,
                    "csv_row": row_number,
                    "prompt_id": prompt_idx,
                    "run": run_idx,
                    "variable_set": variable_set_label,
                    "model": model,
                    "prompt": prompt_text,
                    "response": response_text,
                    "reasoning_trace": reasoning_trace,
                    "ablation_type": ablation_type,
                    "ablated_variables": ablated_variables,
                    **diag_values,
                }
                rows.append(row_entry)
                writer.writerow(row_entry)
                fh.flush()

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


