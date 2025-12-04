import concurrent.futures
import csv
import json
import os
import threading
import time
from typing import List, Optional
import requests

# Data CSV Input/Output & Save Information
CSV_PATH = "bm227_final_data_1000_sampled.csv"
RUN_TIMESTAMP = int(time.time())
RESULTS_DIR = "llm_results"
OUTPUT_CSV = os.path.join(RESULTS_DIR, f"results_{RUN_TIMESTAMP}.csv")

# OpenRouter API Information
API_KEY = "YOUR_OPEN_ROUTER_API_KEY_HERE" # Please do not leak this or spam high cost requests; it's connected to my CC. 
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
RUNS_PER_PROMPT = 1
PROMPTS: List[str] = [
    "You are an expert epidemiologist and clinician specializing in cognitive disorders. You will be given survey data for an individual, including sociodemographic variables and physical and cognitive health measures. Some data may be missing; ignore missingness and base your decision only on what is available. Do not assume dementia from missingness alone. Your task is to classify dementia status in two ways (Binary, and as a probability). Output 1: 0 = without dementia, 1 = with dementia. Output 2: A probability ranging from 0 meaning the probability of not having dementia to 1 meaning having dementia. Respond only with output 1, output 2.",
]

# Patient data configuration
PATIENT_LIMIT: Optional[int] = 1  # None for all
ACTIVE_VARIABLE_SETS: Optional[List[Optional[str]]] = ["langa_weir_vars_only", "expert_model", "everything"]  # None for all variables
VARIABLE_SETS = {
    "langa_weir_vars_only": [
        "imrc",
        "dlrc",
        "bwc20",
        "ser7",
        "prmem",
        "int_proxy_cog_rating",
        "iadl5h"
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

# Load patient data from CSV and combine it with the prompt. 
def load_patient_data(
    csv_path: str,
    variable_set: Optional[str],
    diag_columns: List[str],
    max_patients: Optional[int],
) -> List[dict]:
    
    patients: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        columns = reader.fieldnames
        if variable_set:
            columns = [col for col in VARIABLE_SETS[variable_set] if col in reader.fieldnames]

        for row in reader:
            patient_id = row.get("hhidpn") or row.get("row") or str(len(patients) + 1)
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


def call_model(
    prompt: str,
    model: str,
    rate_limit_state: dict,
    rate_limit_condition: threading.Condition,
    max_attempts: int = 5,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        with rate_limit_condition:
            while rate_limit_state["inflight"] >= rate_limit_state["max_parallel_requests"]:
                rate_limit_condition.wait()
            rate_limit_state["inflight"] += 1

        response: Optional[requests.Response] = None
        pause = False
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=120,
            )
            if response.status_code == 429:
                last_error = f"429: {response.text[:200]}"
                pause = True
            elif response.ok:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                last_error = f"{response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last_error = str(exc)
        finally:
            with rate_limit_condition:
                rate_limit_state["inflight"] -= 1
                rate_limit_condition.notify_all()

        if pause and attempt < max_attempts:
            print("429 (rate limit); pausing new requests for 30 seconds.")
            time.sleep(30)

    raise RuntimeError(f"Failed after {max_attempts} attempts: {last_error}")


def main() -> None:
    diag_columns = [
        "lasso_dem",
        "expert_dem",
        "hurd_dem",
        "langa_weir_2cat",
        "cog_impair_2cat",
    ]
    response_column_lookup = {
        "langa_weir_vars_only": "response_langa_weir",
        "expert_model": "response_expert",
        "everything": "response_everything",
        "all_columns": "response_all_columns",
    }
    prompt_column_lookup = {
        "langa_weir_vars_only": "prompt_langa_weir",
        "expert_model": "prompt_expert",
        "everything": "prompt_everything",
        "all_columns": "prompt_all_columns",
    }

    raw_variable_sets = ACTIVE_VARIABLE_SETS if ACTIVE_VARIABLE_SETS is not None else [None]
    variable_set_batches: List[tuple[str, str, str, List[dict]]] = []
    prompt_columns: List[str] = []
    response_columns: List[str] = []
    for variable_set in raw_variable_sets:
        variable_set_label = variable_set or "all_columns"
        response_column = response_column_lookup.get(variable_set_label, f"response_{variable_set_label}")
        prompt_column = prompt_column_lookup.get(variable_set_label, f"prompt_{variable_set_label}")
        prompt_columns.append(prompt_column)
        response_columns.append(response_column)
        patient_data = load_patient_data(CSV_PATH, variable_set, diag_columns, PATIENT_LIMIT)
        variable_set_batches.append((variable_set_label, response_column, prompt_column, patient_data))

    fieldnames = [
        "patient_id",
        "csv_row",
        "prompt_id",
        "run",
        "model",
        *prompt_columns,
        *response_columns,
        *diag_columns,
    ]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows_by_key: dict = {}
    tasks: List[dict] = []

    for variable_set_label, response_column, prompt_column, patient_data in variable_set_batches:
        for patient in patient_data:
            patient_id, row_number = patient["id"], patient["row"]
            for prompt_idx, prompt in enumerate(PROMPTS, start=1):
                for run_idx in range(1, RUNS_PER_PROMPT + 1):
                    for model in MODELS:
                        key = (patient_id, row_number, prompt_idx, run_idx, model)
                        row = rows_by_key.setdefault(
                            key,
                            {
                                "patient_id": patient_id,
                                "csv_row": row_number,
                                "prompt_id": prompt_idx,
                                "run": run_idx,
                                "model": model,
                                **{col: "" for col in prompt_columns},
                                **{col: "" for col in response_columns},
                                **patient["diag"],
                            },
                        )
                        row[prompt_column] = f"{prompt}\n\nPatient data:\n{patient['content']}"
                        if not row.get(response_column):
                            tasks.append(
                                {
                                    "key": key,
                                    "response_column": response_column,
                                    "prompt_text": row[prompt_column],
                                    "model": model,
                                }
                            )

    if not tasks:
        print("Nothing to do; all responses already present.")
        return

    rate_limit_state = {"max_parallel_requests": 50, "inflight": 0}
    rate_limit_condition = threading.Condition()
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=rate_limit_state["max_parallel_requests"]) as executor:
        future_map = {
            executor.submit(
                call_model,
                task["prompt_text"],
                task["model"],
                rate_limit_state,
                rate_limit_condition,
            ): task
            for task in tasks
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            task = future_map[future]
            try:
                answer = future.result()
            except Exception as exc:
                answer = f"ERROR after retries: {exc}"
            rows_by_key[task["key"]][task["response_column"]] = answer

            elapsed = max(0.001, time.time() - start_time)
            rate = idx / elapsed
            remaining = len(tasks) - idx
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta_text = (
                f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
                if eta_seconds != float("inf")
                else "--:--"
            )
            print(f"Progress {idx}/{len(tasks)} | ETA {eta_text}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows_by_key):
            writer.writerow(rows_by_key[key])

    print(f"Saved {len(rows_by_key)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
