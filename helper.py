import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.metrics import confusion_matrix

def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity


def sensitivity_specificity(y_true, y_pred):

    def compute_metrics(gold, pred):
        mask = pred.isin([0, 1])
        gold = gold[mask]
        pred = pred[mask]

        n = len(gold)
        if n == 0:
            return 0, 0, 0, 0, 0, 0
        try:
            tn, fp, fn, tp = confusion_matrix(gold, pred).ravel()
        except Exception:
            tn = fp = fn = tp = 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        return sensitivity, specificity,  precision, accuracy,f1, n

    # Ensure no NaN values in the gold standard and predictions
    valid_indices_v1 = y_pred['response_lasso_label'].notna() & y_true.notna()
    valid_indices_v2 = y_pred['response_expert_label'].notna() & y_true.notna()
    valid_indices_v3 = y_pred['response_everything_label'].notna() & y_true.notna()

    # Extract valid data
    gold_standard_v1 = y_true[valid_indices_v1]
    predictions_v1 = y_pred['response_lasso_label'][valid_indices_v1]

    gold_standard_v2 = y_true[valid_indices_v2]
    predictions_v2 = y_pred['response_expert_label'][valid_indices_v2]

    gold_standard_v3 = y_true[valid_indices_v3]
    predictions_v3 = y_pred['response_everything_label'][valid_indices_v3]

    # Calculate metrics
    sensitivity_v1, specificity_v1, precision_v1, accuracy_v1, f1_v1, n_v1 = compute_metrics(gold_standard_v1, predictions_v1)
    sensitivity_v2, specificity_v2, precision_v2, accuracy_v2, f1_v2, n_v2 = compute_metrics(gold_standard_v2, predictions_v2)
    sensitivity_v3, specificity_v3, precision_v3, accuracy_v3, f1_v3, n_v3 = compute_metrics(gold_standard_v3, predictions_v3)


    print(f"response_lasso - N: {n_v1}, Sensitivity: {sensitivity_v1:.3f}, Specificity: {specificity_v1:.3f}, Precision: {precision_v1:.3f}, Accuracy: {accuracy_v1:.3f}, F1: {f1_v1:.3f}")
    print(f"response_expert - N: {n_v2}, Sensitivity: {sensitivity_v2:.3f}, Specificity: {specificity_v2:.3f}, Precision: {precision_v2:.3f}, Accuracy: {accuracy_v2:.3f}, F1: {f1_v2:.3f}")
    print(f"response_everything - N: {n_v3}, Sensitivity: {sensitivity_v3:.3f}, Specificity: {specificity_v3:.3f}, Precision: {precision_v3:.3f}, Accuracy: {accuracy_v3:.3f}, F1: {f1_v3:.3f}")

    return {
        'response_lasso': (n_v1, sensitivity_v1, specificity_v1, precision_v1, accuracy_v1, f1_v1),
        'response_expert': (n_v2, sensitivity_v2, specificity_v2, precision_v2, accuracy_v2, f1_v2),
        'response_everything': (n_v3, sensitivity_v3, specificity_v3, precision_v3, accuracy_v3, f1_v3)
    }




# original series
# original series
# def process_response(d1, col_name):
#     s = d1[col_name]

#     # keep original index and handle NaNs
#     s_str = s.where(s.notna(), None).astype(object)

#     # normalize both comma and newline separators
#     s_str = s_str.str.replace('\n', ',', regex=False)

#     # split into two parts
#     split_df = s_str.str.split(',', n=1, expand=True)

#     # extract label and score
#     label_series = pd.to_numeric(split_df[0].str.strip(), errors='coerce').astype('Float32')
#     if 1 in split_df.columns:
#         score_series = pd.to_numeric(split_df[1].str.strip(), errors='coerce').astype('Float32')
#     else:
#         score_series = pd.Series([pd.NA] * len(s), index=s.index, dtype='float32')

#     # assemble final DataFrame
#     result = pd.DataFrame({
#         f'{col_name}_label': label_series,
#         f'{col_name}_score': score_series
#     }, index=s.index)

#     return result



def process_response(d1, col_name):
    s = d1[col_name]

    s_clean = (
        s.where(s.notna(), None)
         .astype(object)
         .str.replace('\n', ',', regex=False)
    )

    extracted = s_clean.apply(
        lambda x: re.findall(r"[-+]?\d*\.?\d+", str(x)) if x is not None else []
    )

    label_list = []
    score_list = []

    for nums in extracted:
        if len(nums) == 0:
            label_list.append(None)
            score_list.append(None)
        elif len(nums) == 1:
            label_list.append(nums[0])
            score_list.append(None)
        else:
            label_list.append(nums[0])
            score_list.append(nums[1])

    label_series = pd.to_numeric(label_list, errors='coerce').astype("float32")
    score_series = pd.to_numeric(score_list, errors='coerce').astype("float32")

    result = pd.DataFrame({
        "patient_id": d1["patient_id"],                
        f"{col_name}_label": label_series,
        f"{col_name}_score": score_series
    }, index=d1.index)

    return result

