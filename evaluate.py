import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROB_OF_ERROR = 0.8
PROMPT_METHOD = "FEW_SHOT"

df = pd.read_csv("cleaned_dataset_with_gpt4o-mini_" + str(PROB_OF_ERROR) + "_" + PROMPT_METHOD + ".csv")

# Load gold truth dataset

gold_truth_df = pd.read_csv("Gold_truth_dataset.csv")

# Align both datasets
df = df.sort_values(by="MOVIES").reset_index(drop=True)
gold_truth_df = gold_truth_df.sort_values(by="MOVIES").reset_index(drop=True)


def compute_overall_metrics(cleaned_df, gold_df):
    # Ensure both dataframes have the same structure
    assert cleaned_df.shape == gold_df.shape, "Cleaned and gold dataframes must have the same shape."

    # Align both dataframes
    cleaned_df, gold_df = cleaned_df.align(gold_df, join='inner', axis=0)

    # Flatten both dataframes to compare row-wise
    cleaned_flat = cleaned_df.astype(str).values.flatten()
    gold_flat = gold_df.astype(str).values.flatten()

    # Compute metrics
    accuracy = accuracy_score(gold_flat, cleaned_flat)
    precision = precision_score(gold_flat, cleaned_flat, average='macro')
    recall = recall_score(gold_flat, cleaned_flat, average='macro')
    f1 = f1_score(gold_flat, cleaned_flat, average='macro')

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# Normalize and prepare the entire dataframe for comparison
def normalize_dataframe(df):
    df_normalized = df.copy()

    # Normalize each column as per its type
    if "GENRE" in df.columns:
        df_normalized["GENRE"] = df_normalized["GENRE"].dropna().apply(
            lambda x: [genre.strip().lower() for genre in x.split(",")]
        )
    if "RATING" in df.columns:
        df_normalized["RATING"] = pd.to_numeric(df_normalized["RATING"], errors='coerce').fillna(0.0).astype(float)
    if "ONE-LINE" in df.columns:
        df_normalized["ONE-LINE"] = df_normalized["ONE-LINE"].fillna("None").apply(lambda x: x.strip())
    return df_normalized


# Normalize both cleaned and gold dataframes
cleaned_df_normalized = normalize_dataframe(df)
gold_df_normalized = normalize_dataframe(gold_truth_df)

# Compute overall metrics for the entire table
overall_metrics = compute_overall_metrics(cleaned_df_normalized, gold_df_normalized)

print("\nOverall Metrics for the Entire Table:")
print(overall_metrics)
