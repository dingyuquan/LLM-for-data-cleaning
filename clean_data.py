import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
from tqdm import tqdm
from add_noise import PROB_OF_ERROR

# Customize your batch size for experiment, recommend 1 - 10
BATCH_SIZE = 5

# Define the prompt method
PROMPT_METHOD = "CHAIN_OF_THOUGHT"  # Options: "FEW_SHOT", "ZERO_SHOT", "CHAIN_OF_THOUGHT"

# set up OpenAI API
OPEN_AI_KEY = "YOUR_OPENAI_KEY"
client = OpenAI(api_key=OPEN_AI_KEY)


def get_response(input):
    response = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                        {"role": "user", "content": input}],
                                              max_tokens=1000,
                                              temperature=0,
                                              timeout=60)
    response = response.choices[0].message.content.strip()
    # Debug: Log response for unexpected results
    # print(f"DEBUG: GPT response length = {len(response.splitlines())}")
    if len((response.splitlines())) != BATCH_SIZE:
        print(response)
    response = response.replace("```", '')
    response = response.strip()

    return response


# Load the noisy dataset
df = pd.read_csv("dataset_with_all_errors_" + str(PROB_OF_ERROR) + ".csv")


# Cleaning function
def clean_batch(batch_data, column_name, prompt_template):
    dirty_input = batch_data.to_csv(sep='\t', index=False, header=None)
    prompt = prompt_template.format(input_text=dirty_input)
    response = get_response(prompt)
    cleaned_data = [line for line in response.split("\n") if line.strip()]
    return cleaned_data


# Few-shot Prompt templates
few_shot_prompts = {
    "RATING": "You will be given a list of movie ratings. Correct each rating's format to ensure it is a valid numeric value \
    with one decimal point, within the range 0.0 to 10.0. For each input rating, there must be exactly one corresponding output rating. \
    Do not add any extra text or explanations. Ensure the number of output lines matches the number of input lines.\
    If the rating is already valid, return it as is.\n\n{input_text}\n\nOutput only the corrected ratings as a list, one per line. \
    REMEMBER: DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THE LINE OF OUTPUT MUST BE EQUAL TO " + str(BATCH_SIZE)
    ,
    "GENRE": "You will be given a list of movie genres. Correct any spelling errors and separate genres with commas. \
    The number of output lines must match the number of input lines exactly. Do not add any extra text or explanations. \
    If the input is already valid or 'None', return it unchanged. Keep the genres in the original order.\n\n{input_text}\n\nOutput only\
    the corrected genres as a list, one per line. REMEMBER: DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THE LINE OF OUTPUT MUST BE EQUAL TO " + str(BATCH_SIZE)
    ,
    "ONE-LINE": "You will be given a list of movie descriptions. Remove all special characters (including @, #, $, %, ^, & and *), \
    to make it valid. Do not alter the meaning of the text or the original order of words. \
    If the input is already valid or is None, return it unchanged. Ensure the number of output lines matches the number of input lines exactly. \
    Do not add any additional text or explanations.\n\n{input_text}\n\nOutput only the cleaned movie descriptions as a list, one per line. \
    REMEMBER: DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THE LINE OF OUTPUT MUST BE EQUAL TO " + str(BATCH_SIZE)

}
# Zero-shot Prompt templates
zero_shot_prompts = {
    "RATING": "You are provided a list of movie ratings. Correct the data.\
If the rating is already valid, return it unchanged. Return only the corrected ratings, one per line, \
matching the number of input lines exactly.\n\n{input_text}\n\nOutput only the corrected ratings as a list, one per line. \
    REMEMBER: DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THE LINE OF OUTPUT MUST BE EQUAL TO " + str(BATCH_SIZE)
    ,
    "GENRE": "You are provided a list of movie genres. Correct the data. \
If the input is already valid or 'None', return it unchanged. Return only the corrected genres, one per line, \
matching the number of input lines exactly.\n\n{input_text}\n\nOutput only the corrected ratings as a list, one per line. \
    REMEMBER: DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THE LINE OF OUTPUT MUST BE EQUAL TO " + str(BATCH_SIZE)
    ,
    "ONE-LINE": "You are provided a list of movie descriptions. Correct the data. \
If the input is already valid or 'None', return it unchanged. \
Return only the cleaned descriptions, one per line, matching the number of input lines exactly.\n\n{input_text}\n\nOutput only the corrected ratings as a list, one per line. \
    REMEMBER: DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THE LINE OF OUTPUT MUST BE EQUAL TO " + str(BATCH_SIZE)
}

# Chain-of-Thought Prompt templates
cot_prompts = {
    "RATING": "You are provided a list of movie ratings. Your task is to ensure each rating is a valid numeric value \
with one decimal point, within the range 0.0 to 10.0. Follow the example provided below to understand why and how to clean the ratings. \
Then apply the same logic to the given input.\n\n\
Example:\n\
Input:\n\
8.5\n\
12\n\
bad_data\n\
\n\
Reasoning:\n\
1. The first rating '8.5' is valid because it is within the range 0.0 to 10.0 and has one decimal point. No changes are needed.\n\
2. The second rating '12' exceeds the valid range. It should be capped at the value '1.2'.\n\
3. The third entry 'bad_data' is not a numeric value and should be replaced with '0.0' as a default for invalid entries.\n\
\n\
Output:\n\
8.5\n\
1.2\n\
0.0\n\
\n\
Now apply the same reasoning to the following input ratings:\n\n{input_text}\n\nOutput only the corrected ratings as a list, one per line.\
REMEMBER: DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THE LINE OF OUTPUT MUST BE EQUAL TO " + str(BATCH_SIZE),

    "GENRE": "You are provided a list of movie genres. Your task is to correct any spelling errors and format the genres \
as comma-separated values. Follow the example provided below to understand the reasoning. Then apply the same logic to the input data.\n\n\
Example:\n\
Input:\n\
actiion, dram\n\
comedy\n\
None\n\
\n\
Reasoning:\n\
1. The first line contains spelling errors: 'actiion' should be corrected to 'action', and 'dram' should be corrected to 'drama'.\n\
2. The second line is valid as 'comedy' is correctly spelled. No changes are needed.\n\
3. The third line is 'none', which is a valid placeholder. It should remain unchanged.\n\
\n\
Output:\n\
action, drama\n\
comedy\n\
None\n\
\n\
Now apply the same reasoning to the following input genres:\n\n{input_text}\n\nOutput only the corrected genres as a list, one per line.\
REMEMBER: DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THE LINE OF OUTPUT MUST BE EQUAL TO " + str(BATCH_SIZE),

    "ONE-LINE": "You are provided a list of movie descriptions. Your task is to remove all special characters (such as @, #, $, %, ^, & and *) \
while keeping the meaning and word order intact. Follow the example below to understand the reasoning. Then apply the same logic to the input data.\n\n\
Example:\n\
Input:\n\
A thr@illing adventu#re\n\
An epic jo%urney of disco^very\n\
Add a Plot\n\
\n\
Reasoning:\n\
1. In the first line, remove the special characters '@' and '#', resulting in 'A thrilling adventure'.\n\
2. In the second line, remove the special characters '%' and '^', resulting in 'An epic journey of discovery'.\n\
3. The third line is 'Add a Plot', which is valid and should remain unchanged.\n\
\n\
Output:\n\
A thrilling adventure\n\
An epic journey of discovery\n\
Add a Plot\n\
\n\
Now apply the same reasoning to the following input descriptions:\n\n{input_text}\n\nOutput only the cleaned movie descriptions as a list, one per line.\
REMEMBER: DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THE LINE OF OUTPUT MUST BE EQUAL TO " + str(BATCH_SIZE)
}

# Batch process data
for column in ["RATING", "GENRE", "ONE-LINE"]:
    # Select prompt templates based on PROMPT_METHOD
    if PROMPT_METHOD == "FEW_SHOT":
        selected_prompts = few_shot_prompts
    elif PROMPT_METHOD == "ZERO_SHOT":
        selected_prompts = zero_shot_prompts
    elif PROMPT_METHOD == "CHAIN_OF_THOUGHT":
        selected_prompts = cot_prompts
    else:
        raise ValueError(f"Invalid PROMPT_METHOD: {PROMPT_METHOD}.")

    print(f"Cleaning {column} column...")
    cleaned_column = []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Processing {column} batches"):
        batch_data = df[column].iloc[i:i + BATCH_SIZE]
        cleaned_batch = clean_batch(batch_data, column, selected_prompts[column])

        # Check and fill missing rows
        if len(cleaned_batch) != len(batch_data):
            print(f"Warning: Mismatch in batch size for {column} at index {i}.")
            # Fill missing rows with placeholders
            while len(cleaned_batch) < len(batch_data):
                cleaned_batch.append("")  # Use an appropriate placeholder
            # Truncate excess rows if too many
            cleaned_batch = cleaned_batch[:len(batch_data)]

        cleaned_column.extend(cleaned_batch)
    df[column] = cleaned_column

# Save cleaned dataset
df.to_csv("cleaned_dataset_with_gpt4o-mini_" + str(PROB_OF_ERROR) + "_" + PROMPT_METHOD + ".csv", index=False)

# Display cleaned dataset
print("Cleaned Dataset:")
print(df.head())

# # Load gold truth dataset
# gold_truth_df = pd.read_csv("Gold_truth_dataset.csv")
#
# # Align both datasets
# df = df.sort_values(by="MOVIES").reset_index(drop=True)
# gold_truth_df = gold_truth_df.sort_values(by="MOVIES").reset_index(drop=True)
#
#
# def compute_overall_metrics(cleaned_df, gold_df):
#     # Ensure both dataframes have the same structure
#     assert cleaned_df.shape == gold_df.shape, "Cleaned and gold dataframes must have the same shape."
#
#     # Align both dataframes
#     cleaned_df, gold_df = cleaned_df.align(gold_df, join='inner', axis=0)
#
#     # Flatten both dataframes to compare row-wise
#     cleaned_flat = cleaned_df.astype(str).values.flatten()
#     gold_flat = gold_df.astype(str).values.flatten()
#
#     # Compute metrics
#     accuracy = accuracy_score(gold_flat, cleaned_flat)
#     precision = precision_score(gold_flat, cleaned_flat, average='macro')
#     recall = recall_score(gold_flat, cleaned_flat, average='macro')
#     f1 = f1_score(gold_flat, cleaned_flat, average='macro')
#
#     return {
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall,
#         "F1 Score": f1
#     }
#
#
# # Normalize and prepare the entire dataframe for comparison
# def normalize_dataframe(df):
#     df_normalized = df.copy()
#
#     # Normalize each column as per its type
#     if "GENRE" in df.columns:
#         df_normalized["GENRE"] = df_normalized["GENRE"].dropna().apply(
#             lambda x: [genre.strip().lower() for genre in x.split(",")]
#         )
#     if "RATING" in df.columns:
#         df_normalized["RATING"] = pd.to_numeric(df_normalized["RATING"], errors='coerce').fillna(0.0).astype(float)
#     if "ONE-LINE" in df.columns:
#         df_normalized["ONE-LINE"] = df_normalized["ONE-LINE"].fillna("None").astype(str)
#     return df_normalized
#
#
# # Normalize both cleaned and gold dataframes
# cleaned_df_normalized = normalize_dataframe(df)
# gold_df_normalized = normalize_dataframe(gold_truth_df)
#
# # Compute overall metrics for the entire table
# overall_metrics = compute_overall_metrics(cleaned_df_normalized, gold_df_normalized)
#
# print("\nOverall Metrics for the Entire Table:")
# print(overall_metrics)


# def compute_metrics(cleaned_series, gold_series):
#     def normalize_genres(genre_list):
#         return [genre.strip().lower() for genre in genre_list.split(",")]
#
#     # Normalize GENRE columns before comparison
#     if "GENRE" in cleaned_series.name.upper():
#         cleaned_series = cleaned_series.dropna().apply(normalize_genres)
#         gold_series = gold_series.dropna().apply(normalize_genres)
#
#     # Ensure RATING columns are numeric before comparison
#     if "RATING" in cleaned_series.name.upper():
#         cleaned_series = pd.to_numeric(cleaned_series, errors='coerce').dropna()
#         gold_series = pd.to_numeric(gold_series, errors='coerce').dropna()
#
#     # Align both series
#     cleaned_series, gold_series = cleaned_series.align(gold_series, join='inner')
#
#     # Ensure both series have the same data type
#     cleaned_series = cleaned_series.astype(str)
#     gold_series = gold_series.astype(str)
#
#     # Compute metrics
#     accuracy = accuracy_score(gold_series, cleaned_series)
#     precision = precision_score(gold_series, cleaned_series, average='micro')
#     recall = recall_score(gold_series, cleaned_series, average='micro')
#     f1 = f1_score(gold_series, cleaned_series, average='micro')
#
#     return {
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall,
#         "F1 Score": f1
#     }
#
#
# # Evaluate 'MOVIES', 'RATING', and 'GENRE' columns
# rating_metrics = compute_metrics(df["RATING"], gold_truth_df["RATING"])
# genre_metrics = compute_metrics(df["GENRE"], gold_truth_df["GENRE"])
# oneline_metrics = compute_metrics(df["ONE-LINE"], gold_truth_df["ONE-LINE"])
#
# print("\nMetrics for 'RATING' column:")
# print(rating_metrics)
#
# print("\nMetrics for 'GENRE' column:")
# print(genre_metrics)
#
# print("\nMetrics for 'ONE-LINE' column:")
# print(genre_metrics)
