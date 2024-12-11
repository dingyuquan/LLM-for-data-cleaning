import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
from tqdm import tqdm
from add_noise import PROB_OF_ERROR

# Customize your batch size for experiment, recommend 1 - 10
BATCH_SIZE = 5

# set up OpenAI API
OPEN_AI_KEY = "your_openai_api"
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


# Prompt templates
prompts = {
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


# Batch process data
for column in ["RATING", "GENRE", "ONE-LINE"]:
    print(f"Cleaning {column} column...")
    cleaned_column = []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc=f"Processing {column} batches"):
        batch_data = df[column].iloc[i:i + BATCH_SIZE]
        cleaned_batch = clean_batch(batch_data, column, prompts[column])

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
df.to_csv("cleaned_dataset_with_gpt4o-mini_" + str(PROB_OF_ERROR) + ".csv", index=False)

# Display cleaned dataset
print("Cleaned Dataset:")
print(df.head())

# Load gold truth dataset
gold_truth_df = pd.read_csv("Gold_truth_dataset.csv")

# Align both datasets
df = df.sort_values(by="MOVIES").reset_index(drop=True)
gold_truth_df = gold_truth_df.sort_values(by="MOVIES").reset_index(drop=True)


def compute_metrics(cleaned_series, gold_series):
    def normalize_genres(genre_list):
        return [genre.strip().lower() for genre in genre_list.split(",")]

    # Normalize GENRE columns before comparison
    if "GENRE" in cleaned_series.name.upper():
        cleaned_series = cleaned_series.dropna().apply(normalize_genres)
        gold_series = gold_series.dropna().apply(normalize_genres)

    # Ensure RATING columns are numeric before comparison
    if "RATING" in cleaned_series.name.upper():
        cleaned_series = pd.to_numeric(cleaned_series, errors='coerce').dropna()
        gold_series = pd.to_numeric(gold_series, errors='coerce').dropna()

    # Align both series
    cleaned_series, gold_series = cleaned_series.align(gold_series, join='inner')

    # Ensure both series have the same data type
    cleaned_series = cleaned_series.astype(str)
    gold_series = gold_series.astype(str)

    # Compute metrics
    accuracy = accuracy_score(gold_series, cleaned_series)
    precision = precision_score(gold_series, cleaned_series, average='micro')
    recall = recall_score(gold_series, cleaned_series, average='micro')
    f1 = f1_score(gold_series, cleaned_series, average='micro')

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# Evaluate 'MOVIES', 'RATING', and 'GENRE' columns
rating_metrics = compute_metrics(df["RATING"], gold_truth_df["RATING"])
genre_metrics = compute_metrics(df["GENRE"], gold_truth_df["GENRE"])
oneline_metrics = compute_metrics(df["ONE-LINE"], gold_truth_df["ONE-LINE"])

print("\nMetrics for 'RATING' column:")
print(rating_metrics)

print("\nMetrics for 'GENRE' column:")
print(genre_metrics)

print("\nMetrics for 'ONE-LINE' column:")
print(genre_metrics)
