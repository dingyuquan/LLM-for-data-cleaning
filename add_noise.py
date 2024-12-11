import pandas as pd
import random
import re

# Modify error rate here
PROB_OF_ERROR = 0.5

# Load dataset from CSV file
df = pd.read_csv("movies.csv")
random.seed(42)

# Sample 2000 rows from the dataset
df = df.sample(n=2000, random_state=42)
# Fill empty fields in specific columns, no empty row in ONE-LINE column
df["GENRE"].fillna("None", inplace=True)
df["RATING"].fillna(0.0, inplace=True)

# Generate gold_truth_dataset
df.to_csv("Gold_truth_dataset.csv", index=False)


# Function to remove the decimal point in 'Rating'
def add_noise_to_rating(rating):
    """Introduce noise to 'Rating' by removing the decimal point."""
    try:
        if pd.isnull(rating) or rating == "":
            return rating
        rating = str(float(rating))  # Ensure it's a string with a valid format
        if random.random() < PROB_OF_ERROR:  # Add noise with a certain probability
            return rating.replace(".", "")  # Remove the decimal point
        return rating
    except:
        return rating


# Function to add typo to 'Genres'
def add_typo_to_genres(genres):
    if pd.isnull(genres) or genres.strip() == "" or genres.strip() == "None":
        return genres
    genres_list = [genre.strip() for genre in genres.split(",")]
    genres_list = [introduce_typo(genre) for genre in genres_list]
    return ", ".join(genres_list)


# Helper function
def introduce_typo(word):
    if len(word) <= 1:
        return word
    if random.random() < PROB_OF_ERROR:
        typo_type = random.choice(["swap", "remove", "replace", "duplicate"])
        if typo_type == "swap" and len(word) > 2:
            idx = random.randint(0, len(word) - 2)
            word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
        elif typo_type == "remove":
            idx = random.randint(0, len(word) - 1)
            word = word[:idx] + word[idx + 1:]
        elif typo_type == "replace":
            idx = random.randint(0, len(word) - 1)
            word = word[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[idx + 1:]
        elif typo_type == "duplicate":
            idx = random.randint(0, len(word) - 1)
            word = word[:idx] + word[idx] + word[idx] + word[idx + 1:]
    return word


# Function to add special characters to ONE-LINE field
def add_noise_to_one_line(one_line):
    if pd.isnull(one_line) or one_line == "" or one_line == "\nAdd a Plot\n":
        return one_line

    if random.random() < PROB_OF_ERROR:
        # List of special characters to inject
        special_chars = ['@', '#', '$', '%', '^', '&', '*']

        # Randomly decide the number of characters to insert
        num_chars = random.randint(1, 10)

        # Choose random positions in the string to insert the characters
        positions = sorted(random.sample(range(len(one_line)), num_chars))

        # Insert special characters at chosen positions
        for pos in reversed(positions):  # Insert in reverse to avoid shifting indices
            one_line = one_line[:pos] + random.choice(special_chars) + one_line[pos:]

    return one_line


# Apply noise to the dataset (all errors in one file)
df_with_errors = df.copy()
df_with_errors["ONE-LINE"] = df_with_errors["ONE-LINE"].apply(add_noise_to_one_line)
df_with_errors["RATING"] = df_with_errors["RATING"].apply(add_noise_to_rating)
df_with_errors["GENRE"] = df_with_errors["GENRE"].apply(add_typo_to_genres)

file_name = f"dataset_with_all_errors_" + str(PROB_OF_ERROR) + ".csv"

# Save the modified dataset with all errors
df_with_errors.to_csv(file_name, index=False)

# Display the modified DataFrame
print(f"Dataset with all errors saved to {file_name}")
print(df_with_errors.head())
