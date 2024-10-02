import json
import os
import random
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from datasets import Dataset
from rich import print as rprint
from rich.progress import track
from sklearn.feature_extraction.text import TfidfVectorizer

YEARS=[ # dumps with filtered dir
    "2024-22",
    "2014-41",
    "2019-26",
    "2023-50",


]

def get_adult_files(input_path: str, language: str) -> List[str]:
    adult_content_path = Path(input_path) / "filtered"

    rprint(f"Looking for non-adult files in {adult_content_path}")

    files = [
        f for f in os.listdir(adult_content_path) if f.endswith(f"{language}.jsonl")
    ]

    # filter files that start with 2 since this is the indicator of adult content
    files = [f for f in files if f.startswith("2")]

    # filter the files that contain the language in the name
    files = [f for f in files if language in f]

    # convert to Path
    files = [Path(adult_content_path) / f for f in files]

    rprint(f"Found {len(files)} adult files")

    return files


def get_non_adult_files(input_path: str, language: str) -> List[str]:
    non_adult_content_path = Path(input_path) / language
    rprint(f"Looking for non-adult files in {non_adult_content_path}")

    files = [f for f in os.listdir(non_adult_content_path) if f.endswith("jsonl")]

    # convert to Path
    files = [Path(non_adult_content_path) / f for f in files]

    rprint(f"Found {len(files)} non-adult files")

    return files



def random_line_lazy(file_path: Path, to_keep: int) -> List[str]:
    """
    A lazy version of random_line that uses reservoir sampling for large files.

    Args:
        file_path: Path to the input file.
        to_keep: Number of random lines to keep.

    Returns:
        A list of selected text lines from the JSONL file.
    """
    selected_lines = []
    with open(file_path, "r") as f:
        for line_number, line in enumerate(f):
            # Read each line and parse it lazily
            if line_number < to_keep:
                selected_lines.append(json.loads(line)["text"])
            else:
                r = random.randint(0, line_number)
                if r < to_keep:
                    selected_lines[r] = json.loads(line)["text"]

    return selected_lines


def load_dataframe(
    adult_content_files: List[str],
    non_adult_content_files: List[str],
):
    """
    Loads a random sample of text lines from a list of adult and non-adult JSONL files
    and returns them as a DataFrame.

    Args:
        adult_content_files: List of file paths containing adult content.
        non_adult_content_files: List of file paths containing non-adult content.
        to_keep: Number of lines to randomly select from each file.

    Returns:
        A DataFrame containing the sampled text lines and their corresponding labels.
    """
    adult_content = []
    non_adult_content = []
    to_keep = 10000

    # Read adult content files
    for file in track(adult_content_files, description="Reading adult content files"):
        adult_content += random_line_lazy(file, to_keep)

    # rprint the number of lines and size list in memory
    rprint(
        f"Read {len(adult_content)} adult content lines with size {sum([len(x) for x in adult_content])/1_000_000:.2f} MB"
    )

    # Read non-adult content files
    for file in track(
        non_adult_content_files, description="Reading non-adult content files"
    ):
        non_adult_content += random_line_lazy(file, to_keep//180) # non adult content is 18x larger than adult content

    rprint(
        f"Read {len(non_adult_content)} non-adult content lines with size {sum([len(x) for x in non_adult_content])/1_000_000:.2f} MB"
    )

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "text": adult_content + non_adult_content,
            "label": [1] * len(adult_content) + [0] * len(non_adult_content),
        }
    )

    rprint(f"Created DataFrame with {len(df)} rows")
    return df


def vectorize_data(df: pd.DataFrame):
    rprint("Vectorizing the data")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        # stop_words='english',
        ngram_range=(1, 4),
        use_idf=True,
        max_features=30000,  # Limit the number of features
        min_df=5,  # Ignore terms that appear in less than 5 documents


    )

    # Use fit_transform directly on the dataframe's text column
    X = vectorizer.fit_transform(df["text"])

    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    rprint(f"Vectorized data with shape {df.shape}")

    return X, vectorizer


def save_vectorizer(vectorizer, output_path: Path, language: str):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    vectorizer_file = output_path / f"vectorizer_{language}.joblib"
    joblib.dump(vectorizer, vectorizer_file)
    rprint(f"Vectorizer saved to {vectorizer_file}")


def save_df(df, output_path: Path, name: str):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    df_file = output_path / f"df_{name}.joblib"
    joblib.dump(df, df_file)
    rprint(f"DataFrame saved to {df_file}")


def generate_text_data(input_dir: str, output_path: str, language: str):
    
    adult_files=[]
    non_adult_files=[]
    for dumps in YEARS:
        new_input_dir = f"{input_dir}/{dumps}"
    
        adult_files += get_adult_files(new_input_dir, language)
        non_adult_files += get_non_adult_files(new_input_dir, language)

        if not adult_files:
            rprint(f"No adult files found in {new_input_dir}")
            continue

        if not non_adult_files:
            rprint(f"No non-adult files found in {new_input_dir}")
            continue

    # shuffle the files
    random.shuffle(adult_files)
    random.shuffle(non_adult_files)

    rprint(f"Found {len(adult_files)} adult files and {len(non_adult_files)} non-adult files.\nNow creating df")


    df = load_dataframe(adult_files, non_adult_files)
    save_df(df, output_path, f"text_{language}")

    return df


def load_text_data(input_dir: str, output_path: str, language: str):
    df_file = Path(output_path) / f"df_text_{language}.joblib"
    if df_file.exists():
        df = joblib.load(df_file)
        rprint(f"Loaded DataFrame from {df_file}")
    else:
        df = generate_text_data(input_dir, output_path, language)

    return df


def generate_vectorized_data(df: pd.DataFrame, output_path: str, language: str):
    df_vector, vectorizer = vectorize_data(df)
    save_vectorizer(vectorizer, output_path, language)
    save_df(df_vector, output_path, f"vectorized_{language}")

    return df_vector, vectorizer


def load_vectorized_data(df: pd.DataFrame, output_path: str, language: str):
    vectorizer_file = Path(output_path) / f"vectorizer_{language}.joblib"
    df_file = Path(output_path) / f"df_vectorized_{language}.joblib"

    if vectorizer_file.exists() and df_file.exists():
        vectorizer = joblib.load(vectorizer_file)
        df_vector = joblib.load(df_file)
        rprint(f"Loaded vectorizer from {vectorizer_file}")
        rprint(f"Loaded DataFrame from {df_file}")
    else:
        df_vector, vectorizer = generate_vectorized_data(df, output_path, language)

    return df_vector, vectorizer


def load_data(input_dir: str, output_path: str, language: str):
    df = load_text_data(input_dir, output_path, language)
    #df_vector, vectorizer = load_vectorized_data(df, output_path, language)
    # Assuming 'data' is a list of dictionaries or a pandas DataFrame
    if isinstance(df, list):
        dataset_dict = {key: [item[key] for item in df] for key in df[0].keys()}
    elif isinstance(df, pd.DataFrame):
        dataset_dict = {column: df[column].tolist() for column in df.columns}
    
    dataset_dict = dict(list(dataset_dict)[:10])

    dataset = Dataset.from_dict(dataset_dict)
    # Split the dataset into train and validation sets
    dataset = dataset.train_test_split(test_size=0.2)

    return dataset
