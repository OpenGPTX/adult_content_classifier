import json
import random
from click import Tuple
import joblib
import pandas as pd
from rich.progress import track
from rich import print as rprint

import os
from typing import List
from pathlib import Path

YEARS = [  # dumps with filtered dir
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

    # shuffle the files
    random.shuffle(files)

    return files


def get_non_adult_files(input_path: str, language: str) -> List[str]:
    non_adult_content_path = Path(input_path) / language
    rprint(f"Looking for non-adult files in {non_adult_content_path}")

    files = [f for f in os.listdir(non_adult_content_path) if f.endswith("jsonl")]

    # convert to Path
    files = [Path(non_adult_content_path) / f for f in files]

    rprint(f"Found {len(files)} non-adult files")

    # shuffle the files
    random.shuffle(files)

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


def create_dataframe_from_docs(
    adult_content_files: List[str],
    non_adult_content_files: List[str],
    lines_to_keep: int = 100000,
) -> pd.DataFrame:
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

    to_keep_adult = lines_to_keep // len(adult_content_files)
    if not to_keep_adult:
        to_keep_adult = 1

    # Read adult content files
    for file in track(
        adult_content_files,
        description=f"Reading adult content files (keeping {to_keep_adult} lines x file)",
    ):
        adult_content += random_line_lazy(file, to_keep_adult)

    # rprint the number of lines and size list in memory
    rprint(
        f"Read {len(adult_content)} adult content lines with size {sum([len(x) for x in adult_content])/1_000_000:.2f} MB"
    )

    to_keep_non_adult = lines_to_keep // len(non_adult_content_files)
    if not to_keep_non_adult:
        to_keep_non_adult = 1

    # Read non-adult content files
    for file in track(
        non_adult_content_files,
        description=f"Reading non-adult content files (keeping {to_keep_non_adult} lines x file)",
    ):
        non_adult_content += random_line_lazy(
            file, to_keep_non_adult
        )  # non adult content is 18x larger than adult content

    rprint(
        f"Read {len(non_adult_content)} non-adult content lines with size {sum([len(x) for x in non_adult_content])/1_000_000:.2f} MB"
    )

    # Shuffle the data
    random.shuffle(adult_content)
    random.shuffle(non_adult_content)

    # Keep only the first 100k lines
    adult_content = adult_content[:lines_to_keep]
    non_adult_content = non_adult_content[:lines_to_keep]

    # Create the DataFrame
    df = pd.DataFrame(
        {
            "text": adult_content + non_adult_content,
            "label": [1] * len(adult_content) + [0] * len(non_adult_content),
        }
    )

    rprint(f"Created DataFrame with {len(df)} rows")
    return df


def save_df(df, output_path: Path, name: str):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    df_file = output_path / f"df_{name}.joblib"
    joblib.dump(df, df_file)
    rprint(f"DataFrame saved to {df_file}")


def load_all_files(input_dir: str, language: str) -> Tuple[List[str], List[str]]:
    adult_files = []
    non_adult_files = []
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

    return adult_files, non_adult_files


def generate_text_data(
    input_dir: str, output_path: str, language: str, should_save=True
) -> pd.DataFrame:
    adult_files, non_adult_files = load_all_files(input_dir, language)

    rprint(
        f"Found {len(adult_files)} adult files and {len(non_adult_files)} non-adult files.\nNow creating df"
    )

    df = create_dataframe_from_docs(adult_files, non_adult_files)
    if should_save:
        save_df(df, output_path, f"text_{language}")

    return df


def load_text_data(input_dir: str, output_path: str, language: str) -> pd.DataFrame:
    df_file = Path(output_path) / f"df_text_{language}.joblib"
    if df_file.exists():
        df = joblib.load(df_file)
        rprint(f"Loaded DataFrame from {df_file} with {len(df)} lines ")
    else:
        df = generate_text_data(input_dir, output_path, language)

    return df
