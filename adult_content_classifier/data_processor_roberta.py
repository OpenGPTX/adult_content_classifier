import json
import os
import random
from functools import partial
from itertools import chain
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from datasets import Dataset
from joblib import Parallel, delayed
from loguru import logger
from rich import print as rprint
from rich.progress import track
from sklearn.model_selection import train_test_split

YEARS = [  # dumps with filtered dir
    "2024-22",
    "2014-41",
    "2019-26",
    "2023-50",
]

SAMPLE_SIZE = 10000
READING_SAMPLE_SIZE  = 1000
N_JOBS = 50
TEST_SPLIT_SIZE = 0.05

def get_adult_files(input_path: str, language: str) -> List[str]:
    adult_content_path = Path(input_path) / "filtered"

    # logger.info(f"Looking for adult files in {adult_content_path}")
 

    files = [
        f for f in os.listdir(adult_content_path) if any(f.endswith(f"{lang}.jsonl") for lang in language)
    ]

    # filter files that start with 2 since this is the indicator of adult content
    files = [f for f in files if f.startswith("2")]

    # filter the files that contain the language in the name
    files = [f for f in files if language in f]

    # convert to Path
    files = [Path(adult_content_path) / f for f in files]

    # logger.info(f"Found {len(files)} adult files")

    # shuffle the files
    # random.shuffle(files)
    
    # if len(files) > SAMPLE_SIZE:
    #     files = files[:SAMPLE_SIZE]

    return files


def get_non_adult_files(input_path: str, language: str) -> List[str]:
    non_adult_content_path = Path(input_path) / language
    # logger.info(f"Looking for non-adult files in {non_adult_content_path}")

    files = [f for f in os.listdir(non_adult_content_path) if f.endswith("jsonl")]

    # convert to Path
    files = [Path(non_adult_content_path) / f for f in files]

    # logger.info(f"Found {len(files)} non-adult files")

    # shuffle the files
    # random.shuffle(files)
    
    # if len(files) > SAMPLE_SIZE:
    #     files = files[:SAMPLE_SIZE]

    return files


def random_line_lazy(file_path: Path, read_all: bool) -> List[str]:
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
            # avoid reading full file
            if not read_all and line_number >= READING_SAMPLE_SIZE:
                return [selected_lines[random.randint(0, line_number-2)]]
            selected_lines.append(json.loads(line)["text"])
                
            # else:
            #     r = random.randint(0, line_number)
            #     if r < to_keep:
            #         selected_lines[r] = json.loads(line)["text"]

    return selected_lines



# def process_file(file, to_keep_adult):
#     return random_line_lazy(file, to_keep_adult)

def save_df(df, output_path: Path, name: str):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    df_file = output_path / f"df_{name}.joblib"
    joblib.dump(df, df_file)
    logger.info(f"DataFrame saved to {df_file}")


def get_documents(files):
    if len(files) ==  SAMPLE_SIZE:
        documents = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(random_line_lazy)(file, False) for file in files)
        documents = list(chain.from_iterable(documents))
    else: 
        documents = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(random_line_lazy)(file, True) for file in files)
        documents = list(chain.from_iterable(documents))
    
    if len(documents) > SAMPLE_SIZE:
        random.shuffle(documents)
        documents = documents[0: SAMPLE_SIZE]
    
    return documents
        


def load_documents_all_languages(input_dir: str, languages: List[str]) -> Tuple[List[str], List[str]]:
    adult_documents_all = [] # For all languages
    non_adult_documents_all = []
    languages = [
        "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", 
        "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", 
        "ro", "sk", "sl", "sv"
    ]

    for language in languages:
        adult_files_per_language = []
        non_adult_files_per_language = []
        for dump in YEARS:   
            # logger.info(f"Getting files for {language}") 
            new_input_dir = f"{input_dir}/{dump}"

            # adult files per language
            adult_files_per_language.extend(get_adult_files(new_input_dir, language))
            
            non_adult_files_per_language.extend(get_non_adult_files(new_input_dir, language))

        if not adult_files_per_language:
            logger.info(f"No adult files found in for langugage: {language}")
            continue

        if not non_adult_files_per_language:
            logger.info(f"No non-adult files found for langugage: {language}")
            continue
        
        random.shuffle(adult_files_per_language)
        random.shuffle(non_adult_files_per_language)
        
        if len(adult_files_per_language) > SAMPLE_SIZE:
            adult_files_per_language = adult_files_per_language[:SAMPLE_SIZE]
        if len(non_adult_files_per_language) > SAMPLE_SIZE:
            non_adult_files_per_language = non_adult_files_per_language[:SAMPLE_SIZE]
        
        
        adult_documents_per_language = get_documents(adult_files_per_language)
        logger.info(f"Got {len(adult_documents_per_language)} number of adult documents for langugage: {language}")
        
        non_adult_documents_per_language = get_documents(non_adult_files_per_language)
        # assuring no of non-adult docs per language is same as the no of adult docs per language
        if len(non_adult_documents_per_language) > len(adult_documents_per_language):
            non_adult_documents_per_language = non_adult_documents_per_language[0: len(adult_documents_per_language)]
            
        logger.info(f"Got {len(non_adult_documents_per_language)} number of non adult documents for langugage: {language}")
        
    
        adult_documents_all.extend(adult_documents_per_language)
        non_adult_documents_all.extend(non_adult_documents_per_language)

    # shuffle the files
    random.shuffle(adult_documents_all)
    random.shuffle(non_adult_documents_all)

    return adult_documents_all, non_adult_documents_all


def create_dataframe_from_docs(
    adult_documents: List[str],
    non_adult_documents: List[str],
) -> pd.DataFrame:
    """
    Returns a DataFrame containing adult and non adult data.

    Args:
        adult_content_files: List of file paths containing adult content.
        non_adult_content_files: List of file paths containing non-adult content.
        to_keep: Number of lines to randomly select from each file.

    Returns:
        A DataFrame containing the sampled text lines and their corresponding labels.
    """
    # Create the DataFrame
    df = pd.DataFrame(
        {
            "text": adult_documents + non_adult_documents,
            "label": [1] * len(adult_documents) + [0] * len(non_adult_documents),
        }
    )

    logger.info(f"Created DataFrame with {len(df)} rows")
    return df


def generate_text_data(
    input_dir: str, output_path: str, languages: List[str], should_save=True
) -> pd.DataFrame:
    
    
    name = "eu24"
    adult_documents, non_adult_documents = load_documents_all_languages(input_dir, languages)
    
    df = create_dataframe_from_docs(adult_documents, non_adult_documents)
    if should_save:
        save_df(df, output_path, f"text_{name}")

    return df


def load_text_data(input_dir: str, output_path: str, language: List[str]) -> pd.DataFrame:
    

    name = "eu24"
    df_file = Path(output_path) / f"df_text_{name}.joblib"
    if df_file.exists():
        df = joblib.load(df_file)
        logger.info(f"Loaded DataFrame from {df_file} with {len(df)} lines ")
    else:
        df = generate_text_data(input_dir, output_path, language)

    train_df, val_df = train_test_split(df, test_size=TEST_SPLIT_SIZE, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    return train_dataset, val_dataset