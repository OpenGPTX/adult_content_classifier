from pathlib import Path
import time

import click
import joblib
from adult_content_classifier.bow.filtering_functions import load_models_and_vectorizers
from adult_content_classifier.cli import INPUT_DIR, OUTPUT_PATH
from adult_content_classifier.data import load_all_files, create_dataframe_from_docs
from sklearn.metrics import classification_report
from rich import print as rprint


@click.command()
@click.option(
    "--input_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True
    ),
    help="Path to the input directory.",
    default=INPUT_DIR,
)
@click.option(
    "--output_path",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    help="Path to the output directory.",
    default=OUTPUT_PATH,
)
@click.option("--language", help="Language of the text data.")
def profile(input_dir: str, output_path: str, language: str):
    time_start = time.time()
    models, vectorizers = load_models_and_vectorizers()

    model = models[language]
    vectorizer = vectorizers[language]

    time_load = time.time()

    rprint(f"Time to load models and vectorizers: {time_load-time_start}")

    adult_files, non_adult_files = load_all_files(input_dir, language)

    # keep only 100 files
    adult_files = adult_files[:5000]
    non_adult_files = non_adult_files[:100]
    to_keep = 5000

    df = create_dataframe_from_docs(adult_files, non_adult_files, lines_to_keep=to_keep)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Keep first 100k rows
    df = df.head(100000)

    time_data_generation = time.time()

    rprint(f"Time to generate data: {time_data_generation-time_load}")
    rprint(f"Data shape: {df.shape}")
    rprint(f"Size of the vectorized data: {df.memory_usage().sum()/1024/1024} MB")
    rprint(f"Total number of characters: {df['text'].apply(len).sum()}")

    # Vectorize data

    X = vectorizer.transform(df["text"])
    y = df["label"]

    time_vectorization = time.time()

    rprint(f"Time to vectorize data: {time_vectorization-time_data_generation}")
    rprint(f"Data shape after vectorization: {X.shape}")

    # print the size in MB of the vectorized data
    rprint(f"Size of the vectorized data: {X.data.nbytes/1024/1024} MB")

    # Predict

    y_pred = model.predict(X)

    time_prediction = time.time()

    rprint(f"Time to predict: {time_prediction-time_vectorization}")

    # Evaluate

    rprint(classification_report(y, y_pred))

    time_end = time.time()

    rprint(f"Total time: {time_end-time_start}")


def check_data_sizes(output_path):
    languages = ["it", "en", "fr", "es", "de"]

    def print_data_info(df):
        # print the size in MB of the data
        rprint(f"Size of the  data: {df.memory_usage().sum()/1024/1024} MB")
        # print the total number of characters
        rprint(f"Total number of characters: {df['text'].apply(len).sum()}")
        # print the shape of the data
        rprint(f"Data shape: {df.shape}")

    for lang in languages:
        df_file = Path(output_path) / f"df_text_{lang}.joblib"
        if df_file.exists():
            df = joblib.load(df_file)
            print_data_info(df)

        else:
            rprint(f"Dataframe for {lang} does not exist")
