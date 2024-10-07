from pathlib import Path
import click
import joblib
from sklearn.model_selection import train_test_split

from adult_content_classifier.bow import (
    load_vectorizer_and_data,
    evaluate_model,
    save_model,
    train_model,
)
from rich import print as rprint
from rich.traceback import install

from adult_content_classifier.bow.data import generate_vectorized_data
from adult_content_classifier.data import create_dataframe_from_docs, load_all_files

install(show_locals=True)

INPUT_DIR = (
    "/data/horse/ws/s6690609-gptx_traindata/raw_data/cc/cc_wet_dumps_converted_dt/"
)
OUTPUT_PATH = "/data/horse/ws/s6690609-gptx_traindata/brandizzi/adult_content_classifier/artifacts"
LANGUAGE = "it"
RATIO_ADULT_NON_ADULT = 0.0055555556 # 1/180

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
@click.option("--language", default=LANGUAGE, help="Language of the text data.")
def train_bow(input_dir, output_path, language):
    rprint(f"Input dir: {input_dir}")
    rprint(f"Output path: {output_path}")
    rprint(f"Language: {language}")

    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    X, y, _ = load_vectorizer_and_data(input_dir, output_path, language)

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, output_path, language)


@click.command()
@click.option(
    "--model_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help="Path to the model file.",
)
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help="Path to the data file.",
    default=INPUT_DIR,
)
@click.option("--language", default=LANGUAGE, help="Language of the text data.")
def eval_bow(model_dir, data_dir, language):

    adult_files, non_adult_files=load_all_files(data_dir, language)

    MAX_FILES=300
    max_adult_files=MAX_FILES
    max_non_adult_files=int(MAX_FILES*RATIO_ADULT_NON_ADULT)

    rprint(f"Using {max_adult_files} adult files and {max_non_adult_files} non-adult files")
    adult_files=adult_files[:MAX_FILES]
    non_adult_files=non_adult_files[:MAX_FILES]

    df=create_dataframe_from_docs(adult_files, non_adult_files, lines_to_keep=2000)

    y=df["label"]
    X, _ =generate_vectorized_data(df=df, language=language, output_path=None)


    # find the model and data
    model_path = Path(model_dir)
    model_path = model_path / f"model_{language}.joblib"
    model= joblib.load(model_path)

    evaluate_model(model, X, y)


if __name__ == "__main__":

    eval_bow()