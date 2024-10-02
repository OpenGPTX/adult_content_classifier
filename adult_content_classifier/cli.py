from pathlib import Path
import click
from sklearn.model_selection import train_test_split

from adult_content_classifier.data import (
    load_data,
)
from adult_content_classifier.bow.model import evaluate_model, save_model, train_model
from rich import print as rprint

INPUT_DIR = "/data/horse/ws/s6690609-gptx_traindata/raw_data/cc/cc_wet_dumps_converted_dt/"
OUTPUT_PATH = "/data/horse/ws/s6690609-gptx_traindata/brandizzi/adult_content_classifier/artifacts"
LANGUAGE = "it"

from rich.traceback import install
install(show_locals=True)


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
def main(input_dir, output_path, language):
    rprint(f"Input dir: {input_dir}")
    rprint(f"Output path: {output_path}")
    rprint(f"Language: {language}")

    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    X, y, vectorizer = load_data(input_dir, output_path,language)

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, output_path, language)


if __name__ == "__main__":
    main()
