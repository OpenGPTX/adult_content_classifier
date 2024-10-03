from pathlib import Path

import click
from rich import print as rprint
from rich.traceback import install
from sklearn.model_selection import train_test_split

from adult_content_classifier.bow import (
    evaluate_model,
    load_vectorizer_and_data,
    save_model,
    train_model,
)
from adult_content_classifier.roberta_classifier import TextClassifier

install(show_locals=True)

INPUT_DIR = (
    "/data/horse/ws/s6690609-gptx_traindata/raw_data/cc/cc_wet_dumps_converted_dt/"
)
OUTPUT_PATH = "/data/horse/ws/s6690609-gptx_traindata/anirban/adult_content_classifier/artifacts"
LANGUAGE = ["en","de","fr", "it", "es"]


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
@click.option("--model_name", default="xlm-roberta-large", help="Hugging Face model to use for the transformer-based classifier.")
def run_transformer_classifier(input_dir, output_path, language, model_name):
    """
    Runs the Hugging Face transformer-based model training pipeline.
    """
    rprint(f"[bold]Running Transformer-based Classifier...[/bold]")
    rprint(f"[bold]Input directory:[/bold] {input_dir}")
    rprint(f"[bold]Output directory:[/bold] {output_path}")
    rprint(f"[bold]Language:[/bold] {language}")
    rprint(f"[bold]Model Name:[/bold] {model_name}")

    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # Initialize the Hugging Face TextClassifier
    classifier = TextClassifier(
        model_name=model_name,
        input_dir=input_dir,
        output_dir=output_path,
        language=language
    )

    # Run the Hugging Face training pipeline
    classifier.run()

    rprint("[bold green]Transformer-based Classifier run successfully![/bold green]")
