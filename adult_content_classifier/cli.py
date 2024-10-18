#!/usr/bin/env python3
from pathlib import Path

import click
from bow import (
    evaluate_model,
    save_model,
    train_model,
)
from rich import print as rprint
from rich.traceback import install
from roberta_classifier import TextClassifier
from sklearn.model_selection import train_test_split

install(show_locals=True)

INPUT_DIR = (
    "/data/horse/ws/s6690609-gptx_traindata/raw_data/cc/cc_wet_dumps_converted_dt/"
)
OUTPUT_PATH = "/data/horse/ws/s6690609-gptx_traindata/anirban/adult_content_classifier/artifacts"
LANGUAGE = [
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", 
    "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", 
    "ro", "sk", "sl", "sv"
]


# @click.group()
# def main():
#     """Main entry point for commands."""
#     pass



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
@click.option("--language", default=LANGUAGE, multiple=True, type=click.STRING, help="Language of the text data.")
@click.option("--model_name", default="xlm-roberta-large", help="Hugging Face model to use for the transformer-based classifier.")
def run_transformer_classifier(input_dir, output_path, language, model_name):
    """
    Runs the Hugging Face transformer-based model training pipeline.
    """
    language = list(language)
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
        languages=language
    )

    # Run the Hugging Face training pipeline
    classifier.run()

    rprint("[bold green]Transformer-based Classifier run successfully![/bold green]")

if __name__ == "__main__":
    run_transformer_classifier()