from pathlib import Path
from typing import Any, Dict, List

import joblib

ARTIFACT_PATH = "/data/horse/ws/s6690609-gptx_traindata/anirban/adult_content_classifier/artifacts"


def load_models(artifact_paths) -> Dict[str, Any]:
    # list all the models in the artifact path
    model_files = list(artifact_paths.glob("model_*.joblib"))
    models = {}

    # load the models
    for model_file in model_files:
        model = joblib.load(model_file)
        language = model_file.stem.split("_")[1]
        models[language] = model

    return models


def load_vectorizers(artifact_paths) -> Dict[str, Any]:
    # list all the vectorizers in the artifact path
    vectorizer_files = list(artifact_paths.glob("vectorizer_*.joblib"))
    vectorizers = {}

    # load the vectorizers
    for vectorizer_file in vectorizer_files:
        vectorizer = joblib.load(vectorizer_file)
        language = vectorizer_file.stem.split("_")[1]
        vectorizers[language] = vectorizer

    return vectorizers


def load_models_and_vectorizers(artifact_paths=ARTIFACT_PATH) -> Dict[str, Any]:
    artifact_paths = Path(artifact_paths)
    models = load_models(artifact_paths)
    vectorizers = load_vectorizers(artifact_paths)
    return models, vectorizers


def classify_text(
    text: List[str], language: str, models: Dict[str, Any], vectorizers: Dict[str, Any]
):
    # get the model and vectorizer for the language
    model = models[language]
    vectorizer = vectorizers[language]

    # vectorize the text
    text_vectorized = vectorizer.transform(text)

    # classify the text
    prediction = model.predict(text_vectorized)

    return prediction[0]
