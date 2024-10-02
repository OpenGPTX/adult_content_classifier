from pathlib import Path
import joblib
import pandas as pd
from rich import print as rprint
import joblib
from adult_content_classifier.data import load_text_data, save_df
from rich import print as rprint
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_data(df: pd.DataFrame):
    rprint("Vectorizing the data")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        # stop_words='english',
        ngram_range=(1, 3),
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


def load_vectorizer_and_data(input_dir: str, output_path: str, language: str):
    df = load_text_data(input_dir, output_path, language)
    df_vector, vectorizer = load_vectorized_data(df, output_path, language)

    return df_vector, df["label"], vectorizer
