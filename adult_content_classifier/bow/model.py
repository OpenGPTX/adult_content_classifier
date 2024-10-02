import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
)
import joblib
from rich import print as rprint


# train the model
def train_model(X_train, y_train):
    # take the time
    start = time.time()
    # train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # print the time
    rprint(f"Training time: {time.time() - start}")
    return model


def evaluate_model(model, features_test_transformed, y_test):
    # evaluate the model

    start = time.time()
    # predict the test set
    y_pred = model.predict(features_test_transformed)

    # calculate the f1 score
    f1 = f1_score(y_test, y_pred)
    rprint(f"F1 score: {f1}")

    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    rprint(f"Accuracy: {accuracy}")

    # print the classification report
    rprint(classification_report(y_test, y_pred))

    # print the confusion matrix
    rprint(confusion_matrix(y_test, y_pred))

    rprint(f"Evaluation time: {time.time() - start}")
    return f1, accuracy


def save_model(model, output_path, language):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # save the model to disk
    model_file = output_path / f"model_{language}.joblib"
    joblib.dump(model, model_file)
    rprint(f"Model saved to {model_file}")
