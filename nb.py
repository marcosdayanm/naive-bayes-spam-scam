import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score

from preprocessing import load_dataset, clean_text

def train_nb(train: pd.DataFrame) -> tuple[TfidfVectorizer, MultinomialNB]:
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(train["text"])
    y = train["label"].to_numpy()
    model = MultinomialNB()
    model.fit(X_tfidf, y)
    return vectorizer, model


def save_model_and_vectorizer(model, vectorizer, model_path="bin/nb.pkl", vectorizer_path="bin/tfidf_vectorizer.pkl"):
    """
    Guarda el modelo y el vectorizador en archivos usando pickle.
    """
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_model_and_vectorizer(model_path="bin/nb.pkl", vectorizer_path="bin/tfidf_vectorizer.pkl") -> tuple[MultinomialNB | None, TfidfVectorizer | None]:
    """
    Carga el modelo y el vectorizador desde archivos usando pickle.
    Devuelve None si no existen los archivos.
    """
    model = None
    vectorizer = None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError("No se pudo cargar el modelo. Asegúrate de que el archivo exista.")
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError("No se pudo cargar el vectorizador. Asegúrate de que el archivo exista.")
    return model, vectorizer


def predict(model: MultinomialNB, vectorizer: TfidfVectorizer, text: str) -> int:
    processed_text = clean_text(text)
    X_tfidf = vectorizer.transform([processed_text])
    return model.predict(X_tfidf).tolist()[0]


if __name__ == "__main__":
    train = load_dataset("data/train.csv")
    if train is None:
        raise RuntimeError("No se pudo cargar el dataset de entrenamiento.")

    vectorizer, model = train_nb(train)
    print("Naive Bayes entrenado con data/train.csv")

    save_model_and_vectorizer(model, vectorizer)
