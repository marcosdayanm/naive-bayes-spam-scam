import re
import string
import pandas as pd

# se compilan expresiones regulares para que sea más eficientes a la hora de limpiar el texto
_URL = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
_EMAIL = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', flags=re.IGNORECASE)
_PHONE = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,3}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b')
_MONEY = re.compile(r'([$€£₹]\s?\d[\d,\.]*|\b\d[\d,\.]*\s?(?:usd|mxn|eur|dólares?|pesos?)\b)', flags=re.IGNORECASE)
_NUM = re.compile(r'\b\d[\d,\.]*\b')
_EMOJI = re.compile(r'[\U00010000-\U0010FFFF]', flags=re.UNICODE)
_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def load_dataset(path, encoding="utf-8", sep=","):
    try:
        df = pd.read_csv(path, header=None, sep=sep, encoding=encoding)
        if df.shape[1] < 2:
            df = pd.read_csv(path, header=None, sep=";", encoding=encoding)
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].copy()
            df.columns = ["label", "raw_text"]
            return df
    except Exception as e:
        raise RuntimeError(f"No se pudo leer el CSV con los encoders {encoding}. Error: {e}")
    

def store_dataset(df: pd.DataFrame, path, encoding="utf-8", sep=","):
    try:
        df.to_csv(path, index=False, header=False, sep=sep, encoding=encoding)
    except Exception as e:
        raise RuntimeError(f"No se pudo escribir el CSV con los encoders {encoding}. Error: {e}")


def clean_text(s: str) -> str:
    s = _EMOJI.sub(" ", s)
    s = s.translate(_PUNCT_TABLE)
    s = s.lower()

    s = _URL.sub(" [URL] ", s)
    s = _EMAIL.sub(" [EMAIL] ", s)
    s = _PHONE.sub(" [PHONE] ", s)
    s = _MONEY.sub(" [MONEY] ", s)
    s = _NUM.sub(" [NUM] ", s)

    # s = re.sub(r'(.)\1{2,}', r'\1\1', s) # palabras con caracteres largos
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    label_map = {"spam": 0, "scam": 1}
    dataset["label"] = dataset["label"].astype(str).str.strip().str.lower().map(label_map)
    if dataset["label"].isna().any():
        bad = dataset[dataset["label"].isna()].head(10)
        raise ValueError(f"Se hallaron etiquetas fuera de {{spam, scam}}. Ejemplos:\n{bad}")

    dataset["text"] = dataset["raw_text"].astype(str).apply(clean_text)
    return dataset


if __name__ == "__main__":
    dataset_path = "data/crude_data.csv"
    dataset = load_dataset(dataset_path)
    if dataset is None:
        raise RuntimeError(f"No se pudo cargar el dataset desde {dataset_path}.")

    dataset = normalize_dataset(dataset)
    store_dataset(dataset[["label","text"]], "data/dataset.csv")



