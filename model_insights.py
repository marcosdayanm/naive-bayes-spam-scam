# ==== IMPORTS EXTRA PARA INSIGHTS ====
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from preprocessing import load_dataset



def k_fold_validation(dataset: pd.DataFrame, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y = dataset["label"].to_numpy()
    n = len(dataset)
    y_true_all = np.zeros(n, dtype=int)
    y_pred_oof = np.zeros(n, dtype=int)
    y_proba_oof = np.zeros(n, dtype=float)

    metrics = {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []}

    fold = 1
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset.iloc[train_index], dataset.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train["text"])
        X_test_tfidf = vectorizer.transform(X_test["text"])

        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)
        y_proba = model.predict_proba(X_test_tfidf)[:, 1]

        # Guardar OOF
        y_true_all[test_index] = y_test
        y_pred_oof[test_index] = y_pred
        y_proba_oof[test_index] = y_proba

        # Métricas por fold
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        auc = roc_auc_score(y_test, y_proba)

        metrics["acc"].append(acc)
        metrics["prec"].append(prec)
        metrics["rec"].append(rec)
        metrics["f1"].append(f1)
        metrics["auc"].append(auc)

        print(f"fold {fold}: acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}, auc: {auc:.4f}")
        fold += 1

    avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}

    print("\nMétricas promedio con K-Fold:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}", end=", ")
    print()
    return avg_metrics, y_true_all, y_pred_oof, y_proba_oof


def plot_kfold_insights(dataset: pd.DataFrame, y_true_all, y_pred_oof, y_proba_oof):
    vec_full = TfidfVectorizer()
    X_full = vec_full.fit_transform(dataset["text"])
    y_full = dataset["label"].to_numpy()

    nb_full = MultinomialNB(alpha=1.0)
    nb_full.fit(X_full, y_full)

    coefs = nb_full.feature_log_prob_[1] - nb_full.feature_log_prob_[0]
    feature_names = np.array(vec_full.get_feature_names_out())

    # Top 20 SCAM (clase 1) -> pesos más altos
    top_idx_1 = np.argsort(coefs)[-20:]
    plt.figure(figsize=(6, 6))
    plt.barh(feature_names[top_idx_1], coefs[top_idx_1])
    plt.title("Top 20 tokens indicativos de SCAM (clase 1)")
    plt.xlabel("Peso (log-prob diff)")
    plt.tight_layout()
    plt.show()

    # Top 20 SPAM (clase 0) -> pesos más bajos
    top_idx_0 = np.argsort(coefs)[:20]
    plt.figure(figsize=(6, 6))
    plt.barh(feature_names[top_idx_0], -coefs[top_idx_0])  # magnitud positiva para visualizar
    plt.title("Top 20 tokens indicativos de SPAM (clase 0)")
    plt.xlabel("Peso (|log-prob diff|)")
    plt.tight_layout()
    plt.show()

    # ROC con OOF
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_oof)
    auc_cv = roc_auc_score(y_true_all, y_proba_oof)
    plt.figure()
    plt.plot(fpr, tpr, label=f"MultinomialNB (AUC={auc_cv:.3f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - spam vs scam (KFold OOF)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Matriz de confusión OOF
    cm = confusion_matrix(y_true_all, y_pred_oof)
    plt.figure(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Spam (0)", "Scam (1)"])
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title("Matriz de confusión OOF (MultinomialNB)")
    plt.tight_layout()
    plt.show()

    # Precision–Recall con OOF
    precision, recall, _ = precision_recall_curve(y_true_all, y_proba_oof)
    ap = average_precision_score(y_true_all, y_proba_oof)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (KFold OOF)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # resumen
    print(f"AUC OOF: {auc_cv:.4f} | AP OOF: {ap:.4f}")
    print("Tokens SCAM (top 20):", list(feature_names[top_idx_1]))
    print("Tokens SPAM/NO-SCAM (top 20):", list(feature_names[top_idx_0]))


if __name__ == "__main__":
    dataset = load_dataset("data/dataset.csv")
    if dataset is None:
        raise RuntimeError("No se pudo cargar el dataset procesado.")

    avg_metrics, y_true_all, y_pred_oof, y_proba_oof = k_fold_validation(dataset, n_splits=5, random_state=42)
    plot_kfold_insights(dataset, y_true_all, y_pred_oof, y_proba_oof)
