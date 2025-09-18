from preprocessing import load_dataset

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer


if __name__ == "__main__":
    dataset = load_dataset("data/dataset.csv")
    if dataset is None:
        raise RuntimeError("No se pudo cargar el dataset procesado.")
    
    print(f"Forma: {dataset.shape}")
    # print(dataset.head(5))
    print("Etiquetas únicas:", dataset["label"].unique().tolist())
    print(dataset["label"].value_counts())
    
    plt.figure(figsize=(5,4))
    ax = sns.countplot(x="label", data=dataset, palette=["#fbc02d", "#e53935"], hue="label", dodge=False)
    plt.xticks([0,1], ["Spam (0)","Scam (1)"])
    plt.title("Distribución de etiquetas")
    plt.xlabel("Clase")
    plt.ylabel("Número de mensajes")
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, padding=3)
    plt.tight_layout()
    plt.show()