
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Data from user
CLASS_NAMES_DISPLAY = ["Włosowate", "Małe", "Średnie", "Duże"]

cms = {
    "EfficientNet": np.array([
        [53, 5, 0, 0],
        [9, 137, 4, 0],
        [0, 21, 211, 15],
        [0, 0, 14, 526]
    ]),
    "ConvNeXt": np.array([
        [54, 3, 0, 1],
        [1, 145, 4, 0],
        [0, 13, 225, 9],
        [0, 0, 14, 526]
    ]),
    "Ensemble (Soft Voting)": np.array([
        [54, 4, 0, 0],
        [2, 144, 4, 0],
        [0, 15, 223, 9],
        [0, 0, 14, 526]
    ])
}


def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_DISPLAY,
                yticklabels=CLASS_NAMES_DISPLAY)

    plt.title(f'Macierz Pomyłek - {model_name}', fontsize=16)
    plt.ylabel('Prawdziwa Etykieta', fontsize=12)
    plt.xlabel('Przewidziana Etykieta', fontsize=12)
    plt.tight_layout()

    filename = f'confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres: {filename}")


print("--- Rgenerowanie Wykresów i Raportów ---\n")

for model_name, cm in cms.items():
    # 1. Plot
    plot_confusion_matrix(cm, model_name)

    # 2. Calculate Metrics
    total = np.sum(cm)
    accuracy = np.trace(cm) / total

    precisions = []
    recalls = []
    f1s = []
    supports = []
    specificities = []

    print(f"\n### Wyniki dla modelu: {model_name}")
    print(f"**Dokładność (Accuracy): {accuracy:.4f}**")

    print(f"\n| Klasa | Precyzja | Czułość (Recall) | F1-Score | Specyficzność | Liczebność |")
    print(f"|---|---|---|---|---|---|")

    for i, class_name in enumerate(CLASS_NAMES_DISPLAY):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        support = np.sum(cm[i, :])

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        specificities.append(specificity)

        print(
            f"| {class_name} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {specificity:.4f} | {support} |")

    avg_precision = np.average(precisions, weights=supports)
    avg_recall = np.average(recalls, weights=supports)
    avg_f1 = np.average(f1s, weights=supports)

    print(
        f"| **Średnia ważona** | **{avg_precision:.4f}** | **{avg_recall:.4f}** | **{avg_f1:.4f}** | - | {total} |")
