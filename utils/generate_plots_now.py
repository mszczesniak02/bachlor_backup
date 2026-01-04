
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data provided by user
cms_data = {
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
    "Ensemble": np.array([
        [54, 4, 0, 0],
        [2, 144, 4, 0],
        [0, 15, 223, 9],
        [0, 0, 14, 526]
    ])
}

CLASS_NAMES = ["Włosowate", "Małe", "Średnie", "Duże"]


def plot_cm(name, cm):
    # Dimensions increased slightly to accommodate large text
    plt.figure(figsize=(10, 8))

    # Try using seaborn for a nicer heatmap, fallback to basic matplotlib if needed
    try:
        # annotations font size increased
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                              xticklabels=CLASS_NAMES,
                              yticklabels=CLASS_NAMES,
                              cbar=False,  # Remove colorbar
                              square=True,  # Align tiles (make them square)
                              # Larger numbers inside cells
                              annot_kws={"size": 16})

        # Axis labels rotation and size
        heatmap.set_xticklabels(
            heatmap.get_xticklabels(), rotation=45, fontsize=14)
        heatmap.set_yticklabels(
            heatmap.get_yticklabels(), rotation=0, fontsize=14)
    except:
        # Fallback
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(CLASS_NAMES))
        plt.xticks(tick_marks, CLASS_NAMES, rotation=45, fontsize=14)
        plt.yticks(tick_marks, CLASS_NAMES, fontsize=14)

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     fontsize=16,
                     color="white" if cm[i, j] > thresh else "black")

    # Titles 2x larger (assuming base was ~14, so ~28)
    plt.title(f'Macierz Pomyłek: {name}', fontsize=28, pad=20)
    plt.ylabel('Prawdziwa Etykieta', fontsize=24)
    plt.xlabel('Przewidziana Etykieta', fontsize=24)
    plt.tight_layout()

    filename = f'confusion_matrix_{name}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Wygenerowano: {filename}")


if __name__ == "__main__":
    print("Rozpoczynam generowanie wykresów...")
    for name, data in cms_data.items():
        plot_cm(name, data)
    print("Gotowe.")
