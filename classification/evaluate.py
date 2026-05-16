"""
evaluate.py – Valutazione e visualizzazione matrici di confusione (solo matplotlib).
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def print_classification_report(report, class_names=None):
    """Stampa metriche di classificazione."""
    print("="*50)
    print("REPORT DI CLASSIFICAZIONE")
    print("="*50)
    print(f"Accuracy  : {report['accuracy']:.4f}")
    print(f"Precision : {report['precision']:.4f}")
    print(f"Recall    : {report['recall']:.4f}")
    print(f"F1-score  : {report['f1']:.4f}")
    print()
    if class_names:
        print("Matrice di confusione (conteggi):")
        cm = report['confusion_matrix']
        header = "          " + " ".join([f"{name:>8}" for name in class_names])
        print(header)
        for i, row in enumerate(cm):
            print(f"{class_names[i]:>8}  " + " ".join([f"{int(val):8d}" for val in row]))
        print()
    else:
        print("Matrice di confusione (array):")
        print(report['confusion_matrix'])


def plot_confusion_matrices(confusion_matrix, class_names, title="Confusion Matrix",
                            cmap="Blues", output_dir=None):
    """
    Visualizza una matrice di confusione usando matplotlib (senza seaborn).
    """
    plt.figure(figsize=(6,5))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Annotazioni con i valori
    thresh = confusion_matrix.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = title.lower().replace(" ", "_") + ".png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        print(f"Matrice di confusione salvata in {filepath}")

    plt.show()