"""
evaluate.py – Funzioni di valutazione avanzate: stampa report dettagliato e
              visualizzazione delle matrici di confusione.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def print_classification_report(report, class_names=None):
    """
    Stampa a schermo le metriche contenute in un report (da train_evaluate_model).
    """
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
    Visualizza una o più matrici di confusione. Se viene passata una singola matrice,
    la mostra; se una lista di matrici, mostra un subplot per ognuna.
    Ora supporta il salvataggio opzionale in `output_dir`.
    """
    if isinstance(confusion_matrix, list):
        n = len(confusion_matrix)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
        if n == 1:
            axes = [axes]
        for ax, cm, t in zip(axes, confusion_matrix, [title]*n):
            sns.heatmap(cm, annot=True, fmt='g', cmap=cmap, ax=ax,
                        xticklabels=class_names, yticklabels=class_names)
            ax.set_title(t)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        plt.tight_layout()
    else:
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Crea un nome file pulito a partire dal titolo
        filename = title.lower().replace(" ", "_") + ".png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        print(f"Matrice di confusione salvata in {filepath}")

    plt.show()