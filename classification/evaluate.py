"""
evaluate.py – Funzioni di valutazione avanzate: stampa report dettagliato e
              visualizzazione delle matrici di confusione.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def print_classification_report(report, class_names=None):
    """
    Stampa a schermo le metriche contenute in un report (da train_evaluate_model).

    Parametri:
        report : dict – con 'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'
        class_names : list di str – opzionali, nomi delle classi
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
        # Stampa formattata
        header = "          " + " ".join([f"{name:>8}" for name in class_names])
        print(header)
        for i, row in enumerate(cm):
            print(f"{class_names[i]:>8}  " + " ".join([f"{int(val):8d}" for val in row]))
        print()
    else:
        print("Matrice di confusione (array):")
        print(report['confusion_matrix'])

def plot_confusion_matrices(confusion_matrix, class_names, title="Confusion Matrix", cmap="Blues"):
    """
    Visualizza una o più matrici di confusione. Se viene passata una singola matrice,
    la mostra; se una lista di matrici, mostra un subplot per ognuna (es. per confronto modelli).
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
    plt.show()