"""
train_models.py – Addestramento e validazione dei classificatori (SVM, KNN, Random Forest)
                  con cross‑validation stratificata a 10 fold, standardizzazione interna
                  e calcolo di metriche multiple.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def _get_classifier(model_name):
    """Restituisce l'istanza del classificatore in base al nome."""
    if model_name.lower() in ['svm', 'svc']:
        return SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    elif model_name.lower() in ['knn', 'kneighbors']:
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name.lower() in ['rf', 'randomforest', 'ensemble']:
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Classificatore '{model_name}' non supportato.")

def train_evaluate_model(X, y, model_name='SVM', cv_folds=10, output_dir=None, class_names=None):
    """
    Addestra e valuta un singolo classificatore con cross‑validation stratificata.

    Parametri:
        X : np.ndarray (n_samples, n_features) – feature già selezionate
        y : np.ndarray (n_samples,) – etichette intere
        model_name : str – 'SVM', 'KNN' o 'RandomForest'
        cv_folds : int – numero di fold (default 10)
        output_dir : str o None – se fornita, salva la matrice di confusione media
        class_names : list di str – nomi delle classi per le metriche

    Restituisce:
        report : dict con:
            - 'accuracy' : accuratezza media
            - 'precision' : precision media (weighted)
            - 'recall' : recall media (weighted)
            - 'f1' : f1 medio (weighted)
            - 'confusion_matrix' : matrice di confusione cumulativa sui fold di test
    """
    clf = _get_classifier(model_name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Definiamo le metriche di scoring
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    
    # Pipeline con StandardScaler interno al fold (per evitare data leakage)
    # Utilizzeremo cross_validate con un approccio manuale per poter catturare anche
    # le confusion matrices di ogni fold e calcolarne la media.
    # In alternativa, possiamo fare un loop manuale per accumulare le confusion matrices.
    
    # Eseguiamo cross_validate per ottenere le metriche medie e anche le confusion matrices
    # Purtroppo cross_validate non restituisce le confusion matrices. Dobbiamo fare un loop manuale.
    
    # Accumulatori
    accuracies, precisions, recalls, f1s = [], [], [], []
    cms = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardizzazione
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Addestramento
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        # Metriche
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        # Matrice di confusione (in valore assoluto, non normalizzata)
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        cms.append(cm)
    
    # Medie
    report = {
        'accuracy': np.mean(accuracies),
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s),
        'confusion_matrix': np.sum(cms, axis=0)  # somma delle matrici (conteggi totali)
    }
    
    # Opzionale: salvataggio su file
    if output_dir:
        np.savez(f"{output_dir}/confusion_matrix_{model_name}.npz", cm=report['confusion_matrix'])
    
    return report

def train_evaluate_all_models(X, y, cv_folds=10, verbose=False):
    """
    Addestra e valuta SVM, KNN, Random Forest e restituisce un dizionario riassuntivo.

    Restituisce:
        results : dict di dict – chiave: nome modello, valore: report con metriche
    """
    models = ['SVM', 'KNN', 'RandomForest']
    results = {}
    for m in models:
        if verbose:
            print(f"  Addestramento {m}...")
        report = train_evaluate_model(X, y, model_name=m, cv_folds=cv_folds)
        results[m] = report
    return results