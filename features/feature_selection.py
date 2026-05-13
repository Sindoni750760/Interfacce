"""
feature_selection.py – Selezione feature tramite ANOVA e riduzione dimensionale
                        con PCA, come descritto nel paper.
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

def anova_selection(X, y, p_threshold=0.05):
    """
    Seleziona le feature con p-value < soglia usando ANOVA (f_classif).

    Parametri:
        X : np.ndarray (n_samples, n_features)
        y : np.ndarray (n_samples,) – etichette intere
        p_threshold : float – soglia di significatività (default 0.05)

    Restituisce:
        X_selected : np.ndarray – solo le colonne selezionate
        selected_indices : list – indici originali delle feature selezionate
    """
    # f_classif restituisce F-value e p-value
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    pvalues = selector.pvalues_
    # Seleziona feature con p < p_threshold
    selected_mask = pvalues < p_threshold
    selected_indices = np.where(selected_mask)[0].tolist()
    X_selected = X[:, selected_mask]
    return X_selected, selected_indices

def apply_pca(X, variance_ratio=0.95):
    """
    Applica PCA conservando una data percentuale di varianza spiegata.

    Parametri:
        X : np.ndarray – feature di input
        variance_ratio : float – varianza minima da conservare (default 0.95)

    Restituisce:
        X_pca : np.ndarray – dati proiettati sulle componenti principali
        pca : PCA object – l'oggetto PCA addestrato
    """
    pca = PCA(n_components=variance_ratio, svd_solver='full')
    X_pca = pca.fit_transform(X)
    return X_pca, pca