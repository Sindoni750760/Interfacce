#!/usr/bin/env python3
"""
main.py – Pipeline completa per la classificazione di stati psicofisiologici da ECG,
          basata sul paper "Detection and analysis driver state with electrocardiogram"
          e applicata al dataset WESAD.

Utilizzo:
    python main.py --task binary --classifier all
    python main.py --task multiclass --classifier rf --use_pca --output ./risultati
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Import dei moduli del progetto (verranno creati nei passi successivi)
from preprocessing.ecg_cleaning import butter_bandpass_filter, reconstruct_peaks
from preprocessing.r_peak_detection import detect_r_peaks, compute_rr_intervals
from features.extract_features import extract_all_features
from features.feature_selection import anova_selection, apply_pca
from classification.train_models import train_evaluate_model, train_evaluate_all_models
from classification.evaluate import plot_confusion_matrices, print_classification_report
from utils.visualization import plot_raw_vs_filtered, plot_rr_intervals

# ============================================
# 1. Caricamento del dataset WESAD
# ============================================
def load_wesad_data(wesad_root, window_sec=60, target_fs=256):
    """
    Carica i dati WESAD e restituisce finestre ECG etichettate.
    
    Parametri:
        wesad_root : str o Path
            Cartella radice contenente i file .pkl dei soggetti (es. S2.pkl, S3.pkl, ...).
        window_sec : int
            Durata in secondi di ogni finestra di analisi (default 60).
        target_fs : int
            Frequenza di campionamento target (256 Hz, come da paper).
    
    Restituisce:
        X : np.ndarray, shape (n_windows, window_sec*target_fs)
            Finestre di segnale ECG filtrato (sarà fatto nel preprocessing).
        y : np.ndarray, shape (n_windows,)
            Etichette corrispondenti (0=baseline, 1=stress, 2=amusement per task multiclass).
        label_map : dict
            Mappatura etichette numeriche -> nome condizione.
    """
    X_windows = []
    y_windows = []
    
    for subject_file in sorted(Path(wesad_root).glob('S*.pkl')):
        data = pd.read_pickle(subject_file)
        ecg_raw = data['signal']['chest']['ECG'].values.flatten()  # array 1D
        labels_raw = data['label']  # già allineato ai campioni
        
        # Ricampionamento a 256 Hz (se necessario)
        original_fs = 700  # RespiBAN ECG originale è 700 Hz
        if original_fs != target_fs:
            # Ricampiona usando interpolazione lineare (per semplicità)
            ratio = target_fs / original_fs
            new_len = int(len(ecg_raw) * ratio)
            ecg_raw = np.interp(np.linspace(0, len(ecg_raw)-1, new_len),
                                np.arange(len(ecg_raw)), ecg_raw)
            
            # Anche le etichette vanno ricampionate (nearest neighbor)
            labels_raw = np.interp(np.linspace(0, len(labels_raw)-1, new_len),
                                   np.arange(len(labels_raw)), labels_raw)
            labels_raw = np.round(labels_raw).astype(int)
        
        # Suddivisione in finestre non sovrapposte
        window_samples = window_sec * target_fs
        n_windows_subj = len(ecg_raw) // window_samples
        
        for i in range(n_windows_subj):
            start = i * window_samples
            end = start + window_samples
            window = ecg_raw[start:end]
            # Etichetta di maggioranza all'interno della finestra
            window_labels = labels_raw[start:end]
            label = np.bincount(window_labels).argmax()
            X_windows.append(window)
            y_windows.append(label)
    
    X = np.array(X_windows)
    y = np.array(y_windows)
    
    # Mappa etichette: per WESAD, 1=baseline, 2=stress, 3=amusement
    label_map = {1: 'baseline', 2: 'stress', 3: 'amusement'}
    
    return X, y, label_map

# ============================================
# 2. Preprocessing di tutte le finestre
# ============================================
def preprocess_windows(X, fs=256):
    """
    Per ogni finestra ECG: ricostruzione dei picchi R, filtraggio Butterworth,
    calcolo degli intervalli RR. Restituisce le finestre filtrate e gli RR.
    """
    X_filtered = []
    rr_intervals_list = []
    
    for ecg_window in X:
        # Ricostruzione picchi (simulata: nel paper si usa un algoritmo ad-hoc,
        # qui usiamo Pan-Tompkins da neurokit2 all'interno di detect_r_peaks)
        ecg_reconstructed = reconstruct_peaks(ecg_window, fs)
        
        # Filtro Butterworth passa-banda
        ecg_filtered = butter_bandpass_filter(ecg_reconstructed, lowcut=0.5,
                                              highcut=49.0, fs=fs, order=6)
        X_filtered.append(ecg_filtered)
        
        # Rilevamento picchi R e calcolo intervalli RR
        r_peaks = detect_r_peaks(ecg_filtered, fs)
        rr = compute_rr_intervals(r_peaks, fs)
        rr_intervals_list.append(rr)
    
    return np.array(X_filtered), rr_intervals_list

# ============================================
# 3. Pipeline principale
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Pipeline ECG per classificazione stati psicofisiologici')
    parser.add_argument('--data_path', type=str, default='./data/WESAD',
                        help='Percorso alla directory contenente i file .pkl di WESAD')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary',
                        help='Tipo di classificazione: binary (stress vs baseline) o multiclass (3 classi)')
    parser.add_argument('--classifier', type=str, choices=['svm', 'knn', 'rf', 'all'], default='all',
                        help='Classificatore da utilizzare')
    parser.add_argument('--use_pca', action='store_true',
                        help='Applica PCA dopo la selezione ANOVA')
    parser.add_argument('--output', type=str, default='./results',
                        help='Directory dove salvare i risultati')
    parser.add_argument('--window_sec', type=int, default=60,
                        help='Durata finestra in secondi')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostra grafici esplicativi')
    args = parser.parse_args()
    
    # Crea directory output se non esiste
    os.makedirs(args.output, exist_ok=True)
    
    # ----------------------------------------------
    # Fase 0: Caricamento e segmentazione
    # ----------------------------------------------
    print("Caricamento dataset WESAD...")
    X_raw, y_raw, label_map = load_wesad_data(args.data_path, window_sec=args.window_sec)
    print(f"Caricate {len(X_raw)} finestre da {len(os.listdir(args.data_path))} soggetti.")
    
    # Adatta il task binario: stress vs baseline (escludi amusement)
    if args.task == 'binary':
        mask = np.isin(y_raw, [1, 2])  # 1=baseline, 2=stress
        X_raw = X_raw[mask]
        y_raw = y_raw[mask]
        # Rinomina etichette: baseline=0, stress=1
        y = (y_raw - 1).astype(int)   # 1->0, 2->1
        class_names = ['baseline', 'stress']
    else:
        # Multiclass: tutte e tre le condizioni
        y = y_raw - 1  # 1->0, 2->1, 3->2
        class_names = ['baseline', 'stress', 'amusement']
    
    # ----------------------------------------------
    # Fase 1: Preprocessing (ricostruzione + filtraggio + RR)
    # ----------------------------------------------
    print("Preprocessing delle finestre ECG...")
    X_filtered, rr_list = preprocess_windows(X_raw)
    
    if args.verbose:
        # Grafico esempio: primo soggetto, prima finestra
        plot_raw_vs_filtered(X_raw[0], X_filtered[0])
        plot_rr_intervals(rr_list[0])
    
    # ----------------------------------------------
    # Fase 2: Estrazione feature (14 lineari + 3 non lineari)
    # ----------------------------------------------
    print("Estrazione feature...")
    features_df = extract_all_features(X_filtered, rr_list)
    feature_names = features_df.columns.tolist()
    X_features = features_df.values
    
    # ----------------------------------------------
    # Fase 3: Selezione feature (ANOVA + eventuale PCA)
    # ----------------------------------------------
    print("Selezione feature con ANOVA (p<0.05)...")
    X_selected, selected_indices = anova_selection(X_features, y, p_threshold=0.05)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    print(f"Feature selezionate: {selected_feature_names}")
    
    if args.use_pca:
        print("Applicazione PCA per ridurre dimensionalità...")
        X_selected, pca_obj = apply_pca(X_selected, variance_ratio=0.95)
        print(f"Nuova dimensionalità dopo PCA: {X_selected.shape[1]}")
    
    # ----------------------------------------------
    # Fase 4: Classificazione e valutazione
    # ----------------------------------------------
    print("Addestramento e validazione dei modelli...")
    
    if args.classifier == 'all':
        # Confronta tutti e tre i classificatori
        results = train_evaluate_all_models(X_selected, y, cv_folds=10,
                                           verbose=args.verbose)
        # Salva i risultati in un CSV
        results_df = pd.DataFrame(results).T
        results_df.to_csv(os.path.join(args.output, 'model_comparison.csv'))
        print("\nRisultati confronto modelli:")
        print(results_df)
        
        # Grafici di confronto (opzionali)
        if args.verbose:
            from utils.visualization import plot_model_comparison
            plot_model_comparison(results_df, metric='accuracy')
    
    else:
        # Singolo classificatore
        model_map = {'svm': 'SVM', 'knn': 'KNN', 'rf': 'RandomForest'}
        model_name = model_map[args.classifier]
        report = train_evaluate_model(X_selected, y, model_name=model_name,
                                      cv_folds=10, output_dir=args.output,
                                      class_names=class_names)
        print(f"\nReport per {model_name}:")
        print_classification_report(report, class_names)
        if args.verbose:
            plot_confusion_matrices(report['confusion_matrix'], class_names)
    
    print(f"\nPipeline completata. I risultati sono stati salvati in '{args.output}'.")

# ============================================
if __name__ == '__main__':
    main()
