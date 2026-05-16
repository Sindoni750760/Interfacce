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
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Import dei moduli del progetto
from preprocessing.ecg_cleaning import butter_bandpass_filter, reconstruct_peaks
from preprocessing.r_peak_detection import detect_r_peaks, compute_rr_intervals
from features.extract_features import extract_all_features
from features.feature_selection import anova_selection, apply_pca
from classification.train_models import train_evaluate_model, train_evaluate_all_models
from classification.evaluate import plot_confusion_matrices, print_classification_report
from utils.visualization import plot_raw_vs_filtered, plot_rr_intervals, plot_model_comparison


# ============================================
# 1. Caricamento del dataset WESAD
# ============================================
def load_wesad_data(wesad_root, window_sec=60, target_fs=256):
    """
    Carica i dati WESAD e restituisce finestre ECG etichettate.
    """
    X_windows = []
    y_windows = []
    subject_files = sorted(Path(wesad_root).glob('S*.pkl'))
    
    if not subject_files:
        raise FileNotFoundError(
            f"Nessun file .pkl trovato in {wesad_root}. Scarica il dataset WESAD da\n"
            "https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/ e posiziona i file (S2.pkl, S3.pkl, ...) in questa cartella."
        )
    
    for subject_file in subject_files:
        data = pd.read_pickle(subject_file)
        ecg_raw = data['signal']['chest']['ECG'].flatten()
        labels_raw = data['label']  # array di interi
        
        original_fs = 700
        if original_fs != target_fs:
            # Ricampionamento ECG (interpolazione lineare)
            ratio = target_fs / original_fs
            new_len = int(len(ecg_raw) * ratio)
            ecg_raw = np.interp(np.linspace(0, len(ecg_raw)-1, new_len),
                                np.arange(len(ecg_raw)), ecg_raw)
            # Ricampionamento etichette (nearest neighbor)
            labels_raw = np.interp(np.linspace(0, len(labels_raw)-1, new_len),
                                   np.arange(len(labels_raw)), labels_raw)
            labels_raw = np.round(labels_raw).astype(int)
        
        window_samples = window_sec * target_fs
        n_windows_subj = len(ecg_raw) // window_samples
        
        for i in range(n_windows_subj):
            start = i * window_samples
            end = start + window_samples
            window = ecg_raw[start:end]
            window_labels = labels_raw[start:end]
            label = np.bincount(window_labels).argmax()
            X_windows.append(window)
            y_windows.append(label)
    
    X = np.array(X_windows)
    y = np.array(y_windows)
    label_map = {1: 'baseline', 2: 'stress', 3: 'amusement'}
    return X, y, label_map


# ============================================
# 2. Preprocessing di tutte le finestre
# ============================================
def preprocess_windows(X, fs=256):
    X_filtered = []
    rr_intervals_list = []
    for ecg_window in X:
        ecg_reconstructed = reconstruct_peaks(ecg_window, fs)
        ecg_filtered = butter_bandpass_filter(ecg_reconstructed, lowcut=0.5,
                                              highcut=49.0, fs=fs, order=6)
        X_filtered.append(ecg_filtered)
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
    
    os.makedirs(args.output, exist_ok=True)
    
    # ------------------- Caricamento -------------------
    print("Caricamento dataset WESAD...")
    X_raw, y_raw, label_map = load_wesad_data(args.data_path, window_sec=args.window_sec)
    print(f"Caricate {len(X_raw)} finestre da {len(list(Path(args.data_path).glob('S*.pkl')))} soggetti.")
    
    # ------------------- Selezione task -------------------
    if args.task == 'binary':
        mask = np.isin(y_raw, [1, 2])
        X_raw = X_raw[mask]
        y_raw = y_raw[mask]
        y = (y_raw - 1).astype(int)
        class_names = ['baseline', 'stress']
    else:
        y = y_raw - 1
        class_names = ['baseline', 'stress', 'amusement']
    
    # ------------------- Preprocessing -------------------
    print("Preprocessing delle finestre ECG...")
    X_filtered, rr_list = preprocess_windows(X_raw)
    
    if args.verbose:
        plot_raw_vs_filtered(X_raw[0], X_filtered[0], output_dir=args.output)
        plot_rr_intervals(rr_list[0], output_dir=args.output)
    
    # ------------------- Estrazione feature -------------------
    print("Estrazione feature...")
    features_df = extract_all_features(X_filtered, rr_list)
    
    # Gestione NaN (dropna)
    initial_len = len(features_df)
    features_df = features_df.dropna()
    if len(features_df) < initial_len:
        print(f"Rimosse {initial_len - len(features_df)} finestre con valori NaN nelle feature non lineari.")
        # Allinea X_filtered e y
        valid_idx = features_df.index
        X_filtered = X_filtered[valid_idx]
        y = y[valid_idx]
    
    feature_names = features_df.columns.tolist()
    X_features = features_df.values
    
    # ------------------- Selezione feature -------------------
    print("Selezione feature con ANOVA (p<0.05)...")
    X_selected, selected_indices = anova_selection(X_features, y, p_threshold=0.05)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    print(f"Feature selezionate: {selected_feature_names}")
    
    if args.use_pca:
        print("Applicazione PCA per ridurre dimensionalità...")
        X_selected, pca_obj = apply_pca(X_selected, variance_ratio=0.95)
        print(f"Nuova dimensionalità dopo PCA: {X_selected.shape[1]}")
    
    # ------------------- Classificazione -------------------
    print("Addestramento e validazione dei modelli...")
    
    if args.classifier == 'all':
        results = train_evaluate_all_models(X_selected, y, cv_folds=10, verbose=args.verbose)
        results_df = pd.DataFrame(results).T
        results_df.to_csv(os.path.join(args.output, 'model_comparison.csv'))
        print("\nRisultati confronto modelli:")
        print(results_df)
        if args.verbose:
            plot_model_comparison(results_df, metric='accuracy', output_dir=args.output)
    else:
        model_map = {'svm': 'SVM', 'knn': 'KNN', 'rf': 'RandomForest'}
        model_name = model_map[args.classifier]
        report = train_evaluate_model(X_selected, y, model_name=model_name,
                                      cv_folds=10, output_dir=args.output,
                                      class_names=class_names)
        print(f"\nReport per {model_name}:")
        print_classification_report(report, class_names)
        if args.verbose:
            plot_confusion_matrices(report['confusion_matrix'], class_names,
                                    title=f"Confusion Matrix - {model_name}",
                                    output_dir=args.output)
    
    print(f"\nPipeline completata. I risultati sono stati salvati in '{args.output}'.")


if __name__ == '__main__':
    main()