"""
visualization.py – Funzioni di visualizzazione per il progetto ECG:
                   - Confronto segnale grezzo vs filtrato
                   - Intervalli RR
                   - Confronto performance modelli
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_raw_vs_filtered(raw, filtered, fs=256, time_window=None):
    """
    Mostra il segnale ECG prima e dopo il filtraggio.
    
    Parametri:
        raw, filtered : array 1D – segnali ECG
        fs : int – frequenza di campionamento (Hz)
        time_window : tuple (start_sec, end_sec) – opzionale, per zoommare
    """
    t = np.arange(len(raw)) / fs
    if time_window:
        start, end = time_window
        mask = (t >= start) & (t <= end)
        t = t[mask]
        raw = raw[mask]
        filtered = filtered[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    
    ax1.plot(t, raw, color='gray', alpha=0.7, linewidth=0.8)
    ax1.set_title('ECG grezzo (prima del preprocessing)')
    ax1.set_ylabel('Ampiezza (μV)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, filtered, color='blue', linewidth=0.8)
    ax2.set_title('ECG filtrato (Butterworth 0.5–49 Hz, 6° ordine)')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Ampiezza (μV)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ecg_raw_vs_filtered.png', dpi=150)
    plt.show()

def plot_rr_intervals(rr_intervals):
    """
    Visualizza la serie degli intervalli RR (HRV) in funzione del battito.
    """
    if len(rr_intervals) == 0:
        print("Nessun intervallo RR da visualizzare.")
        return
    
    beat = np.arange(len(rr_intervals))
    
    plt.figure(figsize=(10, 4))
    plt.plot(beat, rr_intervals, marker='o', linestyle='-', markersize=3, color='darkgreen')
    plt.title('Intervalli RR (HRV)')
    plt.xlabel('Battito #')
    plt.ylabel('Intervallo RR (ms)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('rr_intervals.png', dpi=150)
    plt.show()

def plot_model_comparison(results_df, metric='accuracy'):
    """
    Barplot di confronto tra modelli per la metrica scelta.
    
    Parametri:
        results_df : pd.DataFrame – indice = nome modello, colonne = metriche
        metric : str – nome della metrica da plottare
    """
    if metric not in results_df.columns:
        print(f"Metrica '{metric}' non trovata. Disponibili: {list(results_df.columns)}")
        return
    
    vals = results_df[metric]
    colors = sns.color_palette('Set2', len(vals))
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(vals.index, vals.values, color=colors)
    plt.title(f'Confronto modelli – {metric.capitalize()}')
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1.05)
    
    # Annotazioni sui valori
    for bar, val in zip(bars, vals.values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{metric}.png', dpi=150)
    plt.show()

def plot_feature_importance(feature_names, importance, title='Importanza delle feature'):
    """
    Barplot orizzontale per l'importanza delle feature (es. da Random Forest).
    """
    idx = np.argsort(importance)
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = importance[idx]
    
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_names, sorted_imp, color='steelblue')
    plt.xlabel('Importanza')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()