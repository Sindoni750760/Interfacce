"""
r_peak_detection.py – Rilevamento picchi R e calcolo intervalli RR,
                      utilizzando l'algoritmo di Pan‑Tompkins (via neurokit2)
                      come proxy del metodo citato nel paper.
"""

import numpy as np
import neurokit2 as nk

def detect_r_peaks(ecg_filtered, fs=256):
    """
    Rileva i picchi R in un segnale ECG già filtrato.

    Utilizza il rilevatore QRS di neurokit2 (basato su Pan‑Tompkins o derivati),
    che è uno standard di letteratura e fornisce risultati confrontabili con il
    metodo descritto nel paper (QRS delineator based on Pan & Tompkins).

    Restituisce:
        r_peaks : array 1D di interi – indici dei campioni in cui si trovano i picchi R.
    """
    # neurokit2.ecg_peaks richiede il segnale e la frequenza di campionamento
    _, results = nk.ecg_peaks(ecg_filtered, sampling_rate=fs, method='neurokit')
    r_peaks = results['ECG_R_Peaks']  # array di indici (0-based)
    return r_peaks

def compute_rr_intervals(r_peaks, fs=256):
    """
    Calcola la serie degli intervalli RR (in ms) a partire dagli indici dei picchi R.

    Parametri:
        r_peaks : array di indici (campioni)
        fs : frequenza di campionamento (Hz)
    Restituisce:
        rr_intervals : array di float – differenze temporali in millisecondi
    """
    if len(r_peaks) < 2:
        return np.array([])
    # Differenza in campioni, convertita in ms
    rr_campioni = np.diff(r_peaks)
    rr_intervals = rr_campioni * (1000.0 / fs)  # ms
    return rr_intervals