"""
ecg_cleaning.py – Pulizia e filtraggio del segnale ECG secondo il paper
                   "Detection and analysis driver state with electrocardiogram".

Contiene:
- ricostruzione del segnale per rimuovere picchi anomali (interpolazione);
- applicazione del filtro Butterworth passa-banda (0.5 – 49 Hz, 6° ordine).
"""

import numpy as np
from scipy.signal import butter, filtfilt

def reconstruct_peaks(ecg_raw, fs=256, r_peak_threshold=3000.0):
    """
    Ricostruisce il segnale ECG rimuovendo i picchi R anomali (oltre soglia)
    tramite interpolazione lineare. Il wandering di baseline verrà rimosso
    successivamente dal filtro passa-banda, quindi qui non applichiamo alcun filtro.
    """
    ecg = ecg_raw.copy().astype(np.float64)
    outliers = np.abs(ecg) > r_peak_threshold
    if np.any(outliers):
        x = np.arange(len(ecg))
        valid = ~outliers
        if np.sum(valid) < 2:
            ecg[outliers] = np.mean(ecg[valid]) if np.any(valid) else 0.0
        else:
            ecg[outliers] = np.interp(x[outliers], x[valid], ecg[valid])
    return ecg

def butter_bandpass_filter(data, lowcut=0.5, highcut=49.0, fs=256, order=6):
    """
    Applica un filtro Butterworth passa-banda all'ECG, come da paper.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered