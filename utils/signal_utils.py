"""
utils/signal_utils.py – Funzioni di elaborazione segnali senza librerie specializzate.
"""

import numpy as np
from scipy.signal import find_peaks


def approximate_entropy(rr_intervals, m=2, r_factor=0.2):
    """
    Calcola l'Approssimate Entropy (ApEn) sulla serie di intervalli RR.
    """
    if len(rr_intervals) < m + 1:
        return np.nan
    r = r_factor * np.std(rr_intervals)
    N = len(rr_intervals)

    def _maxdist(xi, xj):
        return np.max(np.abs(xi - xj))

    def _phi(m):
        patterns = np.array([rr_intervals[i:i+m] for i in range(N - m + 1)])
        C = np.zeros(len(patterns))
        for i, pat in enumerate(patterns):
            # Calcola distanza massima rispetto a tutti gli altri pattern
            dists = np.array([_maxdist(pat, other) for other in patterns])
            C[i] = np.sum(dists <= r) / (N - m + 1)
        return np.mean(np.log(C))

    return _phi(m) - _phi(m+1)


def hurst_rs(series):
    """
    Esponente di Hurst con metodo R/S (rescaled range).
    """
    if len(series) < 4:
        return np.nan
    n = len(series)
    mean = np.mean(series)
    y = series - mean
    z = np.cumsum(y)
    R = np.max(z) - np.min(z)
    S = np.std(series, ddof=1)
    if S == 0:
        return np.nan
    return np.log(R / S) / np.log(n)


def detect_r_peaks(ecg_filtered, fs=256):
    """
    Rilevazione picchi R senza neurokit2.
    Utilizza moving average e find_peaks di scipy.
    """
    # Valore assoluto
    ecg_abs = np.abs(ecg_filtered)
    # Moving average finestra di ~150 ms
    window = max(1, int(0.15 * fs))
    kernel = np.ones(window) / window
    ecg_ma = np.convolve(ecg_abs, kernel, mode='same')
    # Distanza minima tra picchi: 300 ms (circa 200 bpm max)
    min_distance = int(0.3 * fs)
    peaks, _ = find_peaks(ecg_ma, distance=min_distance,
                          height=np.percentile(ecg_ma, 50))
    return peaks