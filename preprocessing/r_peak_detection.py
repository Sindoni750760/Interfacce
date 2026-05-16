"""
r_peak_detection.py – Rilevamento picchi R e calcolo intervalli RR,
                      ora senza neurokit2.
"""

import numpy as np
from utils.signal_utils import detect_r_peaks


def compute_rr_intervals(r_peaks, fs=256):
    """
    Calcola la serie degli intervalli RR (in ms) a partire dagli indici dei picchi R.
    """
    if len(r_peaks) < 2:
        return np.array([])
    rr_campioni = np.diff(r_peaks)
    rr_intervals = rr_campioni * (1000.0 / fs)
    return rr_intervals