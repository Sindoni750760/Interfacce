"""
extract_features.py – Estrae le 20 feature (14 lineari + 3 non-lineari/gruppo)
                      descritte nel paper "Detection and analysis driver state
                      with electrocardiogram".

Le feature lineari (dominio del tempo) sono calcolate sul segnale ECG filtrato.
Le feature non-lineari (ApEn, Hurst, misure di tendenza centrale) sono calcolate
sulla serie degli intervalli RR (HRV), come suggerito dalla sezione metodologica
del paper e dalla pratica comune in letteratura.
"""

import numpy as np
from scipy.stats import skew, kurtosis, trim_mean, hmean
import antropy
import nolds

def _window_features_linear(ecg_window):
    """Calcola le 14 feature lineari su un vettore 1D di ECG."""
    feats = {}
    n = len(ecg_window)
    feats['mean'] = np.mean(ecg_window)
    feats['median'] = np.median(ecg_window)
    feats['std'] = np.std(ecg_window, ddof=1)
    feats['var'] = np.var(ecg_window, ddof=1)
    feats['max'] = np.max(ecg_window)
    feats['min'] = np.min(ecg_window)
    feats['q1'] = np.percentile(ecg_window, 25)
    feats['q2'] = np.percentile(ecg_window, 50)  # mediana
    feats['q3'] = np.percentile(ecg_window, 75)
    feats['iqr'] = feats['q3'] - feats['q1']
    feats['skewness'] = skew(ecg_window, bias=False)
    feats['kurtosis'] = kurtosis(ecg_window, bias=False)  # eccesso di curtosi
    feats['rms'] = np.sqrt(np.mean(ecg_window ** 2))
    feats['energy'] = np.sum(ecg_window ** 2)
    feats['power'] = np.mean(ecg_window ** 2)  # potenza media (come da paper)
    return feats

def _window_features_nonlinear(rr_intervals):
    """
    Calcola le feature non-lineari sugli intervalli RR (in ms):
    - Approximate Entropy (ApEn)
    - Hurst exponent
    - Misure di tendenza centrale: Nanmean, Trimmean, Harmonic mean
    """
    feats = {}
    if len(rr_intervals) < 4:  # serve una lunghezza minima per alcune
        feats['approx_entropy'] = np.nan
        feats['hurst'] = np.nan
        feats['harmonic_mean'] = np.nan
        feats['nanmean'] = np.nan
        feats['trimmean'] = np.nan
    else:
        feats['approx_entropy'] = antropy.app_entropy(rr_intervals, order=2,
                                                       metric='chebyshev')
        feats['hurst'] = nolds.hurst_rs(rr_intervals, fit='poly')
        # Central tendency measures
        feats['harmonic_mean'] = hmean(rr_intervals) if np.all(rr_intervals > 0) else np.nan
        feats['nanmean'] = np.nanmean(rr_intervals)  # in pratica coincide con mean se non ci sono NaN
        feats['trimmean'] = trim_mean(rr_intervals, proportiontocut=0.1)
    return feats

def extract_all_features(X_filtered, rr_list):
    """
    Estrae tutte le 20 feature per ogni finestra ECG.

    Parametri:
        X_filtered : np.ndarray shape (n_finestre, n_campioni)
            Finestre di ECG filtrato.
        rr_list : list of np.ndarray
            Lista di array degli intervalli RR (ms) per ogni finestra.

    Restituisce:
        features_df : pd.DataFrame
            DataFrame con n_finestre righe e 20 colonne (nomi feature).
    """
    import pandas as pd
    all_features = []
    for ecg_win, rr in zip(X_filtered, rr_list):
        f_lin = _window_features_linear(ecg_win)
        f_nl = _window_features_nonlinear(rr)
        combined = {**f_lin, **f_nl}
        all_features.append(combined)
    df = pd.DataFrame(all_features)
    # Riordina colonne per coerenza con il paper (lineari, poi non-lineari)
    col_order = [
        'mean', 'median', 'std', 'var', 'max', 'min',
        'q1', 'q2', 'q3', 'iqr', 'skewness', 'kurtosis',
        'rms', 'energy', 'power',
        'approx_entropy', 'hurst', 'harmonic_mean', 'nanmean', 'trimmean'
    ]
    df = df[col_order]
    return df