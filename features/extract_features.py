"""
extract_features.py – Estrazione feature lineari e non lineari.
"""

import numpy as np
from scipy.stats import skew, kurtosis, trim_mean, hmean
from utils.signal_utils import approximate_entropy, hurst_rs


def _window_features_linear(ecg_window):
    """Calcola le 14 feature lineari su una finestra ECG."""
    feats = {}
    n = len(ecg_window)
    feats['mean'] = np.mean(ecg_window)
    feats['median'] = np.median(ecg_window)
    feats['std'] = np.std(ecg_window, ddof=1)
    feats['var'] = np.var(ecg_window, ddof=1)
    feats['max'] = np.max(ecg_window)
    feats['min'] = np.min(ecg_window)
    feats['q1'] = np.percentile(ecg_window, 25)
    feats['q2'] = np.percentile(ecg_window, 50)
    feats['q3'] = np.percentile(ecg_window, 75)
    feats['iqr'] = feats['q3'] - feats['q1']
    feats['skewness'] = skew(ecg_window, bias=False)
    feats['kurtosis'] = kurtosis(ecg_window, bias=False)
    feats['rms'] = np.sqrt(np.mean(ecg_window ** 2))
    feats['energy'] = np.sum(ecg_window ** 2)
    feats['power'] = np.mean(ecg_window ** 2)
    return feats


def _window_features_nonlinear(rr_intervals):
    """
    Calcola le feature non lineari sugli intervalli RR:
    - Approximate Entropy
    - Hurst exponent
    - Central tendency measures (harmonic mean, nanmean, trimmean)
    """
    feats = {}
    if len(rr_intervals) < 4:
        feats['approx_entropy'] = np.nan
        feats['hurst'] = np.nan
        feats['harmonic_mean'] = np.nan
        feats['nanmean'] = np.nan
        feats['trimmean'] = np.nan
    else:
        feats['approx_entropy'] = approximate_entropy(rr_intervals)
        feats['hurst'] = hurst_rs(rr_intervals)
        feats['harmonic_mean'] = hmean(rr_intervals) if np.all(rr_intervals > 0) else np.nan
        feats['nanmean'] = np.nanmean(rr_intervals)
        feats['trimmean'] = trim_mean(rr_intervals, proportiontocut=0.1)
    return feats


def extract_all_features(X_filtered, rr_list):
    """
    Estrae tutte le 20 feature per ogni finestra ECG.
    """
    import pandas as pd
    all_features = []
    for ecg_win, rr in zip(X_filtered, rr_list):
        f_lin = _window_features_linear(ecg_win)
        f_nl = _window_features_nonlinear(rr)
        combined = {**f_lin, **f_nl}
        all_features.append(combined)
    df = pd.DataFrame(all_features)
    col_order = [
        'mean', 'median', 'std', 'var', 'max', 'min',
        'q1', 'q2', 'q3', 'iqr', 'skewness', 'kurtosis',
        'rms', 'energy', 'power',
        'approx_entropy', 'hurst', 'harmonic_mean', 'nanmean', 'trimmean'
    ]
    df = df[col_order]
    return df