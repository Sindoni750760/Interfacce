"""
ecg_cleaning.py – Pulizia e filtraggio del segnale ECG secondo il paper
                   "Detection and analysis driver state with electrocardiogram".

Contiene:
- ricostruzione del segnale per rimuovere picchi anomali e wandering;
- applicazione del filtro Butterworth passa-banda (0.5 – 49 Hz, 6° ordine).
"""

import numpy as np
from scipy.signal import butter, filtfilt

def reconstruct_peaks(ecg_raw, fs=256, r_peak_threshold=3000.0):
    """
    Ricostruisce il segnale ECG rimuovendo i picchi R anomali (oltre soglia)
    e attenuando il wandering di baseline, come descritto nel paper.

    La procedura:
      1. Identifica i campioni il cui valore assoluto supera la soglia (3000 μV),
         che nel paper vengono considerati artefatti o falsi picchi.
      2. Sostituisce tali campioni con una interpolazione lineare tra i vicini validi,
         in modo da non introdurre discontinuità (simile all'azzeramento e ricostruzione
         citato nel paper).
      3. Applica un semplice filtro passa-alto a 0.5 Hz per rimuovere la baseline wander,
         ma solo se il segnale ha una componente continua importante. In alternativa,
         si può demandare al successivo filtro Butterworth, che già taglia le basse frequenze.

    Nota: il paper menziona anche l'uso di FastICA per separare EMG ed ECG; qui non
    implementato perché nel dataset WESAD l'ECG è già pre‑amplificato e pulito.
    """
    # Copia del segnale
    ecg = ecg_raw.copy().astype(np.float64)
    
    # 1. Rimozione dei picchi anomali (valori assoluti > soglia)
    outliers = np.abs(ecg) > r_peak_threshold
    if np.any(outliers):
        # Sostituisco con interpolazione lineare tra i punti validi più vicini
        x = np.arange(len(ecg))
        valid = ~outliers
        if np.sum(valid) < 2:
            # Caso estremo: quasi tutti outlier, non fare nulla o sostituire con media
            ecg[outliers] = np.mean(ecg[valid]) if np.any(valid) else 0.0
        else:
            ecg[outliers] = np.interp(x[outliers], x[valid], ecg[valid])
    
    # 2. Rimozione del wandering di baseline con filtro passa-alto (fc=0.5 Hz)
    #    Si può realizzare con un Butterworth high-pass del 2° ordine.
    nyq = 0.5 * fs
    lowcut_hp = 0.5  # Hz
    if lowcut_hp > 0:
        b_hp, a_hp = butter(2, lowcut_hp / nyq, btype='high')
        ecg = filtfilt(b_hp, a_hp, ecg)
    
    return ecg

def butter_bandpass_filter(data, lowcut=0.5, highcut=49.0, fs=256, order=6):
    """
    Applica un filtro Butterworth passa-banda all'ECG, come da paper.

    Parametri:
        data : array 1D – segnale ECG pre‑ricostruito
        lowcut, highcut : float – frequenze di taglio in Hz (default 0.5 e 49)
        fs : int – frequenza di campionamento (default 256)
        order : int – ordine del filtro (default 6)
    Restituisce:
        filtered : array 1D – segnale filtrato
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered