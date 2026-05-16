# Rilevamento di stati di ipovigilanza mediante segnali ECG - Progetto Interfacce uomo-macchina

Progetto d'esame per il corso di **interfacce uomo-macchina** (a.a. 2025/26)
Università degli studi dell'Insubria

## Obiettivi del progetto

L’obiettivo è sviluppare una pipeline di elaborazione dati che, a partire da segnali elettrocardiografici (ECG), sia in grado di distinguere **stati psicofisiologici** (es. stress vs. baseline, emozioni positive vs. negative) utilizzando tecniche di machine learning.

## Riferimento scientifico principale

- **Titolo:** *Detection and analysis driver state with electrocardiogram*  
- **Autori:** Sahayadhas et al.  
- **Pubblicazione:** (inserire rivista/anno)  
- **File:** `Detection and analysis driver state with electrocardiogram.pdf`

Il paper originale classifica 5 stati di ipovigilanza (normale, sonnolenza, fatica, disattenzione visiva, disattenzione cognitiva) a partire da ECG, utilizzando feature lineari e non lineari e un classificatore Ensemble. 
Costituisce la baseline metodologica per questo progetto.

## Dataset: WESAD

- **Sorgente:** [UCI Machine Learning Repository / IEEE DataPort] (fornire link specifico)  
- **Segnali disponibili:** ECG, PPG, EDA, EMG, respirazione, accelerometro (a seconda del dispositivo).  
- **Partecipanti:** 15 soggetti (dato indicativo; verificare versione impiegata).  
- **Protocollo:** Ogni soggetto ha attraversato tre condizioni: *baseline* (neutro), *stress* (indotto da compiti cognitivi), *amusement* (divertimento).  
- **Frequenza di campionamento:** ECG a 700 Hz (tipica per il dispositivo RespiBAN) – da ricampionare a 256 Hz per allinearsi al paper.

Per il progetto, il segnale ECG del dispositivo toracico (RespiBAN) viene utilizzato come unica sorgente dati, consentendo di applicare fedelmente la pipeline descritta nel paper.

## Metodologia

La pipeline si articola in sei fasi:

1. **Acquisizione e segmentazione**  
   Caricamento del segnale ECG, suddivisione in finestre temporali (es. 60 secondi) associate alle etichette delle condizioni sperimentali.

2. **Pre-processing**  
   *Ricostruzione dei picchi R* tramite algoritmo di Pan-Tompkins per rimuovere falsi picchi e wandering di baseline.  
   *Filtraggio* con Butterworth passa-banda del 6° ordine (0.5 – 49 Hz), come descritto nel paper.

3. **Estrazione delle feature**  
   - **14 feature lineari** calcolate sul segnale ECG filtrato: media, mediana, deviazione standard, varianza, quartili, IQR, massimo, minimo, skewness, kurtosis, RMS, energia, potenza.  
   - **3 feature non lineari** calcolate sugli intervalli RR (HRV): entropia approssimata, esponente di Hurst, misure di tendenza centrale (media armonica, media trimmed, nanmean).

4. **Selezione delle feature**  
   - Test ANOVA per identificare feature statisticamente significative (p < 0.05).  
   - Riduzione dimensionale con PCA (opzionale, per mantenere il 95% della varianza).

5. **Classificazione**  
   Addestramento e validazione di tre classificatori:
   - **SVM** (kernel RBF)
   - **K-Nearest Neighbors**
   - **Ensemble (Random Forest)**  
   *(Tutti disponibili in scikit-learn)*

6. **Valutazione**  
   - Strategia di validazione: cross‑validation stratificata a 10 fold, con normalizzazione (StandardScaler) applicata all'interno di ogni fold.  
   - Metriche: accuratezza, precisione, recall, F1‑score (sia globali che per classe), matrice di confusione.  
   - Confronto sistematico con i risultati del paper (accuratezza binaria ~100% per Normal/Drowsy; multi‑classe ~58.3% con PCA+Ensemble)

## Struttura del repository

```
Progetto_ECG/
│
├── README.md # Questo file
├── requirements.txt # Dipendenze Python
├── main.py # Script principale che esegue l'intera pipeline
│
├── data/ # Istruzioni per scaricare il dataset WESAD
│ └── download_WESAD.txt
│
├── preprocessing/
│ ├── ecg_cleaning.py # Funzioni per pulizia e filtraggio
│ └── r_peak_detection.py # Rilevamento picchi R e calcolo intervalli RR
│
├── features/
│ ├── extract_features.py # Estrazione di tutte le feature
│ └── feature_selection.py # ANOVA, PCA e selezione
│
├── classification/
│ ├── train_models.py # Pipeline di training e cross-validation
│ └── evaluate.py # Metriche e visualizzazione risultati
│
├── utils/
│ └── visualization.py # Grafici (segnale grezzo vs filtrato, feature importance, ecc.)
│
└── results/ # Salvataggio dei modelli e delle metriche
└── (ignorato da git)
```

## Dipendenze e librerie utilizzate

Il progetto è basato sull'uso di librerie standard per l'elaborazione di segnali, machine learning e visualizzazione:
- numpy: Utilizzato per la gestione di array numerici e operazioni vettoriali su segnali ECG e intervalli RR
- scipy: Utilizzato per implementare il filtro Butterworth passa-banda (0.5 Hz, ordine 6)
- pandas: Utilizzato per effettuare la lettura del file .pkl del dataset WESAD e manipolazione del DataFrame delle feature estratte.
- scikit-learn: Utilizzato per i classificatori, come le Random Forest, selezione delle feature, cross-validazione e metriche di performance.
- matplotlib: Usato per generare i grafici, tra i quali segnali grezzo vs filtrato, intervalli RR, feature importance, confronto dei modelli

Tutte le dipendenze possono essere installate automaticamennte con:

```bash
pip install -r requirements.txt
```

## Requisiti e installazione

È richiesto Python 3.8 o superiore. Clonare il repository e installare le dipendenze:

```bash
git clone <url-del-repo>
cd Progetto_ECG
pip install -r requirements.txt
