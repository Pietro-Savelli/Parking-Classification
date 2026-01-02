# Parking Classification

Progetto di **computer vision** sviluppato a scopo didattico, che utilizza una  
**Support Vector Machine (SVM)** per classificare i posti auto in un video come
**liberi** o **occupati**.

Il sistema analizza un video di un parcheggio e visualizza il risultato in tempo reale:
- **Verde** → posto libero  
- **Rosso** → posto occupato  



## Obiettivo del progetto

L’obiettivo del progetto è applicare tecniche di:
- elaborazione di immagini
- machine learning supervisionato
- classificazione binaria

per comprendere il flusso completo di un sistema di **parking occupancy detection**:
raccolta dati, addestramento del modello e inferenza su video.


## Contenuto del repository

Il repository contiene **esclusivamente il codice sorgente**.  
Per motivi di dimensione, il dataset di immagini e il video del parcheggio
non sono inclusi.



## File principali

- `train_model.py`  
  Addestra il modello SVM usando le immagini dei posti auto.

- `mappatore_posti.py`  
  Permette di selezionare manualmente i posti auto sul frame del video.

- `main_video.py`  
  Analizza il video e classifica i posti auto in tempo reale.



