Parking Classification 

Progetto di computer vision che utilizza una SVM per capire se i posti auto in un video sono liberi o occupati.

Il sistema analizza un video di un parcheggio e colora:
- **Verde** → posto libero  
- **Rosso** → posto occupato  


Contenuto del repository:

Questo repository contiene solo il codice.  
Il dataset di immagini e il video non sono inclusi per motivi di dimensione.




File principali

- `train_model.py`  
  Addestra il modello SVM usando le immagini dei posti auto.

- `mappatore_posti.py`  
  Permette di selezionare manualmente i posti auto sul frame del video.

- `main_video.py`  
  Analizza il video e classifica i posti auto in tempo reale.

