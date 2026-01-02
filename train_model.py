#Image classificationw dei parcheggi con 2 classi: 1) pieno 2)vuoto 


import os
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# #input_dir = r"C:\Users\Home\Desktop\Image classification\Dati\clf-data"
input_dir = os.path.join(os.getcwd(), 'Dati', 'clf-data')

categorie = ['libero', 'occupato']

dati = []
labels = []

# Caricamento immagini
for categoria_idx, categoria in enumerate(categorie):
    categoria_path = os.path.join(input_dir, categoria)

    for file in os.listdir(categoria_path):
        img_path = os.path.join(categoria_path, file)

        # Lettura immagine
        img = imread(img_path)

        # Ridimensionamento a 15x15
        img = resize(img, (15, 15))

        # Appiattimento (da 15x15x3 → 675 valori)
        dati.append(img.flatten())

        # Etichetta (0 = libero, 1 = occupato)
        labels.append(categoria_idx)

# Conversione in array NumPy
dati = np.asarray(dati)
labels = np.asarray(labels)

print("Shape dati:", dati.shape) #(resistuisce la somma dei dati (occupati e liberi) e la grandezza dalla quale vieen rappresenteaato ogni dato(15*15*3))
print("Shape labels:", labels.shape) #restituisce l'array di 0,1 a seconda se e' libero o occupato



# Split del dataset in training set e test set
# x_train, x_test → immagini di addestramento e di test
# y_train, y_test → etichette corrispondenti
# test_size=0.2 → 20% dei dati va nel test set
# shuffle=True → mescola i dati prima di dividere
# stratify=labels → mantiene la stessa proporzione di classi in train e test
x_train, x_test, y_train, y_test = train_test_split(dati, labels, test_size=0.2, shuffle=True, stratify=labels)


# Creo il classificatore SVM (Support Vector Classifier) con parametri di default
classifier = SVC()


# Dizionario dei parametri da testare nella Grid Search
# gamma → parametro del kernel RBF della SVM
# C → parametro di penalizzazione (più grande = minor tolleranza agli errori)
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C':[1, 10, 1000]}]


# Scelta dei miglior parametri
grid_search = GridSearchCV(classifier, parameters)


# Alleno il modello sui dati di training (x_train, y_train)
grid_search.fit(x_train, y_train)


# Prendo il miglior classificatore trovato
best_estimator = grid_search.best_estimator_


# Predizioni sul test set 
y_prediction = best_estimator.predict(x_test)


# Percentuale di correttezza (accuracy)
score = accuracy_score(y_prediction, y_test)

print("{}% of sample were completely classified".format(score * 100))


# Salvataggio del modello su disco con pickle
# Usando il costrutto 'with' il file viene chiuso automaticamente
import pickle

with open('model.p', 'wb') as f:
    pickle.dump(best_estimator, f)

print("Modello salvato correttamente in 'model.p'")
