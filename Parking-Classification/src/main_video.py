import pickle
import cv2
import numpy as np
from skimage.transform import resize

# identiche a quelle usate nel Mappatore
rect_width = 63
rect_height = 27


# 1. Carica il modello e le posizioni
try:
    model = pickle.load(open("model.p", "rb"))
    print("Modello caricato.")
except:
    print("ERRORE: model.p non trovato. Hai fatto l'addestramento?")
    exit()

try:
    with open("car_positions.pkl", 'rb') as f:
        posList = pickle.load(f)
    print(f"Caricate {len(posList)} posizioni.")
except:
    print("ERRORE: car_positions.pkl non trovato. Esegui prima il mappatore.")
    exit()

cap = cv2.VideoCapture("parking_1920_1080.mp4") 
categories = ['not_empty', 'empty'] 

def checkParkingSpace(img, imgOriginal):
    spaceCounter = 0

    for pos in posList:
        x, y = pos
        
        crop_img = img[y:y+rect_height, x:x+rect_width]
        
        if crop_img.size == 0: continue

        img_resized = resize(crop_img, (15, 15))
        flat_data = [img_resized.flatten()]
        
        prediction = model.predict(flat_data)
        
        if categories[prediction[0]] == 'empty':
            color = (0, 255, 0) # Verde
            spaceCounter += 1
            thickness = 2
        else:
            color = (0, 0, 255) # Rosso
            thickness = 2

        # Disegna
        cv2.rectangle(imgOriginal, (x, y), (x + rect_width, y + rect_height), color, thickness)

    # Scrivi a scehrmo quanti posti sono liberi
    cv2.putText(imgOriginal, f'Liberi: {spaceCounter}/{len(posList)}', (50, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    checkParkingSpace(frame_rgb, frame)
    
    cv2.imshow("Video Parcheggio AI", frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()