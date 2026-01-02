import cv2
import pickle

# Dimensioni del rettangolo
rect_width = 63
rect_height = 27

positions_file = "car_positions.pkl"

try:
    with open(positions_file, 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

def mouseClick(events, x, y, flags, params):
    #Sinistro->Aggiungi posto
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    
    # Destro->Rimuovi posto
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + rect_width and y1 < y < y1 + rect_height:
                posList.pop(i)

    # salvataggio
    with open(positions_file, 'wb') as f:
        pickle.dump(posList, f)

while True:
    #Modifica il primo frame del video(fisso)
    cap = cv2.VideoCapture('parking_1920_1080.mp4') 
    ret, img = cap.read() 
    
    #Disegna i rettangoli salvati
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + rect_width, pos[1] + rect_height), (255, 0, 255), 2)

    cv2.imshow("Mappatura Parcheggio (Tasto 'q' per uscire)", img)
    cv2.setMouseCallback("Mappatura Parcheggio (Tasto 'q' per uscire)", mouseClick)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break