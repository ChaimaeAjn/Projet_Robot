import os
import cv2

# --- PARAMÈTRES ---
CAMERA_INDEX = 0
COINS = (9, 6)           # coins INTÉRIEURS du damier (colonnes, lignes) : un damier de 10x7 cases a 9x6 coins intérieurs
DOSSIER = "calib_images"
ECHELLE_DETECTION = 0.5  # on cherche le damier sur une image réduite de moitié (beaucoup plus rapide)
# -------------------

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

os.makedirs(DOSSIER, exist_ok=True)
n = len(os.listdir(DOSSIER))   # pour ne pas écraser les photos déjà prises

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit("Impossible d'ouvrir la caméra. Essaie un autre CAMERA_INDEX.")

print("s = sauvegarder la photo (seulement si le damier est détecté) | q = quitter")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Plus d'image reçue.")
        break

    # Détection sur une image réduite : le temps de calcul chute fortement
    petite = cv2.resize(frame, None, fx=ECHELLE_DETECTION, fy=ECHELLE_DETECTION,
                        interpolation=cv2.INTER_AREA)
    gris = cv2.cvtColor(petite, cv2.COLOR_BGR2GRAY)
    trouve, coins = cv2.findChessboardCorners(gris, COINS, flags=FLAGS)

    # On dessine sur une COPIE : la photo sauvegardée doit rester propre et en pleine résolution
    apercu = frame.copy()
    if trouve:
        cv2.drawChessboardCorners(apercu, COINS, coins / ECHELLE_DETECTION, trouve)  # coins remis à l'échelle
    etat = "damier detecte" if trouve else "damier NON detecte"
    cv2.putText(apercu, f"{etat} | photos: {n}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if trouve else (0, 0, 255), 2)
    cv2.imshow("capture damier", apercu)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("s") and trouve:
        cv2.imwrite(f"{DOSSIER}/img_{n:02d}.png", frame)
        n += 1
        print(f"Photo {n} sauvegardée")
    elif touche == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()