import cv2

CAMERA_INDEX = 0  # 0 = première caméra trouvée. Essaie 1, 2... si tu vois la mauvaise (ex: la webcam du PC au lieu de l'iPhone)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit("Impossible d'ouvrir la caméra. Essaie un autre CAMERA_INDEX.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Plus d'image reçue.")
        break

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # appuie sur q pour quitter
        break

cap.release()
cv2.destroyAllWindows()