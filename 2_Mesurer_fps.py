import time
import cv2

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit("Impossible d'ouvrir la caméra. Essaie un autre CAMERA_INDEX.")

# Résolution réellement reçue (peut différer de ce que tu espères)
largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Résolution reçue : {largeur} x {hauteur}")

t_precedent = time.perf_counter()
fps_lisse = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Plus d'image reçue.")
        break

    # Temps écoulé depuis la dernière image -> fps instantané
    t_maintenant = time.perf_counter()
    dt = t_maintenant - t_precedent
    t_precedent = t_maintenant
    if dt > 0:
        fps_inst = 1.0 / dt
        # Moyenne lissée pour que le chiffre soit lisible
        fps_lisse = fps_inst if fps_lisse == 0 else 0.9 * fps_lisse + 0.1 * fps_inst

    cv2.putText(frame, f"{fps_lisse:.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()