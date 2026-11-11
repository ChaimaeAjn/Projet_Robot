import time
import cv2

# --- PARAMÈTRES (à modifier ici, pas dans le code plus bas) ---
CAMERA_INDEX = 0
# Dictionnaire de tags : à confirmer avec le règlement / les arbitres.
# En attendant, n'importe lequel convient pour t'entraîner, mais il doit
# être le même que celui de tes tags imprimés.
DICTIONNAIRE = cv2.aruco.DICT_4X4_50
# ---------------------------------------------------------------

dictionary = cv2.aruco.getPredefinedDictionary(DICTIONNAIRE)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, params)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit("Impossible d'ouvrir la caméra. Essaie un autre CAMERA_INDEX.")

t_precedent = time.perf_counter()
fps_lisse = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Plus d'image reçue.")
        break

    # On détecte TOUS les tags visibles (pas de liste d'identifiants en dur)
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for tag_corners, tag_id in zip(corners, ids.flatten()):
            pts = tag_corners.reshape(4, 2)       # les 4 coins sous forme d'un tableau 4 lignes x 2 colonnes (x, y)
            cx, cy = pts.mean(axis=0)             # moyenne des x et moyenne des y = le centre
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{tag_id}: ({int(cx)}, {int(cy)})", (int(cx) + 10, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    t_maintenant = time.perf_counter()
    dt = t_maintenant - t_precedent
    t_precedent = t_maintenant
    if dt > 0:
        fps_inst = 1.0 / dt
        fps_lisse = fps_inst if fps_lisse == 0 else 0.9 * fps_lisse + 0.1 * fps_inst

    cv2.putText(frame, f"{fps_lisse:.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()