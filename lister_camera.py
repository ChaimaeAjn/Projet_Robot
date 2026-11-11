import cv2

NB_INDEX_A_TESTER = 5   # teste les index 0 à 4

print("Pour chaque caméra trouvée : une fenêtre s'ouvre. Appuie sur une touche pour passer à la suivante.\n")

for i in range(NB_INDEX_A_TESTER):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Index {i} : pas de caméra")
        continue

    ok, frame = cap.read()
    if not ok:
        print(f"Index {i} : ouverte, mais aucune image reçue")
        cap.release()
        continue

    h, w = frame.shape[:2]
    print(f"Index {i} : caméra trouvée, résolution reçue {w} x {h}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, f"index {i} - {w}x{h} - une touche = suivante", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("liste des cameras", frame)
        if cv2.waitKey(1) != -1:
            break

    cap.release()

cv2.destroyAllWindows()
print("\nMets l'index de la bonne caméra dans CAMERA_INDEX, en haut de 6_plateau.py.")