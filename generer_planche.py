import json
import cv2
import numpy as np

# --- PARAMÈTRES ---
# À MESURER : largeur réelle de l'image affichée à l'écran (en mm) divisée par 1600 (largeur de l'image en pixels).
# Première exécution avec une valeur approximative, puis mesure à la règle, corrige, et relance.
MM_PAR_PIXEL = 0.15
DICTIONNAIRE = "DICT_4X4_50"
FICHIER_CONFIG = "plateau_test.json"
# ------------------

W, H, TAG = 1600, 1200, 200   # image 4:3 ; tags de 200 px de côté
REFS = {0: (150, 150), 1: (1450, 150), 2: (1450, 1050), 3: (150, 1050)}   # centres en pixels : 4 coins
TEST = {5: (800, 600)}                                                      # tag de contrôle, au milieu

dico = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONNAIRE))
img = np.full((H, W), 255, np.uint8)
for i, (cx, cy) in {**REFS, **TEST}.items():
    x0, y0 = cx - TAG // 2, cy - TAG // 2
    img[y0:y0 + TAG, x0:x0 + TAG] = cv2.aruco.generateImageMarker(dico, i, TAG)
cv2.imwrite("planche.png", img)

# Repère du plateau de test : origine = coin haut-gauche de l'image, x vers la droite, y vers le bas (en mm)
config = {
    "dictionnaire": DICTIONNAIRE,
    "references": {str(i): [round(cx * MM_PAR_PIXEL, 1), round(cy * MM_PAR_PIXEL, 1)] for i, (cx, cy) in REFS.items()},
    "hauteurs_mm": {},
    "hauteur_par_defaut_mm": 0
}
with open(FICHIER_CONFIG, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"planche.png créée ({W}x{H} px) et {FICHIER_CONFIG} écrit avec {MM_PAR_PIXEL} mm par pixel.")
for i, (cx, cy) in TEST.items():
    print(f"Tag de contrôle {i} : position attendue ({cx * MM_PAR_PIXEL:.1f}, {cy * MM_PAR_PIXEL:.1f}) mm")