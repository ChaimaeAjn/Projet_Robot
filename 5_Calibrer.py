import glob
import cv2
import numpy as np

# --- PARAMÈTRES (doivent correspondre à ton damier imprimé) ---
COINS = (9, 6)            # coins intérieurs (colonnes, lignes)
TAILLE_CASE_MM = 25.0     # côté réel d'une case, MESURÉ sur ton impression
DOSSIER = "calib_images"
FICHIER_SORTIE = "calibration.npz"
AFFICHER_APERCU = True
# ---------------------------------------------------------------

# Position réelle des coins du damier, dans le repère du damier (en mm), z = 0 car il est plat
objp = np.zeros((COINS[0] * COINS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:COINS[0], 0:COINS[1]].T.reshape(-1, 2) * TAILLE_CASE_MM

critere = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

points_3d, points_2d = [], []
taille_image = None
fichiers = sorted(glob.glob(f"{DOSSIER}/*.png"))

for f in fichiers:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    taille = gray.shape[::-1]   # (largeur, hauteur)

    if taille_image is None:
        taille_image = taille
    elif taille != taille_image:
        print(f"Ignorée (résolution différente) : {f}")
        continue

    trouve, coins = cv2.findChessboardCorners(gray, COINS)
    if not trouve:
        print(f"Damier non trouvé : {f}")
        continue

    # Affine la position des coins à une fraction de pixel
    coins = cv2.cornerSubPix(gray, coins, (11, 11), (-1, -1), critere)
    points_3d.append(objp)
    points_2d.append(coins)

print(f"{len(points_2d)} photos utilisables sur {len(fichiers)}")
if len(points_2d) < 10:
    raise SystemExit("Pas assez de photos exploitables (vise 15 à 20, bien variées).")

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(points_3d, points_2d, taille_image, None, None)

# Erreur de reprojection par photo : on reprojette les coins avec le modèle et on compare aux coins vus
erreurs = []
for i in range(len(points_3d)):
    proj, _ = cv2.projectPoints(points_3d[i], rvecs[i], tvecs[i], K, dist)
    # reshape(-1, 2) : selon la version d'OpenCV, les deux tableaux n'ont pas la même forme (N,2) ou (N,1,2)
    erreurs.append(cv2.norm(points_2d[i].reshape(-1, 2), proj.reshape(-1, 2), cv2.NORM_L2) / np.sqrt(len(proj)))

print(f"\nRésolution : {taille_image[0]} x {taille_image[1]}")
print(f"Erreur RMS globale : {rms:.3f} px  (bon < 0.5 px)")
print(f"Pire photo : {max(erreurs):.3f} px (indice {int(np.argmax(erreurs))})")
print(f"Focale : fx={K[0,0]:.1f}  fy={K[1,1]:.1f} (en pixels)")
print(f"Centre optique : cx={K[0,2]:.1f}  cy={K[1,2]:.1f} (centre de l'image : {taille_image[0]/2:.0f}, {taille_image[1]/2:.0f})")
print(f"Distorsion : {dist.ravel()}")

np.savez(FICHIER_SORTIE, camera_matrix=K, dist_coeffs=dist,
         image_size=np.array(taille_image), rms=rms)
print(f"\nSauvegardé dans {FICHIER_SORTIE}")

# Aperçu : image d'origine à gauche, image corrigée à droite
if AFFICHER_APERCU:
    img = cv2.imread(fichiers[0])
    corrigee = cv2.undistort(img, K, dist)
    duo = np.hstack([cv2.resize(img, (960, 540)), cv2.resize(corrigee, (960, 540))])
    cv2.imshow("original (gauche) | corrigee (droite) - une touche pour fermer", duo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()