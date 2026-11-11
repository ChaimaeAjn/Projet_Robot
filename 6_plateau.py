import json
import time
import cv2
import numpy as np

# --- PARAMÈTRES ---
CONFIG = "plateau_test.json"
CALIBRATION = "calibration.npz"
CAMERA_INDEX = 1
SEUIL_ERREUR_MM = 10.0           # au-delà, l'erreur sur les tags de référence s'affiche en rouge
AFFICHER_SANS_CORRECTION = True  # affiche aussi la position obtenue en ignorant la hauteur du tag
GARDER_DERNIERE_POSE = True      # si moins de 4 références sont visibles, garde la dernière pose connue
# ------------------


def charger_config(chemin):
    with open(chemin, encoding="utf-8") as f:
        cfg = json.load(f)
    # Les clés JSON sont du texte : on les convertit en entiers (identifiants de tags)
    cfg["references"] = {int(k): tuple(v) for k, v in cfg["references"].items()}
    cfg["hauteurs_mm"] = {int(k): float(v) for k, v in cfg.get("hauteurs_mm", {}).items()}
    cfg["hauteur_par_defaut_mm"] = float(cfg.get("hauteur_par_defaut_mm", 0.0))
    return cfg


def centre_du_tag(coins_tag):
    return coins_tag.reshape(4, 2).mean(axis=0)


def estimer_pose(refs_vues, K, dist):
    """Pose de la caméra dans le repère du plateau, à partir des tags de référence visibles.
    refs_vues : liste de (id, (x_mm, y_mm), centre_en_pixels). Renvoie (R, C) ou None.
    R : rotation repère plateau -> repère caméra ; C : position de la caméra dans le repère plateau (mm)."""
    obj = np.array([[x, y, 0.0] for _, (x, y), _ in refs_vues], np.float32)   # références : z = 0 par définition
    img = np.array([c for _, _, c in refs_vues], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)  # IPPE : points tous dans un même plan
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec).ravel()
    return R, C


def position_sur_plan(centre_px, hauteur_mm, K, dist, R, C):
    """Position (x, y) en mm d'un point vu au pixel centre_px, sachant qu'il est à hauteur_mm
    au-dessus du plan des références. On intersecte le rayon de vue avec le plan z = hauteur."""
    pt = np.array([[centre_px]], dtype=np.float32)
    n = cv2.undistortPoints(pt, K, dist).reshape(2)   # coordonnées normalisées : distorsion corrigée, K retirée
    d = R.T @ np.array([n[0], n[1], 1.0])             # direction du rayon dans le repère plateau
    # Le repère est défini par les références : l'axe z peut pointer vers la caméra ou à l'opposé.
    # "Au-dessus du plan" = du côté de la caméra, donc le signe de z se déduit de la position de la caméra.
    signe = 1.0 if C[2] > 0 else -1.0
    z_cible = signe * hauteur_mm
    if abs(d[2]) < 1e-9:
        return None
    lam = (z_cible - C[2]) / d[2]
    if lam <= 0:                                      # l'intersection serait derrière la caméra
        return None
    return (C + lam * d)[:2]


def analyser_image(frame, detector, cfg, K, dist, pose_precedente=None):
    coins, ids, _ = detector.detectMarkers(frame)

    tags = {}
    if ids is not None:
        for c, i in zip(coins, ids.flatten()):
            tags[int(i)] = centre_du_tag(c)

    refs_vues = [(i, cfg["references"][i], tags[i]) for i in tags if i in cfg["references"]]

    pose, pose_a_jour = pose_precedente, False
    if len(refs_vues) >= 4:
        nouvelle = estimer_pose(refs_vues, K, dist)
        if nouvelle is not None:
            pose, pose_a_jour = nouvelle, True
    if not pose_a_jour and not GARDER_DERNIERE_POSE:
        pose = None

    suivis, erreurs_refs = [], []
    if pose is not None:
        R, C = pose
        # Contrôle : on recalcule la position des références avec la pose et on compare à la config
        for i, (x, y), c in refs_vues:
            p = position_sur_plan(c, 0.0, K, dist, R, C)
            if p is not None:
                erreurs_refs.append(float(np.hypot(p[0] - x, p[1] - y)))
        # Tags suivis : tous ceux qui ne sont pas des références
        for i, c in tags.items():
            if i in cfg["references"]:
                continue
            h = cfg["hauteurs_mm"].get(i, cfg["hauteur_par_defaut_mm"])
            suivis.append({"id": i, "centre_px": c, "hauteur_mm": h,
                           "xy": position_sur_plan(c, h, K, dist, R, C),
                           "xy_sans_correction": position_sur_plan(c, 0.0, K, dist, R, C)})

    return {"coins": coins, "ids": ids, "nb_refs_vues": len(refs_vues), "pose": pose,
            "pose_a_jour": pose_a_jour, "suivis": suivis, "erreurs_refs": erreurs_refs}


def main():
    cfg = charger_config(CONFIG)
    calib = np.load(CALIBRATION)
    K, dist = calib["camera_matrix"], calib["dist_coeffs"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionnaire"]))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la caméra. Essaie un autre CAMERA_INDEX.")

    taille_ok = None
    pose, t_prec, fps = None, time.perf_counter(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Plus d'image reçue.")
            break

        if taille_ok is None:   # la calibration n'est valable que pour la résolution utilisée pour la faire
            taille_ok = tuple(int(v) for v in calib["image_size"]) == (frame.shape[1], frame.shape[0])
            if not taille_ok:
                print("ATTENTION : la résolution de la caméra n'est pas celle de la calibration !")

        res = analyser_image(frame, detector, cfg, K, dist, pose)
        pose = res["pose"]

        if res["ids"] is not None:
            cv2.aruco.drawDetectedMarkers(frame, res["coins"], res["ids"])

        for s in res["suivis"]:
            cx, cy = int(s["centre_px"][0]), int(s["centre_px"][1])
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            if s["xy"] is not None:
                cv2.putText(frame, f"{s['id']}: ({s['xy'][0]:.0f}, {s['xy'][1]:.0f}) mm", (cx + 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if AFFICHER_SANS_CORRECTION and s["xy_sans_correction"] is not None:
                cv2.putText(frame, f"sans corr.: ({s['xy_sans_correction'][0]:.0f}, {s['xy_sans_correction'][1]:.0f})",
                            (cx + 12, cy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        t = time.perf_counter()
        fps = 1.0 / (t - t_prec) if fps == 0 else 0.9 * fps + 0.1 * (1.0 / (t - t_prec))
        t_prec = t

        cv2.putText(frame, f"{fps:.0f} FPS | references vues: {res['nb_refs_vues']}/{len(cfg['references'])}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if pose is None:
            cv2.putText(frame, "pas de pose : il faut au moins 4 references visibles", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            C = pose[1]
            memo = "" if res["pose_a_jour"] else " (pose memorisee)"
            cv2.putText(frame, f"camera: hauteur {abs(C[2]):.0f} mm, sous la camera ({C[0]:.0f}, {C[1]:.0f}){memo}",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if res["erreurs_refs"]:
                e_moy, e_max = np.mean(res["erreurs_refs"]), np.max(res["erreurs_refs"])
                couleur = (0, 255, 0) if e_max < SEUIL_ERREUR_MM else (0, 0, 255)
                cv2.putText(frame, f"erreur sur les references: moy {e_moy:.1f} mm, max {e_max:.1f} mm",
                            (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)

        cv2.imshow("plateau", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()