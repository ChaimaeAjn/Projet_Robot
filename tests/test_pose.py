"""Tests de estimer_pose et position_sur_plan (6_plateau.py).

Principe : on INVENTE une pose de caméra connue, on calcule avec OpenCV les pixels où
des points du plateau apparaîtraient (cv2.projectPoints), puis on vérifie que les
fonctions du projet retrouvent la position de départ à partir de ces pixels.
"""
import cv2
import numpy as np
import pytest

# 4 tags de référence (id, position en mm) : valeurs fixes, indépendantes de plateau_test.json
REFS = [(0, (25.0, 25.0)), (1, (238.0, 25.0)), (2, (238.0, 172.0)), (3, (25.0, 172.0))]

# Deux poses plateau -> caméra (rvec, tvec) : la caméra est d'un côté, puis de l'autre du plan z = 0.
POSES = {
    "camera_cote_z_negatif": ([0.3, -0.2, 0.1], [-100.0, -80.0, 600.0]),
    "camera_cote_z_positif": ([np.pi - 0.3, 0.2, 0.1], [-100.0, 80.0, 600.0]),
}

TOLERANCE_MM = 0.01


@pytest.fixture(params=list(POSES), ids=list(POSES))
def pose(request): 
    """crée deux poses de caméra connues pour tester les calculs."""
    rvec, tvec = (np.array(v, dtype=float) for v in POSES[request.param])
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec                                   # position réelle de la caméra dans le repère plateau
    return {"rvec": rvec, "tvec": tvec, "R": R, "C": C,
            "cote": 1.0 if C[2] > 0 else -1.0}        # côté du plan où se trouve la caméra


def projeter(calibration, pose, xyz):
    """Transforme les coordonnées 3D d'un point en coordonnées 2D dans l'image."""
    K, dist = calibration
    pts, _ = cv2.projectPoints(np.array([xyz], np.float32), pose["rvec"], pose["tvec"], K, dist)
    return pts.reshape(2)


def estimer(plateau, calibration, pose):
    """Pose estimée par le code du projet à partir des 4 références « vues »."""
    K, dist = calibration
    refs_vues = [(i, xy, projeter(calibration, pose, [*xy, 0.0])) for i, xy in REFS]
    return plateau.estimer_pose(refs_vues, K, dist)


def test_estimer_pose_retrouve_la_camera(plateau, calibration, pose):
    """Vérifie qu'un point posé sur le plateau retrouve ses coordonnées exactes."""
    R, C = estimer(plateau, calibration, pose)

    assert np.allclose(C, pose["C"], atol=TOLERANCE_MM)
    assert np.allclose(R, pose["R"], atol=1e-4)


@pytest.mark.parametrize("xy", [(120.0, 90.0), (30.0, 160.0)])
def test_position_sur_plan_au_niveau_du_plateau(plateau, calibration, pose, xy):
    K, dist = calibration
    R, C = estimer(plateau, calibration, pose)
    pixel = projeter(calibration, pose, [*xy, 0.0])

    trouve = plateau.position_sur_plan(pixel, 0.0, K, dist, R, C)

    assert np.allclose(trouve, xy, atol=TOLERANCE_MM)


def test_position_sur_plan_tag_souleve(plateau, calibration, pose):
    """Vérifie qu'un tag situé à 30 mm au-dessus du plateau est correctement localisé en tenant compte de sa hauteur."""
    K, dist = calibration
    R, C = estimer(plateau, calibration, pose)
    pixel = projeter(calibration, pose, [120.0, 90.0, pose["cote"] * 30.0])

    trouve = plateau.position_sur_plan(pixel, 30.0, K, dist, R, C)

    assert np.allclose(trouve, (120.0, 90.0), atol=TOLERANCE_MM)


def test_ignorer_la_hauteur_decale_le_resultat(plateau, calibration, pose):
    """Vérifie que ne pas prendre en compte la hauteur entraîne une erreur de position."""
    K, dist = calibration
    R, C = estimer(plateau, calibration, pose)
    pixel = projeter(calibration, pose, [120.0, 90.0, pose["cote"] * 30.0])

    sans_correction = plateau.position_sur_plan(pixel, 0.0, K, dist, R, C)

    assert np.hypot(*(sans_correction - (120.0, 90.0))) > 5.0


def test_position_sur_plan_rayon_parallele_au_plan(plateau):
    """Vérifie que le programme renvoie None lorsque le rayon de vision est parallèle au plateau."""
    K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]])
    dist = np.zeros(5)
    R, _ = cv2.Rodrigues(np.array([np.pi / 2, 0.0, 0.0]))

    resultat = plateau.position_sur_plan(np.array([960.0, 540.0]), 0.0, K, dist, R, np.array([0.0, 0.0, -500.0]))

    assert resultat is None


def test_position_sur_plan_plan_derriere_la_camera(plateau, calibration):
    """Vérifie que le programme renvoie None lorsque le plan se trouve derrière la caméra."""
    K, dist = calibration
    R, C = np.eye(3), np.array([0.0, 0.0, -600.0])
    centre_optique = np.array([K[0, 2], K[1, 2]])

    assert plateau.position_sur_plan(centre_optique, 1000.0, K, dist, R, C) is None
    assert plateau.position_sur_plan(centre_optique, 500.0, K, dist, R, C) is not None