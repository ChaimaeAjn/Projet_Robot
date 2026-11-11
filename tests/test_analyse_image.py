"""Tests de analyser_image (6_plateau.py) : la logique de décision, sans caméra.

On remplace le détecteur ArUco par un faux détecteur qui « voit » exactement les tags
qu'on lui indique. On ne teste donc pas la détection d'OpenCV (voir test_detection_tags.py),
mais ce que analyser_image fait des tags détectés.
"""
import cv2
import numpy as np
import pytest

REFS = {0: (25.0, 25.0), 1: (238.0, 25.0), 2: (238.0, 172.0), 3: (25.0, 172.0)}

RVEC = np.array([0.3, -0.2, 0.1])
TVEC = np.array([-100.0, -80.0, 600.0])
R_VRAIE, _ = cv2.Rodrigues(RVEC)
C_VRAIE = -R_VRAIE.T @ TVEC
COTE = 1.0 if C_VRAIE[2] > 0 else -1.0

TOLERANCE_MM = 0.01
IMAGE_VIDE = np.zeros((1080, 1920, 3), np.uint8)


class DetecteurFactice:
    """Même interface que cv2.aruco.ArucoDetector, mais renvoie les tags qu'on lui donne."""

    def __init__(self, centres):
        # Initialise le faux détecteur avec les centres des tags fournis.
        self.centres = centres

    def detectMarkers(self, image):
        # Simule la détection des tags et renvoie leurs coins et leurs identifiants.
        if not self.centres:
            return (), None, ()
        ids = np.array([[i] for i in self.centres], dtype=np.int32)
        coins = [np.array([[[x - 20, y - 20], [x + 20, y - 20], [x + 20, y + 20], [x - 20, y + 20]]],
                          np.float32) for x, y in self.centres.values()]
        return coins, ids, ()


def pixel(calibration, xyz):
    # Convertit la position 3D d'un point du plateau en coordonnées pixels.
    K, dist = calibration
    pts, _ = cv2.projectPoints(np.array([xyz], np.float32), RVEC, TVEC, K, dist)
    return tuple(pts.reshape(2))


def refs_vues(calibration, ids):
    # Calcule les pixels correspondant aux tags de référence demandés.
    return {i: pixel(calibration, [*REFS[i], 0.0]) for i in ids}


@pytest.fixture
def cfg():
    # Fournit une configuration de test avec les références et les hauteurs des tags.
    return {"references": dict(REFS), "hauteurs_mm": {5: 30.0}, "hauteur_par_defaut_mm": 10.0}


def analyser(plateau, calibration, cfg, centres, pose_precedente=None):
    # Appelle analyser_image avec le faux détecteur et les paramètres du test.
    K, dist = calibration
    return plateau.analyser_image(IMAGE_VIDE, DetecteurFactice(centres), cfg, K, dist, pose_precedente)


def test_quatre_references_donnent_une_pose(plateau, calibration, cfg):
    # Vérifie que quatre références permettent de retrouver correctement la pose de la caméra.
    res = analyser(plateau, calibration, cfg, refs_vues(calibration, [0, 1, 2, 3]))

    assert res["nb_refs_vues"] == 4
    assert res["pose_a_jour"] is True
    assert np.allclose(res["pose"][1], C_VRAIE, atol=TOLERANCE_MM)
    assert len(res["erreurs_refs"]) == 4 and max(res["erreurs_refs"]) < TOLERANCE_MM
    assert res["suivis"] == []


def test_tag_suivi_utilise_la_hauteur_de_la_config(plateau, calibration, cfg):
    # Vérifie que la hauteur définie dans la configuration est utilisée pour localiser un tag.
    centres = {**refs_vues(calibration, [0, 1, 2, 3]), 5: pixel(calibration, [120.0, 90.0, COTE * 30.0])}
    res = analyser(plateau, calibration, cfg, centres)

    assert [s["id"] for s in res["suivis"]] == [5]
    assert res["suivis"][0]["hauteur_mm"] == 30.0
    assert np.allclose(res["suivis"][0]["xy"], (120.0, 90.0), atol=TOLERANCE_MM)


def test_tag_suivi_sans_hauteur_utilise_la_hauteur_par_defaut(plateau, calibration, cfg):
    # Vérifie que la hauteur par défaut est utilisée si aucune hauteur spécifique n'est définie.
    centres = {**refs_vues(calibration, [0, 1, 2, 3]), 6: pixel(calibration, [60.0, 100.0, COTE * 10.0])}
    res = analyser(plateau, calibration, cfg, centres)

    assert res["suivis"][0]["hauteur_mm"] == 10.0
    assert np.allclose(res["suivis"][0]["xy"], (60.0, 100.0), atol=TOLERANCE_MM)


def test_trois_references_gardent_la_derniere_pose(plateau, calibration, cfg):
    # Vérifie que la dernière pose est conservée lorsqu'il ne reste que trois références.
    tag5 = {5: pixel(calibration, [120.0, 90.0, COTE * 30.0])}
    premiere = analyser(plateau, calibration, cfg, {**refs_vues(calibration, [0, 1, 2, 3]), **tag5})
    res = analyser(plateau, calibration, cfg, {**refs_vues(calibration, [0, 1, 2]), **tag5},
                   pose_precedente=premiere["pose"])

    assert res["nb_refs_vues"] == 3
    assert res["pose_a_jour"] is False
    assert res["pose"] is premiere["pose"]
    assert np.allclose(res["suivis"][0]["xy"], (120.0, 90.0), atol=TOLERANCE_MM)


def test_trois_references_sans_garder_la_pose(plateau, calibration, cfg, monkeypatch):
    # Vérifie que la pose précédente est abandonnée si sa conservation est désactivée.
    monkeypatch.setattr(plateau, "GARDER_DERNIERE_POSE", False)
    tag5 = {5: pixel(calibration, [120.0, 90.0, COTE * 30.0])}
    premiere = analyser(plateau, calibration, cfg, {**refs_vues(calibration, [0, 1, 2, 3]), **tag5})
    res = analyser(plateau, calibration, cfg, {**refs_vues(calibration, [0, 1, 2]), **tag5},
                   pose_precedente=premiere["pose"])

    assert res["pose"] is None
    assert res["suivis"] == []


def test_aucun_tag_detecte(plateau, calibration, cfg):
    # Vérifie le comportement du programme lorsqu'aucun tag n'est détecté.
    res = analyser(plateau, calibration, cfg, {})

    assert res["ids"] is None
    assert res["nb_refs_vues"] == 0
    assert res["pose"] is None
    assert res["suivis"] == []