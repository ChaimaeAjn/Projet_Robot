"""Tests de charger_config et centre_du_tag (6_plateau.py)."""
import json

import numpy as np
import pytest


def test_charger_config_fichier_reel(plateau, racine):
    """Teste le chargement du vrai fichier plateau_test.json.

        Il vérifie que :
        - Le dictionnaire utilisé est correct.
        - Les références sont bien converties en nombres entiers.
        - Les coordonnées sont converties en tuples.
        - Les hauteurs sont vides et la hauteur par défaut vaut 0.0."""
    cfg = plateau.charger_config(racine / "plateau_test.json")

    assert cfg["dictionnaire"] == "DICT_4X4_50"
    assert set(cfg["references"]) == {0, 1, 2, 3}          # clés converties en int
    assert cfg["references"][0] == (24.6, 24.6)            # valeurs converties en tuple
    assert cfg["hauteurs_mm"] == {}
    assert cfg["hauteur_par_defaut_mm"] == 0.0


def test_charger_config_conversion_des_types(plateau, tmp_path):
    """Crée un petit fichier JSON temporaire pour vérifier les conversions de types.

        Il vérifie que :
        - Les clés "7" et "5" deviennent des entiers.
        - Les coordonnées deviennent des tuples.
        - Les hauteurs deviennent des nombres décimaux (float)."""
    fichier = tmp_path / "cfg.json"
    fichier.write_text(json.dumps({
        "dictionnaire": "DICT_4X4_50",
        "references": {"7": [10, 20]},
        "hauteurs_mm": {"5": 30},
        "hauteur_par_defaut_mm": 12,
    }))

    cfg = plateau.charger_config(fichier)

    assert cfg["references"] == {7: (10, 20)}
    assert cfg["hauteurs_mm"] == {5: 30.0}
    assert isinstance(cfg["hauteurs_mm"][5], float)
    assert isinstance(cfg["hauteur_par_defaut_mm"], float)


def test_charger_config_valeurs_par_defaut(plateau, tmp_path):
    """Teste le comportement lorsque certaines informations sont absentes du fichier JSON."""
    fichier = tmp_path / "cfg.json"
    fichier.write_text(json.dumps({"dictionnaire": "DICT_4X4_50", "references": {"0": [0, 0]}}))

    cfg = plateau.charger_config(fichier)

    assert cfg["hauteurs_mm"] == {}
    assert cfg["hauteur_par_defaut_mm"] == 0.0


def test_charger_config_sans_references(plateau, tmp_path):
    """Vérifie que le programme réagit correctement si la clé obligatoire references est absente.

       Il vérifie qu'une erreur KeyError est déclenchée."""
    fichier = tmp_path / "cfg.json"
    fichier.write_text(json.dumps({"dictionnaire": "DICT_4X4_50"}))

    with pytest.raises(KeyError):
        plateau.charger_config(fichier)


def test_centre_du_tag_format_opencv(plateau):
    """Teste le calcul du centre d'un tag avec le format de coordonnées utilisé par OpenCV.

        Il vérifie que :
        - Le centre possède deux coordonnées (x et y).
        - Le centre calculé est bien (120, 220)."""
    coins = np.array([[[100, 200], [140, 200], [140, 240], [100, 240]]], dtype=np.float32)

    centre = plateau.centre_du_tag(coins)

    assert centre.shape == (2,)
    assert np.allclose(centre, [120, 220])


def test_centre_du_tag_quadrilatere_non_carre(plateau):
    """Teste le calcul du centre d'un tag déformé par la perspective.

        Il vérifie que le programme calcule bien la moyenne des quatre coins, même si le tag n'est pas parfaitement carré."""
    coins = np.array([[[0, 0], [10, 0], [8, 10], [0, 6]]], dtype=np.float32)

    centre = plateau.centre_du_tag(coins)

    assert np.allclose(centre, [4.5, 4.0])