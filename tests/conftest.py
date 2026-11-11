"""Fixtures partagées par tous les tests du dossier tests/."""
import importlib.util
from pathlib import Path
import numpy as np

import pytest

# Racine du dépôt = dossier parent de tests/
RACINE = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def racine():
    """Chemin de la racine du dépôt (là où se trouvent les scripts et les images)."""
    return RACINE


@pytest.fixture(scope="session")
def plateau():
    """Charge 6_plateau.py comme un module.

    Son nom commence par un chiffre : `import 6_plateau` est impossible en Python,
    on passe donc par importlib. Le chargement n'ouvre aucune caméra
    (main() est protégé par `if __name__ == "__main__"`).
    """
    spec = importlib.util.spec_from_file_location("plateau", RACINE / "6_plateau.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope="session")
def calibration():
    """Matrice caméra K et coefficients de distorsion lus dans calibration.npz."""
    with np.load(RACINE / "calibration.npz") as donnees:
        return donnees["camera_matrix"], donnees["dist_coeffs"]