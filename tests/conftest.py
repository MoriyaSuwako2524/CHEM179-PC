import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def ref():
    def load(name):
        p = DATA / f"ref_{name}.npz"
        if not p.exists():
            pytest.skip(f"missing {p.name}; run tools/make_reference.py where PySCF is installed")
        return dict(np.load(p))
    return load


@pytest.fixture(scope="session")
def ref_energies():
    return json.loads((DATA / "reference_energies.json").read_text())
