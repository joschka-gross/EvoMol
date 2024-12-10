from evomol.molgraphops.molgraph import MolGraph
from rdkit import Chem


def test_aspirin_co():
    aspirin = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    mo = MolGraph(aspirin)
    assert set(mo.get_neighbor_elements(1)) == {"C", "O"}
