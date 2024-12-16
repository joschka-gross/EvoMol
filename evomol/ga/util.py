import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs, Chem

morgan_fp_generator = GetMorganGenerator(radius=3, fpSize=2048)


def to_inchi_key(mol):
    return Chem.MolToInchiKey(mol)


def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def find_most_similar(mol: Chem.Mol, reference: list[Chem.Mol]):
    fp = morgan_fp_generator.GetFingerprint(mol)
    reference_fps = [morgan_fp_generator.GetFingerprint(m) for m in reference]
    similarities = [tanimoto_similarity(fp, ref_fp) for ref_fp in reference_fps]
    max_index = np.argmax(similarities)
    return reference[max_index]


def has_close_match(mol: Chem.Mol, reference: list[Chem.Mol]):
    if(len(reference) == 0):
        return False
    
    fp = morgan_fp_generator.GetFingerprint(mol)
    reference_fps = [morgan_fp_generator.GetFingerprint(m) for m in reference]
    similarities = [tanimoto_similarity(fp, ref_fp) for ref_fp in reference_fps]
    return np.isclose(max(similarities), 1.0)
