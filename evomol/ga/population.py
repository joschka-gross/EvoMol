from dataclasses import dataclass
import random
from typing import Callable
import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import Chem

from .data import PopMember
from .util import to_inchi_key, has_close_match, find_most_similar, tanimoto_similarity

morgan_fp = GetMorganGenerator(radius=2, fpSize=2048)


@dataclass
class Population:
    members: list[PopMember]

    def __len__(self) -> int:
        return len(self.members)

    def drop_duplicates(self) -> "Population":
        return Population(list(set(self.members)))

    def repeat(self, n: int) -> "Population":
        new = []
        for _ in range(n):
            new.extend(self.members)
        return Population(new)

    def inchi_keys(self) -> list[str]:
        return [to_inchi_key(Chem.MolFromSmiles(member.str)) for member in self.members]

    def evaluate(self, evaluator: Callable[[str], float], evaluator_key: str = "score"):
        for member in filter(lambda smi: evaluator_key not in smi, self.members):
            member.set(evaluator_key, evaluator(member.str))

    def sample(self, n: int) -> "Population":
        return Population(random.sample(self.members, n))

    def scores(self):
        return np.array([member.score for member in self.members])

    def extend(self, other: "Population") -> None:
        '''
        Extend the population with the members of another population.\n
        Args:
            other (Population): The population to extend with.
        '''
        self.members.extend(other.members)


def pop_minus(pop1: Population, pop2: Population, method="tanimoto"):
    '''
        Returns a new population with the members of pop1 that are not in pop2.

        Args:
            pop1 (Population): The population to subtract from.
            pop2 (Population): The population to subtract.
            method (str): The method to use for comparison. Can be "inchi" or "tanimoto".
        
        Returns:
            Population: The new population with unique samples.
    '''
    if method == "inchi":
        inchi1 = [to_inchi_key(member.rdkit_mol) for member in pop1.members]
        inchi2 = set([to_inchi_key(member) for member in pop2.members])
        new_members = []
        for member, inchi in zip(pop1.members, inchi1):
            if inchi not in inchi2:
                new_members.append(member)
        return Population(new_members)
    if method == "tanimoto":
        mols1 = [member.rdkit_mol for member in pop1.members]
        mols2 = [member.rdkit_mol for member in pop2.members]
        new_members = []
        for member, mol in zip(pop1.members, mols1):
            if mol is None:
                continue
            if not has_close_match(mol, mols2):
                new_members.append(member)
        return Population(new_members)
    raise ValueError("Invalid method")


def pop_join(*pops: Population):
    '''
        Join multiple populations into a single population.

        Args:
            *pops (Population): The populations to join.
        
        Returns:
            Population: The new population with all members from the input populations.
    '''
    all_smiles = []
    if isinstance(pops[0], list):
        pops = pops[0]
    for pop in pops:
        all_smiles.extend(pop.members)
    return Population(list(set(all_smiles)))


def compare_generated_to_reference(generated: Population, reference_mols):
    msm = [
        find_most_similar(member.rdkit_mol, reference_mols)
        for member in generated.members
    ]
    generated_with_similar = []
    inchilabels = []
    similarities = []
    scores = []
    for j, (member, similar_mol) in enumerate(zip(generated.members, msm)):
        mol_gen = member.rdkit_mol
        generated_with_similar.extend([mol_gen, similar_mol])
        inchilabels.extend([to_inchi_key(mol_gen), to_inchi_key(similar_mol)])
        sim = tanimoto_similarity(
            morgan_fp.GetFingerprint(mol_gen), morgan_fp.GetFingerprint(similar_mol)
        )
        sim = f"{j} | Tanimoto sim: {sim:.2f}"
        similarities.extend([sim, ""])
        scores.extend([str(member.score), ""])
    return generated_with_similar, inchilabels, similarities, scores
