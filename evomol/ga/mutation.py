from typing import Callable
from .population import Population
from .data import SmilesList
import random
from joblib import Parallel, delayed

from ..molgraphops.molgraph import MolGraph, MolGraphBuilder
from evomol.molgraphops.exploration import RandomActionTypeSelectionStrategy
from .data import ParametrizedActionSpace

random_generation = RandomActionTypeSelectionStrategy()


def select_for_mutation(population: Population) -> SmilesList:
    k = min(10, len(population.members) // 2)
    return random.choices(population.members, k=k)


def generate_offspring(
    population: Population,
    param_action_space: ParametrizedActionSpace,
    evaluator: Callable,
    max_tries_per_member: int = 10,
    max_offspring_per_member: int = 3,
    action_space_depth: int = 3,
    num_processes: int = 1,
) -> SmilesList:
    action_space, params = param_action_space

    def generate_neighbours(member):
        mo = MolGraph(member.rdkit_mol)
        builder = MolGraphBuilder(params, action_space, mo)
        offspring = []
        for _ in range(max_tries_per_member):
            try:
                mutated, _ = random_generation.generate_neighbour(
                    builder, action_space_depth, None
                )
            except IndexError:
                continue
            offspring.append(mutated)
            if len(offspring) >= max_offspring_per_member:
                break
        return offspring

    offspring = Parallel(n_jobs=num_processes)(
        delayed(generate_neighbours)(member)
        for member in select_for_mutation(population)
    )
    offspring = [item for sublist in offspring for item in sublist]
    return offspring
