import random
from typing import Callable, Literal
import numpy as np
from copy import deepcopy

from .population import Population
from .data import SmilesList, PopMember, ParametrizedActionSpace
from .mutation import generate_offspring
from .logger import Logger


def replace(
    population: Population,
    offspring: Population,
    index: np.ndarray,
) -> Population:
    do_replace_mask = np.zeros(len(population.members), dtype=bool)
    do_replace_mask[index] = True
    itr_offspring = iter(offspring)
    new_members = []
    for member_index, member in enumerate(population.members):
        if not do_replace_mask[member_index]:
            new_member = member
        else:
            try:
                new_member = next(itr_offspring)
            except StopIteration:
                print("Not enough offspring to replace all members")
                new_member = member
        new_members.append(new_member)
    return Population(new_members)


def extend(population: Population, offspring: SmilesList) -> Population:
    new_members = deepcopy(population.members)
    new_members.extend([PopMember(offspring_member) for offspring_member in offspring])
    return Population(new_members)


def replace_random(
    population: Population,
    offspring: Population,
) -> Population:
    if len(offspring) > len(population.members):
        offspring = offspring.sample(len(population.members))
    index = np.random.choice(
        range(len(population.members)), size=len(offspring), replace=False
    )
    return replace(population, offspring, index)


def replace_weighted_random(
    population: Population,
    offspring: Population,
):
    all_members = population.members + offspring.members
    scores = np.array([member.score for member in all_members])
    mean_score = np.mean(scores)
    if np.allclose(scores, mean_score):
        weigths = np.ones_like(scores)
    else:
        std = np.std(scores)
        if np.isnan(std):
            std = 1.0
        np.clip(std, 1e-6, 1e6)
        weigths = np.exp((scores - mean_score) / std)
    new_members = list(
        np.random.choice(
            all_members,
            size=len(population.members),
            p=weigths / np.sum(weigths),
        )
    )
    return Population(new_members)


def insert_best_k(population: Population, offspring: Population) -> Population: ...


def run_ga(
    population: Population,
    parametrized_action_space: ParametrizedActionSpace,
    nsteps: int,
    logger: Logger,
    evaluator: Callable | None = None,
    generation_depth: int = 3,
    num_generation_tries: int = 10,
    obejctive: Literal["min", "max"] = "max",
):
    if evaluator is None:
        evaluator = lambda x: 1.0
    if obejctive == "min":
        evaluator = lambda x: -evaluator(x)
    logger.step(population, step_id=0)
    population.evaluate(evaluator)
    for step_id in range(nsteps):
        offspring = generate_offspring(
            population,
            parametrized_action_space,
            evaluator,
            action_space_depth=generation_depth,
            max_tries_per_member=num_generation_tries,
        )
        population = replace_weighted_random(population, offspring)
        logger.step(population, step_id=step_id)
