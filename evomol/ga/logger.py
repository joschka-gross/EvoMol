from typing import Any

from evomol.ga.population import Population


class Logger:

    def __init__(
        self,
    ) -> None:
        self.populations = []

    def step(self, population: Population, step_id: int):
        self.populations.append(population)
