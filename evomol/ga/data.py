from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from rdkit import Chem

from ..molgraphops.actionspace import ActionSpace


@dataclass
class PopMember:
    smiles: str
    values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        assert isinstance(self.smiles, str)

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            if name in self.values:
                return self.values[name]
            raise e

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def set(self, key: str, value: float) -> None:
        self.values[key] = value

    @property
    def str(self):
        return self.smiles

    @cached_property
    def rdkit_mol(self):
        return Chem.MolFromSmiles(self.smiles)


ActionSpaceParameters = ActionSpace.ActionSpaceParameters
SmilesList = list[str]
ParametrizedActionSpace = tuple[list[ActionSpace], ActionSpaceParameters]
