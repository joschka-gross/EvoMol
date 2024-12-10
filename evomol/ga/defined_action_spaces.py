from ..molgraphops.default_actionspaces import generic_action_space
from .data import ParametrizedActionSpace

action_space, params = generic_action_space(
    ["Cl", "F", "O", "Br", "N"],
    30,
    append_atom=False,
    remove_atom=False,
    change_bond=True,
    substitution=True,
    move_group=True,
    change_bond_prevent_breaking_creating_bonds=False,
    cut_insert=True,
    substitute_atoms_with_non_carbon_neighbors=False,
)
params.accepted_substitutions = {
    "Cl": ["F", "O", "Br", "N"],
    "F": ["O", "Br", "N"],
    "O": ["F", "Br", "N"],
    "Br": ["F", "O", "N"],
    "N": ["F", "O", "Br"],
}
ComplexActionSpace: ParametrizedActionSpace = (action_space, params)

action_space, params = generic_action_space(
    ["Cl", "F", "O", "Br", "N"],
    30,
    append_atom=False,
    remove_atom=False,
    change_bond=False,
    substitution=True,
    move_group=True,
    change_bond_prevent_breaking_creating_bonds=True,
    cut_insert=False,
    substitute_atoms_with_non_carbon_neighbors=False,
)
params.accepted_substitutions = {
    "Cl": ["F", "O", "Br", "N"],
    "F": ["O", "Br", "N"],
    "O": ["F", "Br", "N"],
    "Br": ["F", "O", "N"],
    "N": ["F", "O", "Br"],
}
SubstitutionMovingOnlyActionSpace = (action_space, params)
