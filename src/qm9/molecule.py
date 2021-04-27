import typing

import torch

import gmc.mixture as gm
import qm9.data_constants as data_constants

class AtomData:
    def __init__(self, x: float, y: float, z: float, mulliken_charge: float):
        self.x = x
        self.y = y
        self.z = z
        self.mulliken_charge = mulliken_charge


class Molecule:
    ATOM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    N_ATOMS_MAX_PER_TYPE = 20

    def __init__(self, atoms: typing.Dict[str, typing.Set[AtomData]], properties: typing.Dict[str, float]):
        self.atoms = atoms
        self.properties = properties

    @staticmethod
    def atomic_radii_as_tensor():
        r1 = data_constants.atomic_radius_empirical
        r2 = data_constants.atomic_radius_covalent
        return torch.tensor([[r1('H'), r1('C'), r1('N'), r1('O'), r1('F')],
                             [r2('H'), r2('C'), r2('N'), r2('O'), r2('F')]])

    def as_gaussian_mixture(self, config, atom_weights: torch.Tensor, atom_radii: torch.Tensor) -> torch.Tensor:
        device = atom_weights.device
        weights = torch.zeros(1, len(Molecule.ATOM_TYPES) * (1 + int(config.heavy)), Molecule.N_ATOMS_MAX_PER_TYPE, 1, device=device)
        positions = torch.zeros(1, len(Molecule.ATOM_TYPES) * (1 + int(config.heavy)), Molecule.N_ATOMS_MAX_PER_TYPE, 3, device=device)
        radii = torch.ones(1, len(Molecule.ATOM_TYPES) * (1 + int(config.heavy)), Molecule.N_ATOMS_MAX_PER_TYPE, 1, device=device)

        for t, atom_data in self.atoms.items():
            t_id = Molecule.ATOM_TYPES[t]
            for g_id, d in enumerate(atom_data):
                weights[0, t_id, g_id, 0] = atom_weights[0, t_id]  # math.sqrt(data_constants.atomic_weight(t))
                positions[0, t_id, g_id, 0] = d.x
                positions[0, t_id, g_id, 1] = d.y
                positions[0, t_id, g_id, 2] = d.z
                radii[0, t_id, g_id, 0] = atom_radii[0, t_id]

                if config.heavy:
                    weights[0, t_id + len(Molecule.ATOM_TYPES), g_id, 0] = atom_weights[1, t_id]  # math.sqrt(data_constants.atomic_weight(t))
                    positions[0, t_id + len(Molecule.ATOM_TYPES), g_id, 0] = d.x
                    positions[0, t_id + len(Molecule.ATOM_TYPES), g_id, 1] = d.y
                    positions[0, t_id + len(Molecule.ATOM_TYPES), g_id, 2] = d.z
                    radii[0, t_id + len(Molecule.ATOM_TYPES), g_id, 0] = atom_radii[1, t_id]

        covariance = torch.diag(torch.ones(3, device=device)).view(1, 1, 1, 3, 3) * (radii.unsqueeze(-1) * 2) ** 2
        return gm.pack_mixture(weights, positions, covariance)
