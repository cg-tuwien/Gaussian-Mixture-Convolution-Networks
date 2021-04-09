import os
import typing
import torch
import torch.utils.tensorboard

import gmc.mixture as gm
import gmc.render
import gmc.cpp.gm_vis.gm_vis as gm_vis
from qm9.config import Config

class AtomData:
    def __init__(self, x: float, y: float, z: float, mulliken_charge: float):
        self.x = x
        self.y = y
        self.z = z
        self.mulliken_charge = mulliken_charge


# https://en.wikipedia.org/wiki/Atomic_radius
def atomic_radius_empirical(atom_type: str):
    if atom_type == "H":
        return 0.25
    if atom_type == "C":
        return 0.7
    if atom_type == "N":
        return 0.65
    if atom_type == "O":
        return 0.6
    if atom_type == "F":
        return 0.5
    assert False


# https://en.wikipedia.org/wiki/Atomic_radius
def atomic_radius_calculated(atom_type: str):
    if atom_type == "H":
        return 0.53
    if atom_type == "C":
        return 0.48
    if atom_type == "N":
        return 0.56
    if atom_type == "O":
        return 0.48
    if atom_type == "F":
        return 0.42
    assert False


# https://en.wikipedia.org/wiki/Standard_atomic_weight
def atomic_radius_covalent(atom_type: str):
    if atom_type == "H":
        return 0.37
    if atom_type == "C":
        return 0.77
    if atom_type == "N":
        return 0.75
    if atom_type == "O":
        return 0.73
    if atom_type == "F":
        return 0.71
    assert False


# https://en.wikipedia.org/wiki/Standard_atomic_weight
def atomic_weight(atom_type: str):
    if atom_type == "H":
        return 1.008
    if atom_type == "C":
        return 12.011
    if atom_type == "N":
        return 14.007
    if atom_type == "O":
        return 15.999
    if atom_type == "F":
        return 18.998
    assert False


class Molecule:
    ATOM_TYPES = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    N_ATOMS_MAX_PER_TYPE = 20

    def __init__(self, atoms: typing.Dict[str, typing.Set[AtomData]], properties: typing.Dict[str, float]):
        self.atoms = atoms
        self.properties = properties

    def as_gaussian_mixture(self):
        weights = torch.zeros(1, len(Molecule.ATOM_TYPES), Molecule.N_ATOMS_MAX_PER_TYPE, 1)
        positions = torch.zeros(1, len(Molecule.ATOM_TYPES), Molecule.N_ATOMS_MAX_PER_TYPE, 3)
        radii = torch.ones(1, len(Molecule.ATOM_TYPES), Molecule.N_ATOMS_MAX_PER_TYPE, 1)

        for t, atom_data in self.atoms.items():
            t_id = Molecule.ATOM_TYPES[t]
            for g_id, d in enumerate(atom_data):
                weights[0, t_id, g_id, 0] = 0.1 # math.sqrt(atomic_weight(t))
                positions[0, t_id, g_id, 0] = d.x
                positions[0, t_id, g_id, 1] = d.y
                positions[0, t_id, g_id, 2] = d.z
                radii[0, t_id, g_id, 0] = atomic_radius_empirical(t)

        covariance = torch.diag(torch.ones(3)).view(1, 1, 1, 3, 3) * (radii.unsqueeze(-1) * 2) ** 2
        return gm.pack_mixture(weights, positions, covariance)


# the qm9 dataset doesn't have actual xyz files. there is a fourth atom column, which is also read here.
def read_dataset(config: Config) -> typing.List[Molecule]:
    atom_types = set()
    n_atoms_max = 0
    n_atoms_max_per_type = 0

    molecules = list()

    for filename in os.listdir(config.qm9_data_path):
        with open(f"{config.qm9_data_path}/{filename}") as f:
            # https://www.nature.com/articles/sdata201422/tables/3
            n_atoms = int(f.readline())
            n_atoms_max = max(n_atoms, n_atoms_max)

            properties_raw = f.readline().split('\t')
            assert len(properties_raw) == 17
            # https://www.nature.com/articles/sdata201422/tables/4
            properties = {"id": (properties_raw[0].split()[1]),     # gdb 12345
                          "rotA": float(properties_raw[1]),
                          "rotB": float(properties_raw[2]),
                          "rotC": float(properties_raw[3]),
                          "mu": float(properties_raw[4]),
                          "alpha": float(properties_raw[5]),
                          "e_homo": float(properties_raw[6]),
                          "e_lumo": float(properties_raw[7]),
                          "e_gap": float(properties_raw[8]),
                          "Rsq": float(properties_raw[9]),
                          "zpve": float(properties_raw[10]),
                          "U0": float(properties_raw[11]),
                          "U": float(properties_raw[12]),
                          "H": float(properties_raw[13]),
                          "G": float(properties_raw[14]),
                          "Cv": float(properties_raw[15])}

            atom_types_of_this = dict()
            atoms = dict()
            for i in range(n_atoms):
                atom_properties = f.readline().split('\t')
                assert len(atom_properties) == 5
                atom_type = atom_properties[0]
                atom_x = float(atom_properties[1].replace("*^", "e"))   # this file format uses a funny floating point format: '9.4537*^-6'
                atom_y = float(atom_properties[2].replace("*^", "e"))
                atom_z = float(atom_properties[3].replace("*^", "e"))
                atom_mulliken_charge = float(atom_properties[4].replace("*^", "e"))

                atom_types.add(atom_type)
                if atom_type not in atom_types_of_this:
                    atom_types_of_this[atom_type] = 0
                    atoms[atom_type] = set()

                atom_types_of_this[atom_type] = atom_types_of_this[atom_type] + 1
                atoms[atom_type].add(AtomData(atom_x, atom_y, atom_z, atom_mulliken_charge))

            for c in atom_types_of_this.values():
                n_atoms_max_per_type = max(n_atoms_max_per_type, c)

            molecules.append(Molecule(atoms, properties))
    print(f"n atom_types = {len(atom_types)}, atom_types = {atom_types}, n_atoms_max = {n_atoms_max}, n_atoms_max_per_type = {n_atoms_max_per_type}, len(molecules) = {len(molecules)}")
    return molecules


def render_dataset(molecules: typing.List[Molecule], tensor_board: torch.utils.tensorboard.SummaryWriter, N: int = 100):
    images = list()
    n = 0
    vis = gm_vis.GMVisualizer(False, 150, 150)
    vis.set_density_rendering(True)
    # vis.set_camera_auto(True)
    vis.set_camera_lookat(positions=(10, 10, 5), lookat=(0, 0, 0), up=(0, 0, 1))
    vis.set_density_range_manual(0, 5)

    for molecule in molecules:
        rendering = gmc.render.render3d(molecule.as_gaussian_mixture(), gm_vis_object=vis)
        images.append(rendering)
        n = n + 1
        if n % 10 == 0 or n >= N:
            images = torch.cat(images, dim=1)
            tensor_board.add_image("molecule", images[:, :, :3], n, dataformats='HWC')
            images = list()
        if n >= N:
            break

    vis.finish()
