import random

import os
import typing
import torch
import torch.utils.tensorboard
import torch.utils.data

import gmc.mixture as gm
import gmc.render
import gmc.cpp.gm_vis.gm_vis as gm_vis
from qm9.config import Config
from qm9.exclude_list import exclude_list
import qm9.data_constants as data_constants

# nature article: https://www.nature.com/articles/sdata201422
# download here: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
#                (some download links don't work. remove %XX codes and points, e.g. link.com/Blah%3A_bluh_H%2C_bleh./something => link.com/Blah_bluh_H_bleh/something)
# sota benchmarks: https://paperswithcode.com/dataset/qm9
# description of the dataset from a university course: https://notebook.community/beangoben/dataDrivenChemistry/mainCode

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

    def as_gaussian_mixture(self):
        weights = torch.zeros(1, len(Molecule.ATOM_TYPES), Molecule.N_ATOMS_MAX_PER_TYPE, 1)
        positions = torch.zeros(1, len(Molecule.ATOM_TYPES), Molecule.N_ATOMS_MAX_PER_TYPE, 3)
        radii = torch.ones(1, len(Molecule.ATOM_TYPES), Molecule.N_ATOMS_MAX_PER_TYPE, 1)

        for t, atom_data in self.atoms.items():
            t_id = Molecule.ATOM_TYPES[t]
            for g_id, d in enumerate(atom_data):
                weights[0, t_id, g_id, 0] = 0.1 # math.sqrt(data_constants.atomic_weight(t))
                positions[0, t_id, g_id, 0] = d.x
                positions[0, t_id, g_id, 1] = d.y
                positions[0, t_id, g_id, 2] = d.z
                radii[0, t_id, g_id, 0] = data_constants.atomic_radius_empirical(t)

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

            if int(properties["id"]) in exclude_list:
                continue

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

            # subtract reference energies, same as
            for prop in ("zpve", "U0", "U", "H", "G", "Cv"):
                ref_energy = 0
                for t, c in atom_types_of_this.items():
                    ref_energy += data_constants.REFERENCE_THERMOCHEMICAL_ENERGIES[t][prop] * data_constants.HARTREE * c
                properties[prop] -= ref_energy

            # convert to eV
            for prop in ("e_homo", "e_lumo", "e_gap", "zpve", "U0", "U", "H", "G"):
                properties[prop] = properties[prop] * data_constants.HARTREE

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


class DataSet(torch.utils.data.Dataset):
    def __init__(self, config: Config, start_index: int, end_index: int):
        data = read_dataset(config)
        random.Random(0).shuffle(data)
        data = data[start_index:end_index]
        gm_data = list()
        self.targets = list()
        for d in data:
            gm_data.append(d.as_gaussian_mixture())
            self.targets.append(torch.tensor(d.properties[config.inference_on], dtype=torch.float))

        gm_data = torch.cat(gm_data, 0)
        # # needs an edit to include only molecules that have an atom for a given molecule (something like torch.sum() / torch.sum(weight > 0) )
        # x_abs_integral = gm.integrate(gm_data).abs()
        # x_abs_integral = torch.mean(x_abs_integral, dim=0, keepdim=True)
        # x_abs_integral = torch.max(x_abs_integral, torch.tensor(0.01))
        # new_weights = gm.weights(gm_data) / torch.unsqueeze(x_abs_integral, dim=-1)
        # gm_data = gm.pack_mixture(new_weights, gm.positions(gm_data), gm.covariances(gm_data))
        self.data = gm_data

        # // test reading again. how much memory is taken? can we convert directly to torch.tensor (on cpu)?
        # // implement storage, __len__ ,and get_item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mixture = self.data[index:index+1, :, :, :]

        assert len(mixture.shape) == 4

        assert gm.n_batch(mixture) == 1
        assert gm.n_layers(mixture) > 0
        assert gm.n_components(mixture) > 0
        assert gm.n_dimensions(mixture) == 3

        return mixture[0], self.targets[index]
