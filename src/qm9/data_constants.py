
# https://figshare.com/articles/dataset/Atomref_Reference_thermochemical_energies_of_H_C_N_O_F_atoms/1057643?backTo=/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904?backTo=/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
# first index atom type (string), second index prediction property (string)
REFERENCE_THERMOCHEMICAL_ENERGIES = {
    'H': {"zpve": 0.000000, "U0":  -0.500273, "U":  -0.498857, "H":  -0.497912, "G":  -0.510927, "Cv": 2.981},
    'C': {"zpve": 0.000000, "U0": -37.846772, "U": -37.845355, "H": -37.844411, "G": -37.861317, "Cv": 2.981},
    'N': {"zpve": 0.000000, "U0": -54.583861, "U": -54.582445, "H": -54.581501, "G": -54.598897, "Cv": 2.981},
    'O': {"zpve": 0.000000, "U0": -75.064579, "U": -75.063163, "H": -75.062219, "G": -75.079532, "Cv": 2.981},
    'F': {"zpve": 0.000000, "U0": -99.718730, "U": -99.717314, "H": -99.716370, "G": -99.733544, "Cv": 2.981},
}

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
