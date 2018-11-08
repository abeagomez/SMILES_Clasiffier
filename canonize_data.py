from rdkit import Chem
import csv
import numpy as np

"""
This script reads the data from the data.csv file and uses rdkit
to canonize all the smiles.
Also, creates a list with 12 elements for each row with the values of
the targets and, for the empty target values, it creates a new target array
with all the possible values.
Finally, it saves the list with all the canonized smiles, the
list with all the lists of targets and an augmentd list of canonized smiles
into .npy files.
"""
def get_smiles_and_targets():
    """
    Reads data from the data.csv file and returns two arrays of equal lenght
    Output:
            smiles: the rdkit representation of every smiles in the original
                    data
            targets: an array of arrays with the values for every one of the
                    12 targets
    """
    targets_keys = ["target1", "target2", "target3", "target4", "target5",
                    "target6", "target7", "target8", "target9", "target10",
                    "target11", "target12"]
    smiles = []
    targets = []
    with open("data.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"])))
            row_targets = []
            for key in targets_keys:
                row_targets.append(row[key])
            targets.append(row_targets)
    return smiles, targets

def save_smiles():
    """
    Saves the smile representation from rdkit into a .npy file
    """
    smiles, _ = get_smiles_and_targets()
    with open("canonized_smiles.txt", 'w') as f:
        for i in smiles:
            f.write(i + '\n')

def fill_targets(targets_list):
    l = len(targets_list)
    targets_list = [targets_list]
    index = 0
    filled_targets = []
    while index < l:
        for i in targets_list:
            if i[index] == "":
                t_1 = [i[j] if j != index else "0" for j in range(len(i))]
                filled_targets.append(t_1)
                t_2 = [i[j] if j != index else "1" for j in range(len(i))]
                filled_targets.append(t_2)
        if filled_targets:
            targets_list = [i for i in filled_targets]
            filled_targets = []
        index += 1
    return [list(map(int, l)) for l in targets_list]

def get_all_smiles_targets_possibilities():
    smiles, targets = get_smiles_and_targets()
    n_smiles = []
    n_targets = []
    for i in range(len(smiles)):
        t = fill_targets(targets[i])
        for j in t:
            n_smiles.append(smiles[i])
            n_targets.append(j)
    return n_smiles, n_targets

def save_augmented_data():
    s, t = get_all_smiles_targets_possibilities()
    np.savetxt("augmented_targets.npy", t)
    with open("augmented_smiles.txt", 'w') as f:
        for i in s:
            f.write(i + '\n')

#save_smiles()
#save_augmented_data()
save_targets()
