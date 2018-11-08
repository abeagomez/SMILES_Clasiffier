import numpy as np
import csv
from smiles_lexer import tokenize_smiles

def get_tokens_dict():
    st_dict = {}
    with open("canonized_smiles.txt", 'r') as f:
        smiles = [line.rstrip('\n') for line in f]
    index = 1
    for smile in smiles:
        tokenized_smile = list(tokenize_smiles(smile))
        for token in tokenized_smile:
            if not token in st_dict:
                st_dict[token] = index
                index += 1
    return st_dict

def get_smiles_numerical_vectors_dict():
    with open("canonized_smiles.txt", 'r') as f:
        smiles = [line.rstrip('\n') for line in f]
    tokens_dict = get_tokens_dict()
    d = {}
    for smile in smiles:
        vector = []
        tokens = list(tokenize_smiles(smile))
        for token in tokens:
            vector.append(tokens_dict[token])
        d[smile] = vector
    return d

def get_smiles_as_vectors():
    with open("canonized_smiles.txt", 'r') as f:
        smiles = [line.rstrip('\n') for line in f]
    tokens_dict = get_tokens_dict()
    r = []
    for smile in smiles:
        vector = []
        tokens = list(tokenize_smiles(smile))
        for token in tokens:
            vector.append(tokens_dict[token])
        r.append(vector)
    return r

def get_targets():
    targets_keys = ["target1", "target2", "target3", "target4", "target5",
                    "target6", "target7", "target8", "target9", "target10",
                    "target11", "target12"]
    targets = []
    with open("data.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_targets = []
            for key in targets_keys:
                row_targets.append(row[key])
            targets.append(row_targets)
    n_targets = []
    for target in targets:
        n_targets.append([int(i) if i != "" else None for i in target])
    return n_targets

def missing_values_per_columns():
    targets = get_targets()
    counter = np.zeros(12)
    for target in targets:
        for i in range(len(target)):
            if target[i] is None:
                counter[i] += 1
    return counter

def targets_mode():
    targets = get_targets()
    #for this function, the 2 following arrays are not needed
    #I am using them because I want to "see" the numbers ;)
    ones_counter = np.zeros(12)
    zeros_counter = np.zeros(12)
    for target in targets:
        for i in range(len(target)):
            if target[i] is not None:
                if target[i] == 1:
                    ones_counter[i] += 1
                else:
                    zeros_counter[i] += 1
    print(ones_counter)
    print(zeros_counter)

def get_filled_targets():
    targets = get_targets()
    for target in targets:
        for i in range(len(target)):
            if target[i] is None:
                target[i] = 0
    return targets


