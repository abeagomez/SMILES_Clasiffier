import numpy as np
import csv
import random as rd
from smiles_lexer import tokenize_smiles

def get_tokens_dict():
    st_dict = {}
    with open("canonized_smiles.txt", 'r') as f:
        smiles = [line.rstrip('\n') for line in f]
    index = 2
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
        n_targets.append([float(i) if i != "" else None for i in target])
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
    return ones_counter, zeros_counter

def get_filled_targets():
    """
    This function fills all the missing values in the targets with 0s
    """
    targets = get_targets()
    for target in targets:
        for i in range(len(target)):
            if target[i] is None:
                target[i] = 0.0
    return targets

def get_positive_samples(target_index, smiles, targets):
    positive_values = []
    for i in range(len(targets)):
        if targets[i][target_index]:
            positive_values.append(smiles[i])
    return positive_values

def over_sampling_data_set(target_index, positive_samples, smiles, target):
    positives, negatives = targets_mode()
    ratio = negatives[target_index] - positives[target_index]
    while ratio > 0:
        vector = rd.choice(positive_samples)
        index = rd.randint(0, len(target)-1)
        smiles.insert(index, vector)
        target.insert(index, 1.0)
        ratio -= 1
    return smiles, target

def get_target_values(categories):
    values = []
    for pair in categories:
        if pair[0] > 0.5:
            values.append(0.0)
        else:
            values.append(1.0)
    return values

def evaluation_variables(predicted, expected):
    """
    :returns a tuple: true_positives, true_negatives, false_positives, false_negatives
    """
    t_p, t_n, f_p, f_n = 0, 0, 0, 0
    for i in range(len(predicted)):
        if predicted[i] == 1 and expected[i] == 1:
            t_p += 1
        elif predicted[i] == 1 and expected[i] != 1:
            f_p += 1
        elif predicted[i] == 0 and expected[i] == 0:
            t_n += 1
        else:
            f_n += 1
    print("")
    print("True Positives: %i" % t_p)
    print("True Negatives: %i" % t_n)
    print("False Positives: %i" % f_p)
    print("False Negatives: %i" % f_n)
    print("")
    print("Accuracy: %f" % ((t_p + t_n)/(t_p + t_n + f_p + f_n)))
    print("Precision: %f" % (t_p/(t_p + f_p)))
    print("Recall: %f" % (t_p/(t_p + f_n)))

