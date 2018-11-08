import numpy as np
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

def get_smile_numerical_vectors_dict():
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

def get_vectors_from_smiles():
    with open("augmented_smiles.txt", 'r') as f:
        augmented_smiles = [line.rstrip('\n') for line in f]
    mapping_dict = get_smile_numerical_vectors_dict()
    vectors = []
    for smile in augmented_smiles:
        vectors.append(mapping_dict[smile])
    return vectors


def get_augmented_targets():
    targets = np.loadtxt("augmented_targets.npy")
    return targets

