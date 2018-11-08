from smiles_lexer import tokenize_smiles
import numpy as np

def from_csv():
    data = open('data.csv')
    start = True
    for line in data:
        if start:
            start = False
            continue
        split = line.split(',')
        smiles = split[0]
        print(list(tokenize_smiles(smiles)))
    print('end')

def from_npy():
    smiles = np.loadtxt("canonized_data.npy", dtype=str)
    for smile in smiles:
        print(list(tokenize_smiles(smile)))

