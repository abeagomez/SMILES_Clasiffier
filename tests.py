import canonize_data
import numpy as np

def augmented_smiles_test():
    """
    Test that the original augmented data array and the saved one are the same
    """
    augmented_smiles_array = canonize_data.get_all_smiles_targets_possibilities()[0]
    with open("augmented_smiles.txt", 'r') as f:
        augmented_smiles_file = [line.rstrip('\n') for line in f]
    if len(augmented_smiles_array) != len(augmented_smiles_file):
        print("length is different")
        return
    else:
        for smile in augmented_smiles_array:
            if not i in augmented_smiles_file:
                print("elements are not the same")
                return
    print("boths arrays are equal")

