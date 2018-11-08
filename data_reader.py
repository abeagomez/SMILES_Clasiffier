import csv
import cv2

def get_data():
    """
    Function to read data.csv
    Output: a dictionary with keys equals to columns names.
    """
    d = {"smiles": [],"target1": [], "target2": [], "target3": [], "target4": [],
        "target5": [], "target6": [], "target7": [], "target8": [], "target9": [],
         "target10": [], "target11": [], "target12": []}
    keys = []
    for k in d:
        keys.append(k)
    with open("data.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k in keys:
                d[k].append(row[k])

    return d

def filtering_smiles(data):
    """
    Input: the get_data() function output
    Output:
    """
    r = {}
    for mol_i in range(0, len(data["smiles"])):
        l = [data[k][mol_i] for k in data if k != "smiles"]
        if data["smiles"][mol_i] in r:
            r[data["smiles"][mol_i]].append(l)
        else:
            r[data["smiles"][mol_i]] = []
            r[data["smiles"][mol_i]].append(l)
    #r: a dictionary with "smiles" as keys and a list of lists of targets
    #as values. A smile can have more than a list of targets associated.
    output = {}
    for k in r:
        if len(r[k]) <= 1:
            output[k] = r[k]
        else:
            f = try_merge(r[k])
            if f[0]:
                output[k] = f[1]
    return output

def get_integer_representation(d):
    """
    Input: the output of the get_data() function
    Output: a dictionary with every character in a smile representation
    and the integer associated to it
    """
    int_element_representation = {}
    c = 0
    for smile in d["smiles"]:
        for i in smile:
            if i in int_element_representation:
                continue
            else:
                int_element_representation[i] = c
                c+=1
    return int_element_representation

def get_vector(molecule, dict_int):
    """
    Input: an smile representation of a molecule
            the dictionary that maps smile characters to int
    Output: the integer representation of the molecule
    """
    r = []
    for c in molecule:
        r.append(dict_int[c])
    return r

def integer_representation(data):
    """
    Input: The output of the get_data() function
    Output: A dictionary that associated every smile with it's interger
    representation
    """
    d = get_integer_representation(data)
    mol_to_integer = {}
    for mol in data["smiles"]:
        if mol in mol_to_integer:
            continue
        else:
            mol_to_integer[mol] = get_vector(mol, d)
    return mol_to_integer

def try_merge(l):
    """
    Input: a list of lists to merge
    Output: A tuple, with the first value representing if the list were
    mergeable and a second value with the result of the merging
    """
    r = l[0]
    for i in l:
        for j in range(0,len(i)):
            if r[j] != "" and i[j] != "":
                if r[j] != i[j]:
                    return False, []
            else:
                if r[j] == "":
                    r[j] = i[j]
    return True, r

def get_smile_intVectors_and_targets(data):
    """
    Input: the output of the get_data() function
    Output: a tuple T where:
                T[0]: dictionary smile:integer_smile_representation
                T[1]: dictionary smile:targets_values
    """
    vectors_dict = integer_representation(data)
    all_targets_dict = filtering_smiles(data)
    copy = dict(vectors_dict)
    for k in vectors_dict:
        if not k in all_targets_dict:
            del copy[k]
    vectors_dict = copy
    return vectors_dict, all_targets_dict

#get_smile_intVectors_and_targets(get_data())

