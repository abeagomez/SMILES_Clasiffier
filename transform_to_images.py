import csv
from rdkit import Chem
from rdkit.Chem import Draw

with open("data.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        index = 0
        for row in reader:
            mol = Chem.MolFromSmiles(row["smiles"])
            img_name = str(index) + ".png"
            Draw.MolToFile(mol, 'images/' + img_name)
            index += 1

