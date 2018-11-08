from rdkit import Chem
from rdkit.Chem import Draw

print(Chem.MolToSmiles(Chem.MolFromSmiles("O=c1[nH]c(=O)n([C@H]2C[C@H](O)[C@@H](CO)O2)cc1I")))
mol = Chem.MolFromSmiles("NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1")
Draw.MolToFile(mol,'cdk2.png')