from rdkit import Chem
from rdkit import RDLogger
import pandas as pd
import os
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import AllChem

# Disable warnings
RDLogger.DisableLog('rdApp.*')

# File path
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "..", "data", "cleaned_dataset.csv")

# Load data
df = pd.read_csv(input_file)

# Validate SMILES
def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None


original_len = len(df)

df = df[df["SMILES"].apply(is_valid_smiles)]
df = df.reset_index(drop=True)

print("Original size:", original_len)
print("After cleaning:", len(df))

# Save cleaned dataset
output_file = os.path.join(script_dir, "..", "data", "cleaned_dataset_valid.csv")
df.to_csv(output_file, index=False)

print("Cleaned dataset saved.")

### Ligand Feature Extraction
from rdkit.Chem import Descriptors

def get_ligand_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 5
    return [
        Descriptors.MolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]
ligand_features = df["SMILES"].apply(get_ligand_features)
ligand_df = pd.DataFrame(ligand_features.tolist(), columns=["MolWt", "NumRotatableBonds", "NumHDonors", "NumHAcceptors", "TPSA"])
output_file = os.path.join(script_dir, "..", "data", "ligand_features.csv")
ligand_df.to_csv(output_file, index=False)


### Morgan Fingerprint
# Resolve paths and load required datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
input_df_file = os.path.normpath(os.path.join(script_dir, "..", "data", "cleaned_dataset_valid.csv"))
input_ligand_file = os.path.normpath(os.path.join(script_dir, "..", "data", "ligand_features.csv"))
output_file = os.path.normpath(os.path.join(script_dir, "..", "data", "ligand_full_features.csv"))

df = pd.read_csv(input_df_file)
ligand_df = pd.read_csv(input_ligand_file)
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [None] * 2048
    fp = morgan_gen.GetFingerprint(mol)
    return list(fp)

fp_features = df["SMILES"].apply(get_fingerprint)
fp_df = pd.DataFrame(fp_features.tolist(), columns=[f"FP_{i}" for i in range(2048)])
ligand_full=pd.concat([ligand_df, fp_df], axis=1)
ligand_full.to_csv(output_file, index=False)