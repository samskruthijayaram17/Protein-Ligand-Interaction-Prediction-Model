import os
import joblib
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# -------------------------
# PATH SETUP (ROBUST)
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

# -------------------------
# LOAD FILES
# -------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

# -------------------------
# GLOBALS
# -------------------------
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)

# -------------------------
# PROTEIN FEATURES
# -------------------------
def get_protein_features(seq):
    seq_len = len(seq)
    aa_comp = [seq.count(aa) / seq_len for aa in amino_acids]
    return aa_comp + [seq_len]

# -------------------------
# LIGAND FEATURES
# -------------------------
def get_ligand_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError("Invalid SMILES string")

    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol)
    ]

    fp = morgan_gen.GetFingerprint(mol)

    return features + list(fp)

# -------------------------
# MAIN PREDICTION FUNCTION
# -------------------------
def predict_binding(protein_seq, smiles, threshold=0.45):
    # Generate features
    p_feat = get_protein_features(protein_seq)
    l_feat = get_ligand_features(smiles)

    # Combine
    X = pd.DataFrame([p_feat + l_feat], columns=feature_names)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    pred = int(prob > threshold)

    return {
        "Binding Probability": float(prob),
        "Prediction": "Binds" if pred == 1 else "Does Not Bind"
    }

# -------------------------
# RUN EXAMPLE
# -------------------------
if __name__ == "__main__":
    protein_seq = "MKTLLILAVALAVALAAP"   # Example protein sequence (replace with actual sequence)
    smiles = "CCO"   # Example SMILES string (replace with actual SMILES)

    result = predict_binding(protein_seq, smiles)
    print(result)