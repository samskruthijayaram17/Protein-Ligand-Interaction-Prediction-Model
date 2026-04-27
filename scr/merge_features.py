import pandas as pd
import os
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))

df_file = os.path.normpath(os.path.join(script_dir, "..", "data", "cleaned_dataset_valid.csv"))
protein_file = os.path.normpath(os.path.join(script_dir, "..", "data", "protein_features.csv"))
ligand_file = os.path.normpath(os.path.join(script_dir, "..", "data", "ligand_features.csv"))

protein_df = pd.read_csv(protein_file)
ligand_df = pd.read_csv(ligand_file)

df = pd.read_csv(df_file)
df = df.reset_index(drop=True)

X = pd.concat([protein_df, ligand_df], axis=1)
y = df["Label"]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_out = os.path.normpath(os.path.join(script_dir, "..", "data", "X_features.csv"))
y_out = os.path.normpath(os.path.join(script_dir, "..", "data", "y_labels.csv"))

X.to_csv(X_out, index=False)
y.to_frame().to_csv(y_out, index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv(os.path.normpath(os.path.join(script_dir, "..", "data", "X_train.csv")), index=False)
X_test.to_csv(os.path.normpath(os.path.join(script_dir, "..", "data", "X_test.csv")), index=False)
y_train.to_frame().to_csv(os.path.normpath(os.path.join(script_dir, "..", "data", "y_train.csv")), index=False)
y_test.to_frame().to_csv(os.path.normpath(os.path.join(script_dir, "..", "data", "y_test.csv")), index=False)
