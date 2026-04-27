import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(script_dir, "..", "data", "clean_protein_ligand_dataset.csv")
output_file = os.path.join(script_dir, "..", "data", "raw_dataset.csv")

df = pd.read_csv(input_file)

print(df.head())
print(df.shape)
print(df.info())

df.to_csv(output_file, index=False)