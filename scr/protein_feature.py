import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(script_dir, "..", "data", "cleaned_dataset_valid.csv")
output_file = os.path.join(script_dir, "..", "data", "protein_features.csv")

df = pd.read_csv(input_file)


amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def get_aa_composition(seq):
    seq_len = len(seq)
    return [seq.count(aa)/seq_len for aa in amino_acids]

aa_features = df["Target_Sequence"].apply(get_aa_composition)
aa_df = pd.DataFrame(aa_features.tolist(), columns=amino_acids)

df["Seq_Length"] = df["Target_Sequence"].apply(len)

protein_features = pd.concat([aa_df, df["Seq_Length"]], axis=1)

protein_features.to_csv(output_file, index=False)