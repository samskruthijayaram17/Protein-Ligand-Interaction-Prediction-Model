import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(script_dir, "..", "data", "raw_dataset.csv")
output_file = os.path.join(script_dir, "..", "data", "cleaned_dataset.csv")

df = pd.read_csv(input_file)

# Cleaning
df = df.dropna()
df = df.drop_duplicates()

# Visualization
sns.histplot(df["Affinity"], bins=50)
plt.show()

# Label creation
threshold = 7
df["Label"] = df["Affinity"].apply(lambda x: 1 if x < threshold else 0)

print(df["Label"].value_counts())

# Sampling
df = df.sample(n=min(1500, len(df)), random_state=42)

df.to_csv(output_file, index=False)