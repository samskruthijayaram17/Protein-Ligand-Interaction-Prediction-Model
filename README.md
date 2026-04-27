# Protein-Ligand Interaction Prediction

A machine learning pipeline for predicting protein-ligand binding interactions using sequence-based protein features and molecular descriptor-based ligand features.

## Overview

This project implements an end-to-end machine learning workflow to predict whether a protein-ligand pair will form a binding interaction. The system extracts features from protein sequences and ligand SMILES strings, merges them into a unified feature matrix, and trains classification models to make binding predictions.

## Project Structure

```
protein ligand project/
├── README.md                    # This file
├── data/                        # Data files (raw, cleaned, features)
│   ├── raw_dataset.csv
│   ├── cleaned_dataset.csv
│   ├── cleaned_dataset_valid.csv
│   ├── protein_features.csv     # Extracted protein features
│   ├── ligand_features.csv      # Extracted ligand features
│   ├── X_features.csv           # Merged feature matrix
│   ├── y_labels.csv             # Binary binding labels
│   └── [other intermediate files]
├── src/                         # Source code modules
│   ├── data_loader.py           # Load and validate raw data
│   ├── preprocessing.py         # Data cleaning and validation
│   ├── protein_feature.py       # Extract amino acid composition features
│   ├── ligand_features.py       # Extract molecular descriptors
│   ├── merge_features.py        # Combine protein & ligand features
│   ├── model.py                 # Model training and evaluation
│   └── predict.py               # Inference script for new predictions
├── models/                      # Trained model artifacts
│   ├── final_model.pkl          # Serialized classifier
│   ├── scaler.pkl               # Feature scaler
│   └── feature_names.pkl        # Feature column names
├── notebooks/                   # Jupyter notebooks
│   └── 1_ml_model.ipynb         # Full analysis & model comparison
└── results/                     # Model performance outputs
    ├── roc_curve.pdf
    ├── roc_curve_rf.pdf
    ├── precision_recall_curve.pdf
    └── [evaluation metrics]
```

## Data Pipeline

### 1. **Data Loading & Cleaning** (`data_loader.py`)
   - Load raw protein-ligand dataset
   - Validate data integrity and format

### 2. **Preprocessing** (`preprocessing.py` + `ligand_features.py`)
   - Validate SMILES strings using RDKit
   - Filter out invalid molecular structures
   - Clean dataset and save to `cleaned_dataset_valid.csv`

### 3. **Feature Extraction**

   **Protein Features** (`protein_feature.py`):
   - Amino acid composition (20 standard amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
   - Normalized frequency for each amino acid position
   - Sequence length as additional feature
   - Output: `protein_features.csv` (21 features)

   **Ligand Features** (`ligand_features.py`):
   - Molecular Weight (MW)
   - Number of Rotatable Bonds
   - Hydrogen Bond Donors (HBD)
   - Hydrogen Bond Acceptors (HBA)
   - Topological Polar Surface Area (TPSA)
   - Output: `ligand_features.csv` (5 features)

### 4. **Feature Merging** (`merge_features.py`)
   - Concatenate protein and ligand feature matrices
   - Create final feature matrix `X_features.csv` (26 total features)
   - Extract binary labels to `y_labels.csv`
   - Train/test split: 80/20 with stratification

## Machine Learning Models

### Logistic Regression
- **Algorithm**: Linear classification with L2 regularization
- **Scaler**: StandardScaler (fitted on training data)
- **Key findings**: 
  - Lower recall on binding class (class 1)
  - Better performance on non-binding interactions

### Random Forest Classifier
- **Algorithm**: 100 decision trees with random_state=42
- **No scaling required**: Tree-based model is scale-invariant
- **Feature importance analysis**:
  - **Protein features dominate**: 95.52% of total importance
  - **Ligand features contribute minimally**: 4.48% of importance
  - **Fingerprints**: Not used in this version (0% importance)
  - **Top features**: Amino acids A, I, G, D, L (each ~4.7% importance)

### Performance Metrics
Both models evaluated on test set (20% of data) using:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve & AUC
- Precision-Recall Curve

## Key Findings

1. **Protein sequence composition is the primary predictor** of binding affinity
   - Single amino acids carry more predictive signal than compound ligand properties
   - Suggests binding specificity is primarily driven by target protein characteristics

2. **Class imbalance challenges**
   - Model struggles with recall on binding interactions (class 1)
   - May indicate imbalanced dataset or insufficient feature representation for binding cases

3. **Ligand complexity underutilized**
   - Simple molecular descriptors (MW, HBD/HBA, TPSA) contribute minimally
   - Future work: fingerprints, conformation-dependent features, or docking scores could improve

## Usage

### Installation
```bash
cd protein\ ligand\ project
pip install -r requirements.txt
```

**Dependencies:**
- pandas
- scikit-learn
- rdkit
- joblib
- matplotlib

### Running the Pipeline

1. **Extract protein features**:
   ```bash
   python src/protein_feature.py
   ```

2. **Extract ligand features**:
   ```bash
   python src/ligand_features.py
   ```

3. **Merge features**:
   ```bash
   python src/merge_features.py
   ```

4. **Train models & evaluate** (Jupyter):
   ```bash
   jupyter notebook notebooks/1_ml_model.ipynb
   ```

5. **Make predictions on new data**:
   ```bash
   python src/predict.py
   ```

### Input Data Format

**Protein-Ligand Dataset CSV:**
```
Target_Sequence,SMILES,Label
MVLSPADKTNVIRAAQNCYSTEIN...,[structure],1
MKKFFVIIISLLTAIASSSYCAQV...,[structure],0
...
```

- `Target_Sequence`: Protein amino acid sequence
- `SMILES`: Canonical SMILES representation of ligand
- `Label`: Binary label (1 = binding, 0 = non-binding)

### Output Files

After running the pipeline:
- `data/protein_features.csv` – Amino acid composition for all sequences
- `data/ligand_features.csv` – Molecular descriptors for all ligands
- `data/X_features.csv` – Merged feature matrix (ready for modeling)
- `data/y_labels.csv` – Corresponding labels
- `models/final_model.pkl` – Trained Random Forest classifier
- `models/scaler.pkl` – Fitted StandardScaler
- `results/roc_curve_rf.pdf` – ROC curve visualization

## Model Files

### Trained Artifacts (in `models/`)
- **final_model.pkl**: Serialized Random Forest model (100 estimators)
- **scaler.pkl**: StandardScaler fitted on training features
- **feature_names.pkl**: List of feature column names for consistent prediction

## Next Steps for Improvement

1. **Address class imbalance**
   - Class weights or SMOTE resampling
   - Threshold optimization for binding class

2. **Enhanced ligand features**
   - Morgan/ECFP fingerprints
   - 3D conformer-based descriptors
   - Docking score integration

3. **Protein sequence modeling**
   - N-gram composition patterns
   - Position-specific scoring matrices (PSSM)
   - Deep learning embeddings (ESM-2, ProtBERT)

4. **Ensemble methods**
   - Stacking models (Logistic Regression + Random Forest)
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural networks for learned feature interactions

5. **Cross-validation**
   - K-fold cross-validation for robust performance estimates
   - Nested CV for hyperparameter tuning

## References

- **RDKit**: https://www.rdkit.org/ (Molecular descriptor computation)
- **scikit-learn**: https://scikit-learn.org/ (Machine learning algorithms)
- **Protein Features**: Standard amino acid composition analysis
- **Ligand Features**: Standard Lipinski Rule of Five descriptors