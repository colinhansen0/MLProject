# MLProject
 
Classic ML algorithms built from scratch in numpy, validated against their production equivalents (scikit-learn now; XGBoost and PyTorch to come). A personal learning project for understanding the internals, run on a molecular drug-discovery dataset.
 
> Draft README — work in progress.
 
## Idea
 
Each algorithm is implemented with numpy and plain math, then validated against a production library on the *same* features. The established libraries are used only as a yardstick, never inside the algorithms themselves. The data pipeline uses RDKit for molecular featurization.
 
The data comes from the BACE set (MoleculeNet): ~1,500 molecules carrying both a binary activity label and a continuous potency value (pIC50), so the same models can be tested on classification and regression.
 
## Status
 
- [x] Phase 1 — decision tree and random forest (classification + regression), vs scikit-learn
- [ ] Phase 2 — gradient boosting, vs XGBoost
- [ ] Phase 3 — multilayer perceptron with manual backprop, vs PyTorch
## Layout
 
```
core/
  data.py            # load BACE, featurize to Morgan fingerprints, cache to disk
  RF/
    tree.py          # DecisionTree — objective-agnostic via a pluggable task
    forest.py        # RandomForest — bagging + per-split feature subsampling + vote/average
comparisons/
  tree_vs_sklearn.py
  forest_vs_sklearn.py
  forest_vs_sklearn_regression.py
  phase1_writeup.ipynb   # results table + analysis
data/                 # gitignored (raw csv + cached features)
```
 
## Setup
 
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install numpy pandas scikit-learn rdkit jupyterlab matplotlib
```
 
## Data
 
The dataset isn't committed. Download `bace.csv` into `data/`, then the first call to `get_dataset()` featurizes it (Morgan fingerprints, radius 2, 2048 bits) and caches the result to `data/featurized.npz` for later runs.
 
BACE source: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv
 
## Running
 
The comparison scripts import `core`, so run them as modules from the project root:
 
```bash
python -m comparisons.tree_vs_sklearn
python -m comparisons.forest_vs_sklearn
python -m comparisons.forest_vs_sklearn_regression
```
 
Or open `comparisons/phase1_writeup.ipynb` for the full results and write-up.
 
## Results (phase 1)
 
Held-out test set, identical split for every model.
 
| model | metric | from scratch | scikit-learn |
|---|---|---|---|
| tree (classification) | accuracy | ~0.79 | ~0.80 |
| forest (classification) | accuracy | ~0.83 | ~0.83 |
| tree (regression) | R² | ~0.47 | ~0.43 |
| forest (regression) | R² | ~0.70 | ~0.70 |
 
The from-scratch models track the library closely, and the forest improves on the single tree on both objectives.
 
