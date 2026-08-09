# MLProject

Classic ML algorithms built from scratch in numpy, validated against their production equivalents (currently scikit-learn). A personal learning project for understanding the internals, run on a molecular drug-discovery dataset.

## Idea

Each algorithm is implemented with numpy and plain math, then validated against a production library on the same features. The established libraries are used only as a yardstick, never inside the algorithms themselves. The data pipeline uses RDKit for molecular featurization.

The data comes from the BACE set (MoleculeNet): ~1,500 molecules carrying both a binary activity label and a continuous potency value (pIC50), so the same models can be tested on classification and regression.

## Status

- [x] Phase 1 — decision tree and random forest (classification + regression), vs scikit-learn
- [x] Phase 2 — gradient boosting (classification + regression), vs scikit-learn
- [x] Add-on: Morgan fingerprint from scratch (the featurization step itself), vs RDKit

## IN PROGRESS

- Phase 3 — multilayer perceptron with manual backprop, validated against a deep-learning framework.

## Layout

```
core/
  data.py            # load BACE, featurize to Morgan fingerprints, cache to disk
  tree.py            # DecisionTree, shared base learner, objective-agnostic via a pluggable task
  RF/
    forest.py        # RandomForest: bagging + per-split feature subsampling + vote/average
  Boost/
    boost.py         # Boost: gradient boosting, pluggable loss (squared error / log loss)
  FP/
    fingerprint.py   # Morgan/ECFP fingerprint from scratch, vs RDKit (add-on, not a phase)
comparisons/
  tree_vs_sklearn.py
  forest_vs_sklearn.py
  forest_vs_sklearn_regression.py
  boost_vs_sklearn_regression.py
  boost_vs_sklearn.py
  check_max_depth.py        # confirms max_depth caps tree growth
  phase1_writeup.ipynb      # phase 1 results table + analysis
  phase2_writeup.ipynb      # phase 2 results table + analysis
data/                 # gitignored (raw csv + cached features)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
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
python -m comparisons.boost_vs_sklearn_regression
python -m comparisons.boost_vs_sklearn
```

Or open `comparisons/phase1_writeup.ipynb` and `comparisons/phase2_writeup.ipynb` for the full results and write-ups.

## Results (phase 1)

Held-out test set, identical split for every model.

| model | metric | from scratch | scikit-learn |
|---|---|---|---|
| tree (classification) | accuracy | ~0.79 | ~0.80 |
| forest (classification) | accuracy | ~0.83 | ~0.83 |
| tree (regression) | R² | ~0.47 | ~0.43 |
| forest (regression) | R² | ~0.70 | ~0.70 |

The from-scratch models track the library closely, and the forest improves on the single tree on both objectives.

## Results (phase 2)

Held-out test set; 100 trees, depth 3, learning rate 0.1, matched on both sides.

| model | metric | from scratch | scikit-learn |
|---|---|---|---|
| boosting (classification) | accuracy | ~0.82 | ~0.85 |
| boosting (regression) | R² | ~0.62 | ~0.62 |

Regression matches scikit-learn almost exactly. Classification trails slightly because scikit-learn refines each leaf with a Newton step (the optimal log-loss value) while the from-scratch version uses plain mean leaves; the two still agree on ~93% of predictions.

## Add-on: Morgan fingerprint from scratch

A short side project outside the three phases. Reimplements the Morgan featurization every phase relies on, taking RDKit's parsed mol as input: hash each atom's local properties into a starting id, run Weisfeiler-Lehman rounds that fold in neighbor ids, then collect every id into a 2048-bit vector.

Validated against RDKit by comparing the (atom, radius) substructures each one enumerates, not bit positions (those depend on RDKit's internal hash). Benzene matches exactly; ethanol comes out a strict superset because this version skips ECFP's duplicate-environment rule, adding a few redundant bits without capturing any wrong substructure.
