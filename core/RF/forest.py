"""From-scratch random forest classifier.

An ensemble of decision trees (tree.py) trained with two sources of randomness,
then aggregated by majority vote. Validated against sklearn's
RandomForestClassifier on the same feature matrix.

What makes it a forest rather than one tree:
    - bagging:             each tree trains on a bootstrap sample (rows drawn with
                           replacement) of the training data
    - feature subsampling: each split considers only a random subset of the
                           features (typically ~sqrt(n_features))

Pieces to build:
    - bootstrap sampling of training rows
    - train N trees, each on its own bootstrap sample
    - aggregate the trees' predictions (majority vote)

Depends on: tree.py.
"""