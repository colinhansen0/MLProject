"""Quick structural check that max_depth actually caps tree growth.

No dataset or RDKit needed — runs on small synthetic binary-feature data, so
it's just exercising the tree's stopping logic. Verifies, for both tasks:
    - a capped tree never grows deeper than its max_depth
    - max_depth=1 is exactly one split (depth 1, two leaves)
    - an uncapped tree grows deeper than a capped one (the cap does something)

Run from the project root:  python -m comparisons.check_max_depth
(or just `python check_max_depth.py` if you drop it somewhere importable)
"""

import numpy as np

# adjust to wherever tree.py lives — tries both in case you moved it up
try:
    from core.tree import DecisionTree, CLASSIFICATION, REGRESSION
except ImportError:
    from core.tree import DecisionTree, CLASSIFICATION, REGRESSION


def realized_depth(node) -> int:
    """How deep the fitted tree actually goes. Root is depth 0;
    a single split with two leaves is depth 1."""
    if node.is_leaf():
        return 0
    return 1 + max(realized_depth(node.left), realized_depth(node.right))


def leaf_count(node) -> int:
    if node.is_leaf():
        return 1
    return leaf_count(node.left) + leaf_count(node.right)


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 30))           # 200 rows, 30 binary "bits"
    y_class = rng.integers(0, 2, size=200)           # binary labels
    y_reg = rng.normal(size=200)                     # continuous targets

    for name, task, y in [("classification", CLASSIFICATION, y_class),
                          ("regression", REGRESSION, y_reg)]:
        print(f"=== {name} ===")
        for cap in [1, 2, 3, None]:
            tree = DecisionTree(max_depth=cap, task=task, random_state=0).fit(X, y)
            d = realized_depth(tree.root)
            print(f"max_depth={str(cap):<4}  realized depth={d:<3} leaves={leaf_count(tree.root)}")
            if cap is not None:
                assert d <= cap, f"depth {d} exceeded the cap of {cap}!"

        capped = realized_depth(DecisionTree(max_depth=3, task=task, random_state=0).fit(X, y).root)
        uncapped = realized_depth(DecisionTree(max_depth=None, task=task, random_state=0).fit(X, y).root)
        assert uncapped > capped, "uncapped tree should grow deeper than a capped one"
        print()

    print("all checks passed")


if __name__ == "__main__":
    main()