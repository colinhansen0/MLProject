"""Dataset exploration and loading/featurization script for RF

Main():
    tests loading and featurization pipeline to validate data if needed

Main function:

    get_dataset(rebuild: bool = False) -> tuple(X, y_class, y_reg):
        functionality is to call this to load dataset, if it is already built (featurized.npz) it will load it,
        if not it will call load_dataset and featurize and give you the X, y_class, and y_reg
"""

from pathlib import Path

import pandas as pd
import numpy as np
from rdkit import Chem                         
from rdkit.Chem import rdFingerprintGenerator 
from rdkit import DataStructs 

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bace.csv"
FEATURIZED_PATH = DATA_PATH.parent / "featurized.npz"

def load_dataset(input_path: Path = DATA_PATH) -> tuple[list[str], np.ndarray, np.ndarray]:
    """load bace.csv dataset and return a tuple with list of smiles, array of class, array of pIC50]"""
    df = pd.read_csv(input_path)

    smiles  = df["mol"].tolist()        # verify "mol" is the SMILES column
    y_lab = df["Class"].to_numpy()    # binary target
    y_reg   = df["pIC50"].to_numpy()    # continuous target

    return (smiles, y_lab, y_reg)

def featurize(
    smiles: list[str], y_lab: np.ndarray, y_reg: np.ndarray,
    radius: int = 2, fp_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """featurization function that turns smiles into mols makes fingerprint of each mol
    returns an a tuple with three arrays (X, y_class, y_reg)"""

    X_list = []
    y_class_list = []
    y_reg_list = []

    mols_dropped=0

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    for smile,lab,reg in zip(smiles, y_lab, y_reg):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            mols_dropped+=1
            continue
        fp = gen.GetFingerprint(mol) 
        arr = np.zeros((fp_size,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)  
        X_list.append(arr)
        y_class_list.append(lab)
        y_reg_list.append(reg)

    print(f"{mols_dropped} smiles failed to parse to mol")

    X = np.array(X_list, dtype=np.uint8)
    y_class = np.array(y_class_list)
    y_reg = np.array(y_reg_list)
    return X, y_class, y_reg

def get_dataset(rebuild: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Featurize + cache on first call; load the cache on every call after."""

    if FEATURIZED_PATH.exists() and not rebuild:
        d = np.load(FEATURIZED_PATH)
        return d["X"], d["y_class"], d["y_reg"]
    
    smiles, y_class, y_reg = load_dataset()
    X, y_class, y_reg = featurize(smiles, y_class, y_reg)
    np.savez(FEATURIZED_PATH, X=X, y_class=y_class, y_reg=y_reg)

    return X, y_class, y_reg

if __name__ == "__main__":
    # build the dataset
    X, y_class, y_reg = get_dataset(rebuild=True)

    # --- sanity checks: earn trust in the matrix ---
    print("X shape:       ", X.shape)                          # expect (1513, 2048)
    print("aligned?       ", X.shape[0] == y_class.shape[0] == y_reg.shape[0])
    print("X dtype:       ", X.dtype)                          # expect uint8
    print("unique values: ", np.unique(X))                     # expect [0 1] only
    print("avg bits set:  ", X.sum(axis=1).mean())             # sparse: tens, not ~0 or ~2048
    print("all-zero rows: ", int((X.sum(axis=1) == 0).sum()))  # ideally 0
    print("class balance: ", np.bincount(y_class))             # ~[822 691]
    print("pIC50 range:   ", y_reg.min(), y_reg.max())         # ~2.54 to 10.52

    # --- verify it reloads cleanly ---
    d = np.load(FEATURIZED_PATH)
    print("reloaded:      ", d["X"].shape, d["y_class"].shape, d["y_reg"].shape)