"""From-scratch Morgan/ECFP fingerprint. Takes an RDKit mol, returns a bit vector.

Turns a molecule into a fixed-length 0/1 vector where each bit marks the presence
of a local atom-centered substructure. Takes RDKit's parsed mol as input; SMILES
parsing and graph construction are out of scope, since this is the fingerprint and
not the cheminformatics. Validated against RDKit's own Morgan generator by comparing
the set of (atom, radius) environments each one enumerates rather than bit positions
(see the note below on why).

Pieces:
    - initial_invariants: round-0 per-atom id, hashed from local atom properties
                          (atomic number, degree, attached Hs, formal charge, in-ring).
                          Atoms with identical local descriptions collide to one id.
    - relabel:            one Weisfeiler-Lehman round. Rebuilds each atom's id from its
                          own current id plus its neighbors' (bond type, current id)
                          pairs, sorted so neighbor ordering doesn't matter. The round
                          number t is folded into the hash so radius-t features can't
                          collide with features from other radii. New ids are computed
                          from the previous round's ids, never updated in place mid-round.
    - morgan_fingerprint: seed with round 0, relabel for t = 1..radius, collect every
                          id from every round, fold into fp_size bits via id % fp_size.

Design notes:
    - The WL relabel is the conceptual core. Growing each atom's view one bond further
      out per round is the same neighborhood-aggregation idea that message-passing GNNs
      generalize.
    - Bit positions are NOT matched to RDKit's. Those depend on RDKit's internal hash,
      which is arbitrary and not worth replicating. Correctness is checked instead on the
      (atom, radius) environment sets, which don't depend on the hash.

Validation (radius 2, against RDKit's rdFingerprintGenerator via bitInfo):
    - Benzene: 18 environments, exact match with RDKit.
    - Ethanol: mine is a strict superset. Every environment RDKit finds, mine also finds,
      plus three extras at radius 2, one per atom.
    - Cause: mine leaves out ECFP's duplicate-environment rule. When growing the radius
      produces an environment covering the same atom set as one already recorded (growth
      has saturated; ethanol is only two bonds end to end, so radius 2 covers the whole
      molecule from every atom), RDKit drops the redundant feature while mine still emits
      one per atom per round. This adds a few redundant bits but captures no wrong
      substructure. It's a legitimate circular fingerprint, just without RDKit's dedup
      optimization.
"""

from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit.Chem import Mol

def initial_invariants(mol: Mol) -> list[int]:
    ids = []
    for atom in mol.GetAtoms():
        props = (
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            atom.IsInRing(),
        )
        atom_id = hash(props)
        ids.append(atom_id)
    return ids


def relabel(mol: Mol, prev_ids: list[int], t: int) -> list[int]:
    ids = []
    for atom in mol.GetAtoms():
        bonded = []
        for bond in atom.GetBonds():
            neighbor = bond.GetOtherAtom(atom)
            props = (bond.GetBondTypeAsDouble(),
                     prev_ids[neighbor.GetIdx()])
            bonded.append(props)
        bonded.sort()
        atom_id = prev_ids[atom.GetIdx()]
        to_hash = (t, tuple(bonded), atom_id)
        ids.append(hash(to_hash))
    return ids


def morgan_fingerprint(mol: Mol, radius: int=2, fp_size: int=2048) -> list[int]:
    
    ids_list = []

    ids = initial_invariants(mol)
    ids_list.append(ids)

    for t in range(1, radius +1 ):
        ids = relabel(mol, ids, t)
        ids_list.append(ids)
    
    all_ids = np.array([i for round_ids in ids_list for i in round_ids])

    fp = np.zeros(fp_size, dtype=np.uint8)
    fp[all_ids % fp_size] = 1

    return fp


def main() -> None:
    from rdkit.Chem import rdFingerprintGenerator

    smiles = ["c1ccccc1", "CCO"]
    radius = 2

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)

        ours = set()
        ids = initial_invariants(mol)
        for i in range(len(ids)):
            ours.add((i, 0))
        for t in range(1, radius + 1):
            ids = relabel(mol, ids, t)
            for i in range(len(ids)):
                ours.add((i, t))

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=ao)
        bit_info = ao.GetBitInfoMap()

        theirs = set()
        for envs in bit_info.values():
            for atom_idx, rad in envs:
                theirs.add((atom_idx, rad))


        print(f"\n{smi}")
        print(f"  ours   : {len(ours)} environments")
        print(f"  theirs : {len(theirs)} environments")
        print(f"  match  : {ours == theirs}")
        if ours != theirs:
            print(f"  ours - theirs: {sorted(ours - theirs)}")
            print(f"  theirs - ours: {sorted(theirs - ours)}")


if __name__ == "__main__":
    main()