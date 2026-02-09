"""
generate_derivatives.py
======================

This module provides simplistic routines for enumerating hypothetical
derivatives of cannabinoid molecules.  The aim is not to produce
synthesizable compounds automatically but rather to illustrate how one
might explore structural modifications programmatically using RDKit.

Two basic transformations are implemented:

1. **Chain length modification** –  For linear alkyl side chains (e.g. the
   pentyl tail on THC), we can propose analogues with one extra or one
   fewer methylene unit.  This is done by searching for terminal carbon
   chains and substituting them with longer/shorter chains.  The search
   pattern is very simplistic and may not work on all molecules.
2. **Hydroxylation** –  We introduce a hydroxyl (–OH) group onto an
   sp3 carbon that is not already substituted by heteroatoms.  Again
   this is a heuristic intended for demonstration.

These functions operate on SMILES strings and return lists of SMILES
strings for the proposed derivatives.  If a transformation fails, the
original molecule is returned unchanged.

Users should validate the synthetic feasibility and legal status of any
derivatives before further consideration.  This code is for research
and educational purposes only.
"""

from __future__ import annotations

from typing import List

from rdkit import Chem
from rdkit.Chem import rdChemReactions



def _safe_mol_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    return mol



def propose_chain_variants(smiles: str, delta: int = 1) -> List[str]:
    """Propose side‑chain length variants of a cannabinoid SMILES.

    The function searches for terminal linear carbon chains of length
    greater than two and returns SMILES strings where the chain has been
    lengthened or shortened by ``delta`` carbons.  If ``delta`` is
    positive, chains become longer; if negative, shorter.  If no suitable
    chain is found, the original SMILES is returned.
    """
    mol = _safe_mol_from_smiles(smiles)
    if mol is None:
        return []
    # Define a simple SMARTS pattern for a terminal carbon chain of length >=3
    # pattern: carbon with at least two carbon neighbours and one hydrogen (alkyl)
    pattern = Chem.MolFromSmarts("[CH2][CH2][CH3]")
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return [smiles]
    # For each match, attempt to modify the chain length
    variants = []
    for match in matches:
        # Identify the end atom (terminal carbon)
        # match indices correspond to pattern atoms; choose the last carbon in pattern
        end_idx = match[-1]
        chain_atom = mol.GetAtomWithIdx(end_idx)
        # Build a new molecule with adjusted chain
        rw_mol = Chem.RWMol(mol)
        # Expand or contract chain by adding/removing CH2 groups
        if delta > 0:
            for i in range(delta):
                new_c = Chem.Atom("C")
                new_idx = rw_mol.AddAtom(new_c)
                rw_mol.AddBond(end_idx, new_idx, order=Chem.rdchem.BondType.SINGLE)
                end_idx = new_idx
        elif delta < 0:
            for i in range(-delta):
                # Remove terminal carbon if degree is 1
                if chain_atom.GetDegree() == 1:
                    # Remove bonds then atom
                    rw_mol.RemoveAtom(end_idx)
                else:
                    break
        new_smiles = Chem.MolToSmiles(rw_mol, canonical=True)
        variants.append(new_smiles)
    return list(set(variants)) or [smiles]



def propose_hydroxylated(smiles: str) -> List[str]:
    """Propose a hydroxylated derivative by adding an –OH group.

    This function performs a simple RDKit reaction that converts a
    terminal methyl group (–CH3) into an alcohol (–CH2OH).  If no such
    group is found, the original SMILES is returned.
    """
    reaction_smarts = "[CH3:1]>>[CH2:1]O"
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    mol = _safe_mol_from_smiles(smiles)
    if mol is None:
        return []
    products = rxn.RunReactants((mol,))
    variants = []
    for prod_tuple in products:
        prod = prod_tuple[0]
        smi = Chem.MolToSmiles(prod, canonical=True)
        variants.append(smi)
    return list(set(variants)) or [smiles]



def propose_derivatives(smiles: str) -> List[str]:
    """Combine simple transformations to propose a set of derivatives."""
    results = set()
    for delta in [1, -1]:
        for variant in propose_chain_variants(smiles, delta=delta):
            results.add(variant)
    for variant in propose_hydroxylated(smiles):
        results.add(variant)
    # Always include the original
    results.add(smiles)
    return list(results)
