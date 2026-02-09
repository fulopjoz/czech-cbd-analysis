"""
rdkit_analysis.py
=================

This module contains helper functions for analysing cannabinoid molecules
using RDKit.  Given SMILES strings, it computes a range of
physicochemical descriptors, evaluates drug‑likeness via Lipinski’s
rules and QED, produces simple heuristic ADMET assessments, and can
generate 2D depictions of molecules.

The goal is to provide a lightweight, extensible foundation for
exploring novel cannabinoids scraped from the Czech‑CBD website.  It is
not intended as a definitive ADMET predictor; rather, it uses
rule‑based heuristics to give approximate indications of absorption,
distribution and toxicity.  Users are encouraged to integrate more
sophisticated models and datasets where appropriate.

Dependencies:
    - rdkit (ensure the ``rdkit`` package is installed)
    - pandas (for tabular output)
    - scikit‑learn (optional for future model development)

Usage example::

    from rdkit_analysis import analyse_smiles

    smiles = "O=C(Oc2cc(cc1OC([C@@H]3CC/C(=C\[C@H]3c12)C)(C)C)CCCCC)C"  # THCO
    result = analyse_smiles(smiles)
    print(result.descriptors)
    print(result.lipinski_pass)

The ``Result`` dataclass encapsulates the descriptor dictionary, Lipinski
evaluation, QED score, ADMET predictions and an RDKit Mol object.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED, rdMolDescriptors, Draw

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class AnalysisResult:
    """Container for RDKit analysis results."""

    smiles: str
    mol: Chem.Mol
    descriptors: Dict[str, float]
    lipinski_pass: bool
    lipinski_violations: Dict[str, bool]
    qed_score: float
    admet_predictions: Dict[str, str]
    image_path: Optional[str] = None


def compute_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Compute a set of standard RDKit descriptors for a molecule.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule object.

    Returns
    -------
    dict
        Mapping from descriptor names to values.
    """
    desc = {}
    desc["MolWt"] = Descriptors.MolWt(mol)
    desc["LogP"] = Crippen.MolLogP(mol)
    desc["NumHDonors"] = Lipinski.NumHDonors(mol)
    desc["NumHAcceptors"] = Lipinski.NumHAcceptors(mol)
    desc["NumRotatableBonds"] = Lipinski.NumRotatableBonds(mol)
    desc["tPSA"] = rdMolDescriptors.CalcTPSA(mol)
    desc["FractionCSP3"] = rdMolDescriptors.CalcFractionCSP3(mol)
    desc["HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
    desc["NHOHCount"] = Lipinski.NHOHCount(mol)
    desc["NOCount"] = Lipinski.NOCount(mol)
    desc["RingCount"] = rdMolDescriptors.CalcNumRings(mol)
    return desc


def evaluate_lipinski(descriptors: Dict[str, float]) -> (bool, Dict[str, bool]):
    """Evaluate Lipinski’s rule of five on descriptor values.

    Returns a boolean indicating whether all rules pass, and a dict
    highlighting which rules fail.
    """
    violations = {
        "MolWt<=500": descriptors["MolWt"] <= 500,
        "LogP<=5": descriptors["LogP"] <= 5,
        "HBD<=5": descriptors["NumHDonors"] <= 5,
        "HBA<=10": descriptors["NumHAcceptors"] <= 10,
    }
    lipinski_pass = all(violations.values())
    return lipinski_pass, violations


def heuristic_admet(descriptors: Dict[str, float]) -> Dict[str, str]:
    """Generate simple heuristic ADMET predictions from descriptors.

    These rules are illustrative and should not be considered
    authoritative.  They attempt to mimic common medicinal chemistry
    guidelines:

    * Absorption is high if LogP is between 1 and 5 and tPSA < 140.
    * Brain penetration (distribution) is likely if LogP > 2 and tPSA < 70.
    * Compounds with many rotatable bonds (>10) or MolWt > 600 may have
      poor oral bioavailability and low absorption.
    * Potential toxicity is flagged if LogP > 5, MolWt > 600 or if
      HBD/HBA counts exceed Lipinski’s rules.

    Returns a dict mapping ADMET category names to qualitative labels.
    """
    abs_label = "good"
    if descriptors["MolWt"] > 600 or descriptors["NumRotatableBonds"] > 10:
        abs_label = "poor"
    elif descriptors["LogP"] < 0 or descriptors["tPSA"] > 140:
        abs_label = "poor"
    elif descriptors["LogP"] > 5:
        abs_label = "moderate"

    brain = "unlikely"
    if descriptors["LogP"] > 2 and descriptors["tPSA"] < 70:
        brain = "likely"
    elif descriptors["LogP"] > 1 and descriptors["tPSA"] < 90:
        brain = "possible"

    toxicity = "low"
    if descriptors["LogP"] > 5 or descriptors["MolWt"] > 600:
        toxicity = "high"
    elif descriptors["NumHDonors"] > 5 or descriptors["NumHAcceptors"] > 10:
        toxicity = "moderate"

    clearance = "moderate"
    if descriptors["LogP"] < 2 and descriptors["MolWt"] < 350:
        clearance = "fast"
    elif descriptors["LogP"] > 4:
        clearance = "slow"

    return {
        "Absorption": abs_label,
        "BrainPenetration": brain,
        "Toxicity": toxicity,
        "Clearance": clearance,
    }


def analyse_smiles(smiles: str, image_dir: Optional[str] = None) -> AnalysisResult:
    """Create an ``AnalysisResult`` for a given SMILES string.

    Parameters
    ----------
    smiles: str
        SMILES string of the molecule to analyse.
    image_dir: str or None
        Directory in which to save a PNG depiction of the molecule.  If
        provided, an ``image_path`` will be set in the returned result.

    Returns
    -------
    AnalysisResult
        Dataclass encapsulating descriptors, Lipinski evaluation, QED
        score, heuristic ADMET predictions and optionally the saved
        depiction.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # Add hydrogens for 3D geometry if needed (not required for descriptors)
    # mol = Chem.AddHs(mol)
    desc = compute_descriptors(mol)
    lip_pass, violations = evaluate_lipinski(desc)
    qed_score = QED.qed(mol)
    admet = heuristic_admet(desc)
    image_path = None
    if image_dir:
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        img = Draw.MolToImage(mol, size=(300, 300))
        filename = os.path.join(image_dir, f"{smiles_to_safe_filename(smiles)}.png")
        img.save(filename)
        image_path = filename
    return AnalysisResult(
        smiles=smiles,
        mol=mol,
        descriptors=desc,
        lipinski_pass=lip_pass,
        lipinski_violations=violations,
        qed_score=qed_score,
        admet_predictions=admet,
        image_path=image_path,
    )


def smiles_to_safe_filename(smiles: str) -> str:
    """Convert a SMILES string into a filesystem‑safe filename stub."""
    # Replace characters that are illegal or problematic in filenames
    return (
        smiles.replace("/", "-")
        .replace("\\", "-")
        .replace("#", "hash")
        .replace("*", "star")
        .replace("?", "question")
        .replace(":", "-")
        .replace("<", "-")
        .replace(">", "-")
    )


def analyse_multiple(smiles_list: Iterable[str], image_dir: Optional[str] = None) -> pd.DataFrame:
    """Analyse a list of SMILES strings and return a DataFrame of results."""
    results = []
    for smi in smiles_list:
        try:
            res = analyse_smiles(smi, image_dir=image_dir)
            row = {**res.descriptors, "SMILES": smi, "QED": res.qed_score}
            row.update({f"Lipinski_{k}": v for k, v in res.lipinski_violations.items()})
            row["Lipinski_Pass"] = res.lipinski_pass
            row.update({f"ADMET_{k}": v for k, v in res.admet_predictions.items()})
            row["ImagePath"] = res.image_path
            results.append(row)
        except Exception as e:
            logger.error("Failed to analyse %s: %s", smi, e)
    return pd.DataFrame(results)
