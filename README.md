# Czech‑CBD Analysis Project

This repository contains a lightweight toolkit for scraping, analysing and
exploring the novel cannabinoids sold on the [Czech‑CBD](https://www.czech-cbd.cz) e‑commerce site.  It was developed as part of a research project to assess the structural properties and drug‑likeness of semi‑synthetic cannabinoids such as **10‑OH‑HHC**, **THCV**, **THCO**, **HHC‑P**, **EPN** and **THC‑F**.

> **Important:** The scripts and analyses contained in this project are for educational and research purposes only.  They do **not** constitute medical or legal advice.  Many of the compounds discussed here may be unregulated or illegal in various jurisdictions and could pose health risks.  Do **not** attempt to synthesise or distribute these substances.

## Components

### `scrape_czech_cbd.py`

This script provides a simple web scraper for individual product pages on the Czech‑CBD site.  Given a list of URLs, it attempts to extract:

* **Product name**
* **Price** (if available)
* **Description** and **composition** text
* **Active cannabinoids**, identified via keyword search

The scraper uses `requests` and `BeautifulSoup`.  Because network access may be restricted in some environments, it supports an `--html-dir` argument allowing you to provide pre‑downloaded HTML files instead of fetching pages at runtime.  Running the script outputs a CSV table containing the scraped data.

Usage example:

```bash
python scrape_czech_cbd.py \
  https://www.czech-cbd.cz/thc-f-edibles/ \
  https://www.czech-cbd.cz/thc-o-cookies/ \
  --output products.csv
```

### `rdkit_analysis.py`

This module leverages the RDKit chemistry library to compute physicochemical descriptors, evaluate drug‑likeness via Lipinski’s rules and QED, and generate heuristic ADMET predictions.  It defines an `AnalysisResult` dataclass and helper functions:

* `analyse_smiles(smiles, image_dir=None)` – Takes a SMILES string and returns descriptors, Lipinski evaluation, QED and ADMET labels.  Optionally saves a PNG depiction.
* `analyse_multiple(smiles_list, image_dir=None)` – Vectorised analysis for lists of SMILES strings, returning a `pandas.DataFrame`.

Example:

```python
from rdkit_analysis import analyse_smiles

smiles = "O=C(Oc2cc(cc1OC([C@@H]3CC/C(=C\\[C@H]3c12)C)(C)C)CCCCC)C"  # THCO
result = analyse_smiles(smiles)
print(result.descriptors)
print("Lipinski pass?", result.lipinski_pass)
print("QED score", result.qed_score)
```

### `generate_derivatives.py`

Contains simple heuristics for proposing structural analogues of cannabinoids.  It demonstrates how to:

* Modify the length of terminal alkyl side chains (`propose_chain_variants`)
* Introduce hydroxyl groups onto terminal methyl groups (`propose_hydroxylated`)
* Combine transformations (`propose_derivatives`)

These functions return lists of SMILES strings representing candidate derivatives.  They are intentionally rudimentary and should be treated as a starting point rather than a complete enumeration tool.

## Usage workflow

1. **Scrape product data** – Identify and scrape relevant product pages using `scrape_czech_cbd.py`.  Inspect the CSV to confirm that cannabinoids have been correctly extracted.
2. **Assemble molecular identifiers** – For each cannabinoid of interest, determine its SMILES string.  Use trusted sources (PubChem, ChEMBL, scientific literature) or derive from known scaffolds.  See the plan document for guidance.
3. **Analyse molecules** – Use `rdkit_analysis.analyse_multiple` to compute descriptors, QED and ADMET predictions for your set of SMILES.  Optionally generate images.
4. **Propose derivatives** – Apply functions from `generate_derivatives` to explore hypothetical structural modifications.  Feed the resulting SMILES back into `rdkit_analysis` to evaluate their properties.
5. **Develop models** – Extend the heuristics by incorporating public ADMET datasets (e.g. Tox21) and training machine‑learning models using scikit‑learn.  Integrate the evaluation metrics described in the plan.

## Requirements

* Python 3.8+
* [`rdkit`](https://www.rdkit.org) – installed as a Python package with the relevant chemistry binaries
* `pandas`
* `requests` and `beautifulsoup4` (for scraping)
* `scikit-learn` (optional, for future model development)

These dependencies are available in the current environment.  If running outside, install them via pip:

```bash
pip install rdkit-pypi pandas requests beautifulsoup4 scikit-learn
```

## License

This project is released under the MIT License.  See `LICENSE` for details.
