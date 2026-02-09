"""
Streamlit app for exploring novel cannabinoids from the Czech‑CBD project.

This application wraps the scraping and RDKit analysis routines from the
``czech_cbd_analysis`` package in an interactive web interface.  Users can
browse scraped products, inspect molecular structures and physicochemical
properties, run simple ADMET heuristics and propose derivatives.  The goal
is to present complex cheminformatics analysis in a clear, minimalist
environment that adheres to established UI/UX principles.

Usage::

    streamlit run app/app.py

The app will look for a ``products.csv`` file in the repository root.  If
present, it will be loaded as the product catalogue.  Otherwise, the
scraper can be invoked manually from the sidebar (network permitting).
This design allows offline exploration of pre‑scraped data while still
supporting fresh data collection when possible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rdkit_analysis import analyse_smiles
from generate_derivatives import propose_derivatives
from llm_integration import translate, summarise, chat_completion  # new LLM helpers

# Optional import; scraping may not work if network access is blocked
try:
    from scrape_czech_cbd import crawl_urls
except Exception:
    crawl_urls = None  # type: ignore


def load_products(csv_path: str | os.PathLike) -> Optional[pd.DataFrame]:
    """Load product data from a CSV file if it exists.

    Parameters
    ----------
    csv_path: str or Path
        Path to the products CSV.

    Returns
    -------
    DataFrame or None
        Loaded data or None if the file does not exist.
    """
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def main() -> None:
    st.set_page_config(
        page_title="Czech CBD Cannabinoid Explorer",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Czech CBD Cannabinoid Explorer")
    st.markdown(
        """
        This app allows you to browse novel cannabinoid products scraped from
        **Czech‑CBD**, analyse their molecular structures using RDKit and explore
        simple ADMET heuristics and derivative proposals.  The interface has
        been designed to be minimal and intuitive, with clear separation of
        data, analysis and warnings.  For details on the underlying pipeline,
        see the project documentation.
        """
    )

    # Sidebar navigation
    pages = ["Product Overview", "Molecule Analysis", "Derivative Explorer", "LLM Tools"]
    choice = st.sidebar.radio("Navigate", pages)

    # Try loading existing product catalogue
    repo_root = Path(__file__).resolve().parent.parent
    products_csv = repo_root / "products.csv"
    products_df = load_products(products_csv)

    # Provide scraping option if CSV is missing and crawler is available
    if products_df is None:
        st.sidebar.warning(
            "No `products.csv` file found.  You can load data via the scraper "
            "(requires network access) or upload a CSV in the sidebar."
        )
        uploaded = st.sidebar.file_uploader(
            "Upload products.csv", type=["csv"], key="upload_csv"
        )
        if uploaded is not None:
            products_df = pd.read_csv(uploaded)
        elif crawl_urls is not None:
            if st.sidebar.button("Run scraper (may take a while)"):
                st.sidebar.info(
                    "Scraping Czech‑CBD... please be patient (network access required)"
                )
                try:
                    # Example list of URLs; in practice you might crawl categories
                    seed_urls: List[str] = [
                        "https://www.czech-cbd.cz/10-oh-hhc-brownies",
                        "https://www.czech-cbd.cz/thcv-honey",
                    ]
                    products_df = crawl_urls(seed_urls)
                    products_df.to_csv(products_csv, index=False)
                    st.sidebar.success("Scraping complete.  Data saved to products.csv.")
                except Exception as e:
                    st.sidebar.error(f"Scraping failed: {e}")

    # Upload molecules from user input or selection
    if choice == "Product Overview":
        st.header("Product Catalogue")
        if products_df is not None and not products_df.empty:
            st.dataframe(products_df)
            st.info(
                "Select a row from the above table to analyse its molecule in the "
                "next page."
            )
        else:
            st.warning("No product data available.  Please upload or scrape data.")

    elif choice == "Molecule Analysis":
        st.header("Molecule Analysis")
        # Choose SMILES input method
        with st.sidebar.form("analysis_form"):
            smi_input = st.text_input(
                "Enter SMILES string", value="", help="Paste a canonical SMILES here."
            )
            submit_analyse = st.form_submit_button("Analyse")
        selected_smiles = None
        if products_df is not None and not products_df.empty:
            st.subheader("Or pick from products")
            # Provide selection of molecules for which we know SMILES (optional column)
            if "smiles" in products_df.columns:
                smi_options = products_df["smiles"].dropna().unique().tolist()
                if smi_options:
                    selected_smiles = st.selectbox("Select SMILES", options=smi_options)
        # Determine which SMILES to analyse
        input_smiles = None
        if submit_analyse and smi_input:
            input_smiles = smi_input.strip()
        elif selected_smiles:
            input_smiles = selected_smiles
        if input_smiles:
            try:
                result = analyse_smiles(input_smiles, image_dir=str(repo_root / "molecule_images"))
                st.success("Molecule analysed successfully.")
                # Display 2D depiction
                if result.image_path and os.path.exists(result.image_path):
                    st.image(result.image_path, caption=f"Structure of {input_smiles}")
                # Display descriptors
                st.subheader("Physicochemical descriptors")
                desc_df = pd.DataFrame(result.descriptors, index=[0]).T
                desc_df.columns = ["Value"]
                st.table(desc_df)
                # Lipinski
                st.subheader("Lipinski rule of five")
                lip_df = pd.DataFrame(result.lipinski_violations, index=["Pass"])
                st.table(lip_df)
                st.markdown(f"**Passes all rules:** {result.lipinski_pass}")
                # QED and ADMET
                st.subheader("Drug‑likeness and heuristic ADMET")
                st.markdown(f"**QED score:** {result.qed_score:.3f}")
                admet_df = pd.DataFrame(result.admet_predictions, index=["Prediction"]).T
                st.table(admet_df)
                st.info(
                    "These ADMET predictions are heuristic and based on simple rules. "
                    "They should not be used for clinical or regulatory decisions."
                )
            except Exception as e:
                st.error(f"Failed to analyse molecule: {e}")
        else:
            st.write("Enter a SMILES string or select one from the products to begin.")

    elif choice == "Derivative Explorer":
        st.header("Derivative Explorer")
        st.markdown(
            "Generate simple structural variants of a molecule by modifying side‑chain "
            "lengths and introducing hydroxyl groups.  This tool is illustrative and does "
            "not guarantee synthetic feasibility."
        )
        with st.sidebar.form("deriv_form"):
            smi_deriv = st.text_input(
                "SMILES string for derivative generation", value="", key="deriv_input"
            )
            submit_deriv = st.form_submit_button("Generate derivatives")
        if submit_deriv and smi_deriv:
            try:
                derivatives = propose_derivatives(smi_deriv.strip())
                st.success(f"Generated {len(derivatives)} derivative(s)")
                for smi in derivatives:
                    st.markdown(f"- {smi}")
            except Exception as e:
                st.error(f"Failed to generate derivatives: {e}")
        else:
            st.info("Enter a SMILES string above and click Generate derivatives.")

    elif choice == "LLM Tools":
        st.header("LLM Tools: Translation, Summarisation and Q&A")
        st.markdown(
            "Use large language models to translate Czech text to English, summarise long descriptions or ask questions about cannabinoids.  Enter your API key below to authenticate.  Note that all responses are generated by AI and should be verified against reliable sources."
        )
        api_key = st.text_input(
            "E‑INFRA AI API Key", type="password", help="Enter your personal API key (sk-...)"
        )
        input_text = st.text_area(
            "Input text", 
            help="Paste Czech product descriptions or scientific text here for translation or summarisation."
        )
        col1, col2 = st.columns(2)
        translation_result = None
        summary_result = None
        if col1.button("Translate to English"):
            if not api_key:
                st.error("Please enter an API key.")
            elif not input_text.strip():
                st.error("Please provide text to translate.")
            else:
                try:
                    translation_result = translate(input_text.strip(), api_key=api_key)
                except Exception as e:
                    st.error(f"Translation failed: {e}")
        if col2.button("Summarise"):
            if not api_key:
                st.error("Please enter an API key.")
            elif not input_text.strip():
                st.error("Please provide text to summarise.")
            else:
                try:
                    summary_result = summarise(input_text.strip(), api_key=api_key)
                except Exception as e:
                    st.error(f"Summarisation failed: {e}")
        if translation_result:
            st.subheader("Translation")
            st.write(translation_result)
        if summary_result:
            st.subheader("Summary")
            st.write(summary_result)
        # Q&A section
        st.subheader("Ask a Question")
        question = st.text_area(
            "Question", 
            value="", 
            key="question_area",
            help="Ask a question about cannabinoids or their properties."
        )
        if st.button("Ask LLM"):
            if not api_key:
                st.error("Please enter an API key.")
            elif not question.strip():
                st.error("Please enter a question.")
            else:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a scientific assistant specialising in novel cannabinoids. "
                                "Provide accurate, concise answers based on current scientific knowledge. "
                                "If information is lacking or uncertain, state that explicitly."
                            ),
                        },
                        {"role": "user", "content": question.strip()},
                    ]
                    answer = chat_completion(messages, api_key=api_key, model="gpt-oss-120b")
                    st.subheader("Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"LLM query failed: {e}")


if __name__ == "__main__":
    main()
