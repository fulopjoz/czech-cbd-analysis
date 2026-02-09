"""
scrape_czech_cbd.py
====================

This module provides a small crawler for the Czech‑CBD e‑commerce site.

The goal of the scraper is to extract high‑level information about novel
cannabinoid products (e.g. 10‑OH‑HHC, THCV, THCO, HHC‑P, EPN, THCF) sold
on ``https://www.czech-cbd.cz``.  It fetches product pages, parses their
content and consolidates the results into a ``pandas.DataFrame``.  The
resulting dataset can then be passed to downstream analysis routines in
``rdkit_analysis.py``.

**Implementation notes**
-----------------------

* Network access restrictions:  The environment used to execute this code
  may block direct HTTP requests to external websites.  The ``fetch_page``
  function therefore allows injecting pre‑downloaded HTML from disk.  If
  network access is permitted, it will gracefully fall back to ``requests``.

* Parsing:  Product pages on Czech‑CBD are served as standard HTML.  We
  use BeautifulSoup to extract the product name, price, description,
  composition and active cannabinoids.  The parser is brittle; if the
  site’s layout changes it may require adjustment.

* Identification of cannabinoids:  The function ``identify_cannabinoids``
  searches for known cannabinoid keywords within the product description.
  It returns a list of discovered molecules which can be cross‑referenced
  against the curated list in the project plan.

* Usage:  Execute this module as a script to crawl a list of product
  URLs and write the results to a CSV file.  For example:

  ``python scrape_czech_cbd.py --output products.csv``

This script is provided as a research tool and is not intended for
commercial use.  Please respect the website’s terms of service and
robots.txt when scraping.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterable

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError as e:
    raise ImportError(
        "BeautifulSoup4 is required to run this script. Install via `pip install beautifulsoup4`."
    ) from e

try:
    import requests
except ImportError as e:
    raise ImportError(
        "The requests library is required to run this script. Install via `pip install requests`."
    ) from e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Product:
    """Dataclass representing a scraped cannabinoid product."""

    name: str
    url: str
    price: Optional[str] = None
    description: Optional[str] = None
    composition: Optional[str] = None
    cannabinoids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "name": self.name,
            "url": self.url,
            "price": self.price,
            "description": self.description,
            "composition": self.composition,
            "cannabinoids": ", ".join(self.cannabinoids),
        }


def fetch_page(url: str, timeout: int = 15, headers: Optional[Dict[str, str]] = None, html_override: Optional[str] = None) -> str:
    """Fetch a page via HTTP and return its HTML content as a string.

    If ``html_override`` is provided, it will be returned instead of
    performing a network request.  This is useful for unit tests or
    offline development when network access is restricted.

    Parameters
    ----------
    url: str
        The URL to fetch.
    timeout: int
        Timeout in seconds for the HTTP request.
    headers: dict or None
        Optional HTTP headers.  A desktop browser User‑Agent is used by
        default to increase the likelihood that the server will respond.
    html_override: str or None
        Pre‑downloaded HTML to return instead of performing an HTTP
        request.

    Returns
    -------
    str
        The HTML content of the page.
    """
    if html_override is not None:
        return html_override
    ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.5735.90 Safari/537.36"
    )
    default_headers = {"User-Agent": ua}
    if headers:
        default_headers.update(headers)
    try:
        response = requests.get(url, timeout=timeout, headers=default_headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return ""


def identify_cannabinoids(text: str) -> List[str]:
    """Identify known cannabinoid names within a text block.

    Parameters
    ----------
    text: str
        The text to search.

    Returns
    -------
    List[str]
        A list of cannabinoid identifiers found in ``text``.  Matches are
        case‑insensitive.
    """
    if not text:
        return []
    text_lower = text.lower()
    keywords = [
        "10-oh-hhc",
        "10‑oh‑hhc",
        "10-oh hhc",
        "thcv",
        "thc-v",
        "thco",
        "thc-o",
        "hhc-p",
        "hhcp",
        "epn",
        "thc-f",
        "thcf",
        "nl-1",
    ]
    found = []
    for kw in keywords:
        if kw in text_lower:
            # normalise hyphens and case
            canonical = kw.replace("\u2011", "-").replace("\u2010", "-").lower()
            if canonical not in found:
                found.append(canonical)
    return found


def parse_product_page(html: str, url: str) -> Optional[Product]:
    """Parse a product page to extract fields of interest.

    Parameters
    ----------
    html: str
        The HTML content of the product page.
    url: str
        The URL of the product page (for reference in the Product record).

    Returns
    -------
    Product or None
        A ``Product`` instance containing the parsed data, or None if
        parsing failed.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Extract product name
    title_tag = soup.find("h1", class_=re.compile("product-detail-name|page-title"))
    name = None
    if title_tag:
        name = title_tag.get_text(strip=True)
    else:
        # Fallback: use the page title
        if soup.title:
            name = soup.title.get_text(strip=True)
    if not name:
        logger.warning("Could not find product name on %s", url)
        return None

    # Extract price
    price = None
    price_tag = soup.find(class_=re.compile("price-final|product-price"))
    if price_tag:
        price = price_tag.get_text(strip=True)

    # Extract description and composition
    description = ""
    composition = ""
    # Many product pages embed description in a div with itemprop="description"
    desc_tag = soup.find(attrs={"itemprop": "description"})
    if desc_tag:
        description = desc_tag.get_text(separator="\n", strip=True)
    # Look for composition or ingredients sections
    # Search for paragraphs containing keywords
    sections = soup.find_all(["p", "div"], string=True)
    for sec in sections:
        text = sec.get_text(strip=True)
        if not composition and re.search(r"(slo\u017een\u00ed|ingredients)", text, re.IGNORECASE):
            composition = text
        description += "\n" + text

    cannabinoids = identify_cannabinoids(description + " " + composition)
    return Product(
        name=name,
        url=url,
        price=price,
        description=description.strip() or None,
        composition=composition.strip() or None,
        cannabinoids=cannabinoids,
    )


def scrape_products(urls: Iterable[str], html_dir: Optional[str] = None) -> List[Product]:
    """Scrape multiple product URLs and return a list of ``Product`` instances.

    Parameters
    ----------
    urls: iterable of str
        Collection of product page URLs to scrape.
    html_dir: str or None
        If provided, a directory path containing pre‑saved HTML files.  Each
        filename should match the slug of the URL (e.g., ``thc-f-vape.html``)
        and will be used instead of performing network requests.

    Returns
    -------
    List[Product]
        A list of successfully parsed products.  Any pages that could not
        be fetched or parsed will be skipped with a warning.
    """
    products: List[Product] = []
    for url in urls:
        html = None
        if html_dir:
            # Determine filename by stripping scheme and non‑filename characters
            slug = re.sub(r"[^a-zA-Z0-9-]", "-", url.split("/")[-1])
            path = os.path.join(html_dir, slug + ".html")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    html = f.read()
        logger.info("Fetching %s", url)
        page = fetch_page(url, html_override=html)
        if not page:
            logger.warning("Skipping %s due to fetch failure", url)
            continue
        product = parse_product_page(page, url)
        if product:
            products.append(product)
        else:
            logger.warning("Failed to parse product page: %s", url)
    return products


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape Czech‑CBD product pages into a CSV table")
    parser.add_argument(
        "urls",
        nargs="*",
        help="List of product page URLs to scrape.  If omitted, a default set for novel cannabinoids will be used.",
    )
    parser.add_argument(
        "--output",
        default="products.csv",
        help="Path to the output CSV file (default: products.csv)",
    )
    parser.add_argument(
        "--html-dir",
        default=None,
        help="Directory containing pre‑downloaded HTML files to use instead of network requests",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Provide a default list of URLs for demonstration purposes.
    default_urls = [
        # These URLs correspond to novel cannabinoid products observed during the literature review.
        # They may no longer be valid – update as necessary.
        "https://www.czech-cbd.cz/thc-f-edibles/",
        "https://www.czech-cbd.cz/epn-cartridge/",
        "https://www.czech-cbd.cz/thc-o-cookies/",
        "https://www.czech-cbd.cz/hhc-p-cookies/",
        "https://www.czech-cbd.cz/thcv-honey/",
        "https://www.czech-cbd.cz/10-oh-hhc-vape/",
    ]
    urls = args.urls if args.urls else default_urls
    products = scrape_products(urls, html_dir=args.html_dir)
    if not products:
        logger.error("No products were scraped. Exiting.")
        return 1
    df = pd.DataFrame([p.to_dict() for p in products])
    df.to_csv(args.output, index=False)
    logger.info("Wrote %d products to %s", len(products), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
