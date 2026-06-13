"""Loader for the Cooperative Patent Classification (CPC) scheme XML.

CPC ships as one XML file per subclass (``cpc-scheme-A01B.xml`` ...), each a tree
of nested ``<classification-item>`` elements. The nesting IS the is-a hierarchy:
a classification-item nested inside another is a narrower category of it. We map
each item's ``<classification-symbol>`` to a :class:`Term` id, its
``<class-title>`` text to the name, and the enclosing item's symbol to its is-a
parent. This yields the deep (~250k node) golden taxonomy for the legal/IP arm.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .graph import OntologyGraph, Term


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _title(item: ET.Element) -> str:
    """Concatenate the text fragments under an item's direct <class-title>."""
    for child in item:
        if _local(child.tag) == "class-title":
            parts = [t.text.strip() for t in child.iter()
                     if _local(t.tag) == "text" and t.text and t.text.strip()]
            return "; ".join(parts)
    return ""


def _walk(item: ET.Element, parent_symbol: str, terms: dict[str, Term]) -> None:
    symbol = ""
    for child in item:
        if _local(child.tag) == "classification-symbol":
            symbol = (child.text or "").strip()
            break
    if symbol:
        existing = terms.get(symbol)
        parents = (parent_symbol,) if parent_symbol else ()
        if existing is None:
            terms[symbol] = Term(id=symbol, name=_title(item), parents=parents)
        elif parent_symbol and parent_symbol not in existing.parents:
            terms[symbol] = Term(id=symbol, name=existing.name or _title(item),
                                 parents=tuple(dict.fromkeys((*existing.parents, parent_symbol))))
    next_parent = symbol or parent_symbol
    for child in item:
        if _local(child.tag) == "classification-item":
            _walk(child, next_parent, terms)


def load_cpc(path: str, name: str = "cpc") -> OntologyGraph:
    """Load the CPC scheme from a directory of cpc-scheme-*.xml files (or one file)."""
    root_path = Path(path)
    files = sorted(root_path.glob("cpc-scheme-*.xml")) if root_path.is_dir() else [root_path]
    terms: dict[str, Term] = {}
    version = ""
    for f in files:
        try:
            tree = ET.parse(f)
        except ET.ParseError:
            continue
        scheme = tree.getroot()
        if not version:
            version = scheme.get("publication-date", "")
        for child in scheme:
            if _local(child.tag) == "classification-item":
                _walk(child, "", terms)
    return OntologyGraph(name=name, version=version, terms=terms)
