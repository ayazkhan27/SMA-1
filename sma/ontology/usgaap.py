"""Loader for the US-GAAP financial reporting taxonomy (XBRL presentation linkbase).

FIBO is a schema ontology with no public instance corpus, so the financial arm
uses US-GAAP instead: its concepts form a hierarchy via the presentation
linkbase's parent-child arcs (abstract statement headers subsume line items), and
SEC filings provide real gold (each filing reports a set of US-GAAP concepts).
This parses the core financial-statement presentation linkbases into an
:class:`OntologyGraph` (concept -> parent header).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .graph import OntologyGraph, Term

_PARENT_CHILD = "parent-child"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _attr(el, name: str):
    for k, v in el.attrib.items():
        if _local(k) == name:
            return v
    return None


def _concept(href: str) -> str:
    """'...#us-gaap_Revenues' -> 'Revenues'."""
    frag = href.rsplit("#", 1)[-1]
    return frag.split("_", 1)[1] if "_" in frag else frag


def _humanize(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def load_usgaap(path: str, name: str = "usgaap", pattern: str = "*.xml") -> OntologyGraph:
    root = Path(path)
    files = sorted(root.glob(pattern)) if root.is_dir() else [root]
    parents: dict[str, set[str]] = {}
    seen: set[str] = set()
    for f in files:
        try:
            tree = ET.parse(f)
        except ET.ParseError:
            continue
        for plink in tree.iter():
            if _local(plink.tag) != "presentationLink":
                continue
            loc: dict[str, str] = {}
            for el in plink:
                lt = _local(el.tag)
                if lt == "loc":
                    lab = _attr(el, "label"); href = _attr(el, "href")
                    if lab and href:
                        loc[lab] = _concept(href)
            for el in plink:
                if _local(el.tag) != "presentationArc":
                    continue
                if (_attr(el, "arcrole") or "").rsplit("/", 1)[-1] != _PARENT_CHILD:
                    continue
                pa, ch = loc.get(_attr(el, "from")), loc.get(_attr(el, "to"))
                if pa and ch and pa != ch:
                    parents.setdefault(ch, set()).add(pa)
                    seen.update((pa, ch))
    terms = {c: Term(id=c, name=_humanize(c), parents=tuple(sorted(parents.get(c, ()))))
             for c in sorted(seen)}
    return OntologyGraph(name=name, version="us-gaap-2024", terms=terms)
