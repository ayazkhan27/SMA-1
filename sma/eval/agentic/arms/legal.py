"""Legal/IP arm: patent classification over the Cooperative Patent Classification.

Entities are granted patents; each is annotated with the set of CPC subgroup codes
its examiner assigned (gold, from one week of USPTO grant full-text XML). The hard
query is a partial/imprecise technical profile (a few codes, climbed up the CPC
hierarchy, plus noise); the task is to retrieve the patent by its classification
signature. CPC's deep section->class->subclass->group->subgroup tree is the is-a
lattice (254k nodes), mounted via sma.ontology.load_cpc.
"""
from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET

from sma.ontology import load_cpc, mount

ROOT = pathlib.Path(__file__).resolve().parents[4]
CPC_DIR = ROOT / "data/raw/legal/cpc"
GRANT_XML = ROOT / "data/raw/patents/ipg161011.xml"
MIN_TERMS, MAX_TERMS = 7, 30


def _grant_blocks(path):
    """Yield each <us-patent-grant>...</us-patent-grant> block (the file is many
    concatenated XML documents, so we can't parse it as one tree)."""
    buf, inside = [], False
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("<us-patent-grant"):
                inside, buf = True, [line]
            elif inside:
                buf.append(line)
                if line.startswith("</us-patent-grant>"):
                    yield "".join(buf)
                    inside, buf = False, []


def _cpc_codes(grant_el) -> set[str]:
    """CPC subgroup codes from the bibliographic block only (not cited references)."""
    block = grant_el.find(".//us-bibliographic-data-grant/classifications-cpc")
    if block is None:
        return set()
    codes = set()
    for c in block.iter("classification-cpc"):
        try:
            sec = c.findtext("section", "").strip()
            cls = c.findtext("class", "").strip()
            sub = c.findtext("subclass", "").strip()
            mg = c.findtext("main-group", "").strip()
            sg = c.findtext("subgroup", "").strip()
        except AttributeError:
            continue
        if sec and cls and sub and mg and sg:
            codes.add(f"{sec}{cls}{sub}{mg}/{sg}")
    return codes


def load():
    mounted = mount(load_cpc(str(CPC_DIR), name="cpc"))
    known = mounted.graph.terms
    recs: dict[str, set[str]] = {}
    for block in _grant_blocks(GRANT_XML):
        try:
            g = ET.fromstring(block)
        except ET.ParseError:
            continue
        num = g.findtext(".//publication-reference//doc-number")
        if not num:
            continue
        codes = {c for c in _cpc_codes(g) if c in known}
        if MIN_TERMS <= len(codes) <= MAX_TERMS:
            recs[f"US{num}"] = codes
    return mounted, recs
