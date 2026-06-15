"""SMA-1 Model Context Protocol (MCP) server.

Exposes the structure-mapping memory as MCP tools so an MCP client (Codex CLI,
Claude, the OpenAI Agents SDK, ...) can:

  * mount curated ontologies (one shared core, many domains),
  * index a case base (records expressed as ontology term-ids),
  * retrieve *structurally-analogous* prior cases by logical structure
    (is-a subsumption + typed relations + rarity weighting), with a checkable
    structural citation, a cite-or-abstain decision, and an expectation-violation
    **novelty** flag.

This is the analogical-memory layer for a discovery loop: the LLM generates and
verifies; SMA grounds each step in structurally-analogous precedent and flags the
genuinely never-seen, which surface-similarity (vector RAG) cannot do.

Design notes
------------
* The engine (:class:`SmaEngine`) is import-light: it uses only the SMA core
  (``sma.ontology`` / ``sma.index`` / ``sma.sage``), NOT the eval baselines
  (bm25 / dense / hipporag) and NOT the ``mcp`` SDK -- so it is unit-testable
  without the transport dependency. ``mcp`` is imported lazily in
  :func:`build_server` / :func:`main`.
* Cite-or-abstain gates on the RAW structural grounding score (the codebase is
  explicit that the normalized confidence saturates and does not separate
  known/unknown). A per-ontology ``ground_threshold`` is calibrated offline; when
  unset there is no gate (every non-empty result is "grounded") -- calibrate it.

Run as a stdio MCP server::

    python -m sma.mcp          # or: sma-mcp

Configure a manifest of ontologies to auto-register via the ``SMA_MANIFEST`` env
var (see ``examples/sma_manifest.example.json``).
"""
from __future__ import annotations

import json
import gzip
import os
import pathlib
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Any

from sma.index.macfac import MacFacIndex
from sma.ontology import (
    MountedOntology,
    load_attack_stix,
    load_capec,
    load_cpc,
    load_cwe,
    load_mitre_xml,
    load_obo,
    load_ontology,
    load_owl,
    load_owl_dir,
    load_rdflib,
    load_usgaap,
    mount,
)
from sma.sage.pools import SagePool

# Format -> loader. "auto" dispatches obo/owl by file extension.
_LOADERS = {
    "auto": load_ontology,
    "obo": load_obo,
    "owl": load_owl,
    "owl_dir": load_owl_dir,
    "rdf": load_rdflib,
    "ttl": load_rdflib,
    "stix": load_attack_stix,
    "attack": load_attack_stix,
    "cpc": load_cpc,
    "xbrl": load_usgaap,
    "usgaap": load_usgaap,
    "cwe": load_cwe,
    "capec": load_capec,
    "mitre_xml": load_mitre_xml,
}

SUPPORTED_FORMATS = sorted(_LOADERS)

# Source/catalog format labels -> a loader key (rdflib covers ttl/jsonld/skos/rdfxml).
_FORMAT_ALIAS = {
    "ttl": "rdf", "turtle": "rdf", "jsonld": "rdf", "json-ld": "rdf", "skos": "rdf",
    "rdfxml": "rdf", "rdf-xml": "rdf", "nt": "rdf", "n3": "rdf", "xml": "auto",
}

# Bundled catalog of public ontologies (mounted on demand from their source URLs).
_CATALOG_PATH = pathlib.Path(__file__).with_name("data") / "ontology_catalog.json"

_SENTINEL = object()


def _norm_fmt(fmt: str | None) -> str:
    fmt = (fmt or "auto").lower()
    return _FORMAT_ALIAS.get(fmt, fmt)


def _load_graph(path: str, fmt: str | None):
    fmt = _norm_fmt(fmt)
    loader = _LOADERS.get(fmt)
    if loader is None:
        raise ValueError(
            f"unknown ontology format {fmt!r}; supported: {SUPPORTED_FORMATS} "
            f"(+ aliases {sorted(_FORMAT_ALIAS)})"
        )
    return loader(path)


def _envfloat(name: str, default: float | None) -> float | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass
class _Ont:
    """A registered ontology and (lazily) its mounted lattice + case index."""

    name: str
    path: str
    fmt: str = "auto"
    ground_threshold: float | None = None
    novelty_threshold: float = 0.5
    mounted: MountedOntology | None = None
    cases: list[dict] = field(default_factory=list)
    _index: Any = None
    _pool: Any = None
    _keymap: dict[str, str] = field(default_factory=dict)


class SmaEngine:
    """The MCP-free core: register/mount ontologies, index cases, retrieve, novelty.

    Mirrors ``sma.eval.agentic.memories.SmaMemory`` (build_case + MacFacIndex +
    SagePool) without importing the eval module's heavy baseline dependencies.
    """

    def __init__(self) -> None:
        self.onts: dict[str, _Ont] = {}
        self.default_ground_threshold = _envfloat("SMA_GROUND_THRESHOLD", None)
        self.default_novelty_threshold = _envfloat("SMA_NOVELTY_THRESHOLD", 0.5) or 0.5

    # -- registration / mounting -------------------------------------------
    def register(
        self,
        name: str,
        path: str,
        fmt: str = "auto",
        ground_threshold: float | None = None,
        novelty_threshold: float | None = None,
    ) -> None:
        self.onts[name] = _Ont(
            name=name,
            path=path,
            fmt=(fmt or "auto"),
            ground_threshold=(
                ground_threshold if ground_threshold is not None else self.default_ground_threshold
            ),
            novelty_threshold=(
                novelty_threshold if novelty_threshold is not None else self.default_novelty_threshold
            ),
        )

    def _get(self, name: str) -> _Ont:
        o = self.onts.get(name)
        if o is None:
            raise KeyError(f"ontology {name!r} is not registered; mount it first ({sorted(self.onts)})")
        return o

    def _ensure_mounted(self, name: str) -> _Ont:
        o = self._get(name)
        if o.mounted is None:
            o.mounted = mount(_load_graph(o.path, o.fmt))
        return o

    def mount_ontology(self, name: str, path: str, fmt: str = "auto") -> dict:
        self.register(name, path, fmt)
        o = self._ensure_mounted(name)
        return {
            "ontology": name,
            "format": o.fmt,
            "concepts": len(o.mounted.graph.terms),
            "indexed_cases": len(o.cases),
        }

    # -- indexing -----------------------------------------------------------
    def _rebuild(self, o: _Ont) -> None:
        o._keymap = {}
        o._pool = SagePool("mcp", assimilation_threshold=0.2)
        cases = []
        for c in o.cases:
            case = o.mounted.build_case(frozenset(c["term_ids"]), metadata={"key": c["key"]})
            o._keymap[case.case_id] = c["key"]
            cases.append(case)
            o._pool.assimilate(case)
        o._index = MacFacIndex(config=o.mounted.config, canon=o.mounted.canon)
        o._index.build(cases)

    def index_cases(self, name: str, cases: list[dict]) -> int:
        """Append cases (each ``{key, term_ids, text?}``) and (re)build the index."""
        o = self._ensure_mounted(name)
        for c in cases:
            if "key" not in c or "term_ids" not in c:
                raise ValueError("each case needs 'key' and 'term_ids'")
            o.cases.append({"key": str(c["key"]), "term_ids": list(c["term_ids"]), "text": c.get("text", "")})
        self._rebuild(o)
        return len(o.cases)

    # -- encoding (no LLM): raw text -> ontology term-ids -------------------
    def encode_text(self, name: str, text: str, max_terms: int = 20) -> dict:
        """Deterministic lexical match of term NAMES in ``text`` -> term-ids.

        A starter encoder (no LLM, honoring the no-LLM-in-extraction principle).
        Replace with a domain-specific encoder for production recall.
        """
        o = self._ensure_mounted(name)
        hay = text.lower()
        hits: list[tuple[str, str]] = []
        for tid, term in o.mounted.graph.terms.items():
            nm = (getattr(term, "name", "") or "").strip().lower()
            # token-boundary match (robust to punctuation); deterministic, no LLM
            if len(nm) >= 3 and re.search(rf"\b{re.escape(nm)}\b", hay):
                hits.append((tid, term.name))
        hits.sort(key=lambda h: -len(h[1]))  # prefer more specific (longer) names
        hits = hits[:max_terms]
        return {"ontology": name, "term_ids": [h[0] for h in hits], "matched_names": [h[1] for h in hits]}

    def _resolve_terms(self, o: _Ont, term_ids, text) -> frozenset[str]:
        if term_ids:
            return frozenset(term_ids)
        if text:
            return frozenset(self.encode_text(o.name, text)["term_ids"])
        raise ValueError("provide either term_ids or text")

    # -- retrieval / novelty ------------------------------------------------
    def retrieve(
        self,
        name: str,
        term_ids=None,
        text: str = "",
        k: int = 5,
        ground_threshold=_SENTINEL,
    ) -> dict:
        o = self._ensure_mounted(name)
        tids = self._resolve_terms(o, term_ids, text)
        if not tids:
            return {"ontology": name, "abstain": True, "reason": "no query term-ids (encoding matched nothing)",
                    "citations": [], "novelty": None, "novelty_flag": False, "query_term_ids": []}
        if o._index is None:
            return {"ontology": name, "abstain": True, "reason": "no cases indexed",
                    "citations": [], "novelty": None, "novelty_flag": False, "query_term_ids": sorted(tids)}
        qc = o.mounted.build_case(tids)
        res = o._index.retrieve(qc, k=k, shortlist=80, fac_budget=40)
        nov = float(o._pool.expectation_violation(qc))
        thr = o.ground_threshold if ground_threshold is _SENTINEL else ground_threshold
        top = max((r.score for r in res), default=0.0) or 1.0
        citations = [
            {
                "id": o._keymap.get(r.case_id, ""),
                "score": round(float(r.score), 4),
                "confidence": round(min(max(r.score / top, 0.0), 1.0), 4),
                "rank": i,
            }
            for i, r in enumerate(res, 1)
        ]
        grounded = bool(res) and (thr is None or res[0].score >= thr)
        return {
            "ontology": name,
            "query_term_ids": sorted(tids),
            "abstain": not grounded,
            "abstain_threshold": thr,
            "novelty": round(nov, 4),
            "novelty_flag": nov >= o.novelty_threshold,
            "citations": citations if grounded else [],
            "note": (
                None if grounded
                else "top grounding score below threshold -> no structural precedent; do not fabricate an answer"
            ),
        }

    def novelty(self, name: str, term_ids=None, text: str = "") -> dict:
        o = self._ensure_mounted(name)
        tids = self._resolve_terms(o, term_ids, text)
        if o._pool is None or not tids:
            return {"ontology": name, "novelty": None, "novelty_flag": False,
                    "reason": "novelty cannot be scored (no indexed cases or no query term-ids)",
                    "query_term_ids": sorted(tids)}
        qc = o.mounted.build_case(tids)
        nov = float(o._pool.expectation_violation(qc))
        return {
            "ontology": name,
            "query_term_ids": sorted(tids),
            "novelty": round(nov, 4),
            "novelty_flag": nov >= o.novelty_threshold,
        }

    def list_ontologies(self) -> dict:
        return {
            "ontologies": [
                {
                    "name": o.name,
                    "format": o.fmt,
                    "mounted": o.mounted is not None,
                    "concepts": (len(o.mounted.graph.terms) if o.mounted is not None else None),
                    "indexed_cases": len(o.cases),
                    "ground_threshold": o.ground_threshold,
                    "novelty_threshold": o.novelty_threshold,
                }
                for o in self.onts.values()
            ],
            "supported_formats": SUPPORTED_FORMATS,
        }

    # -- public-ontology catalog (mount any of ~200+ on demand) -------------
    def load_catalog(self) -> list[dict]:
        """The bundled catalog of public ontologies (cached after first read)."""
        if getattr(self, "_catalog", None) is None:
            if _CATALOG_PATH.exists():
                self._catalog = json.loads(_CATALOG_PATH.read_text()).get("ontologies", [])
            else:
                self._catalog = []
        return self._catalog

    def list_catalog(self, category: str | None = None) -> dict:
        """Browse the catalog (does NOT download anything). Filter by category substring."""
        cat = (category or "").lower()
        items = [
            {"id": e["id"], "name": e["name"], "category": e["category"],
             "format": e["format"], "license": e.get("license", ""),
             "mounted": e["id"] in self.onts}
            for e in self.load_catalog()
            if not cat or cat in e["category"].lower()
        ]
        cats = sorted({e["category"] for e in self.load_catalog()})
        return {"count": len(items), "categories": cats, "ontologies": items}

    def search_catalog(self, query: str, limit: int = 25) -> dict:
        """Find catalog ontologies whose id/name/category matches ``query`` (substring, case-insensitive)."""
        q = query.lower()
        hits = [
            {"id": e["id"], "name": e["name"], "category": e["category"], "format": e["format"]}
            for e in self.load_catalog()
            if q in e["id"].lower() or q in e["name"].lower() or q in e["category"].lower()
        ]
        return {"query": query, "count": len(hits), "ontologies": hits[:limit]}

    def _cache_dir(self) -> pathlib.Path:
        d = pathlib.Path(os.environ.get("SMA_CACHE_DIR", pathlib.Path.home() / ".cache" / "sma" / "ontologies"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    _ONTO_EXT = (".obo", ".owl", ".ttl", ".rdf", ".rdfs", ".nt", ".n3", ".json", ".jsonld", ".xml", ".skos")

    def _resolve_file(self, raw: pathlib.Path, d: pathlib.Path) -> pathlib.Path:
        """Decompress .gz / extract .zip to the usable ontology file (keep inner extension)."""
        name = raw.name.lower()
        if name.endswith(".gz") and not name.endswith(".tar.gz"):
            out = d / raw.name[:-3]
            with gzip.open(raw, "rb") as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            return out
        if name.endswith(".zip"):
            with zipfile.ZipFile(raw) as z:
                members = [n for n in z.namelist() if not n.endswith("/")]
                cands = [n for n in members if n.lower().endswith(self._ONTO_EXT)] or members
                if not cands:
                    raise RuntimeError(f"zip for archive contained no usable file")
                pick = max(cands, key=lambda n: z.getinfo(n).file_size)
                out = d / pathlib.Path(pick).name
                with z.open(pick) as zi, open(out, "wb") as fo:
                    shutil.copyfileobj(zi, fo)
            return out
        return raw

    def _fetch(self, url: str, ont_id: str, fmt: str) -> str:
        """Download (lazily, one-time) + decompress an ontology to the cache; return its path."""
        d = self._cache_dir() / ont_id
        d.mkdir(parents=True, exist_ok=True)
        marker = d / ".resolved"
        if marker.exists():
            p = marker.read_text().strip()
            if p and pathlib.Path(p).exists():
                return p
        base = pathlib.Path(url.split("?")[0]).name or f"{ont_id}.dat"
        raw = d / base
        if not (raw.exists() and raw.stat().st_size > 0):
            req = urllib.request.Request(url, headers={"User-Agent": "sma-mcp/ontology-catalog"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r, open(raw, "wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
            except (urllib.error.URLError, OSError) as exc:
                raw.unlink(missing_ok=True)
                raise RuntimeError(f"could not download {ont_id} from {url}: {exc}") from exc
        usable = self._resolve_file(raw, d)
        marker.write_text(str(usable))
        return str(usable)

    def mount_from_catalog(self, ont_id: str) -> dict:
        """Fetch (cache) and mount a catalog ontology by id. The download is lazy and one-time."""
        entry = next((e for e in self.load_catalog() if e["id"] == ont_id), None)
        if entry is None:
            raise KeyError(f"{ont_id!r} is not in the catalog (try search_catalog)")
        if not entry.get("mountable", True):
            raise RuntimeError(
                f"{ont_id!r} is catalogued for awareness but not auto-mountable "
                f"({entry.get('notes', 'repo/API/page-only')}). Obtain it from "
                f"{entry.get('homepage') or entry['url']}, then mount_ontology(path=...)."
            )
        path = self._fetch(entry["url"], ont_id, entry["format"])
        self.register(ont_id, path, entry["format"])
        o = self._ensure_mounted(ont_id)
        return {"ontology": ont_id, "name": entry["name"], "category": entry["category"],
                "format": entry["format"], "concepts": len(o.mounted.graph.terms), "license": entry.get("license", "")}


def load_manifest(engine: SmaEngine, path: str) -> None:
    """Register ontologies (and optionally eager-index cases) from a JSON manifest.

    ``{"ontologies": [{"name","path","format"?,"ground_threshold"?,"novelty_threshold"?}],
       "cases": {"<ontology>": [{"key","term_ids","text"?}, ...]}}``
    """
    with open(path) as fh:
        data = json.load(fh)
    for o in data.get("ontologies", []):
        engine.register(
            o["name"], o["path"], o.get("format", "auto"),
            o.get("ground_threshold"), o.get("novelty_threshold"),
        )
    for name, cases in (data.get("cases") or {}).items():
        if cases:
            engine.index_cases(name, cases)


def build_server(engine: SmaEngine | None = None):
    """Construct the FastMCP server (imports the ``mcp`` SDK lazily)."""
    from mcp.server.fastmcp import FastMCP

    engine = engine or SmaEngine()
    manifest = os.environ.get("SMA_MANIFEST")
    if manifest and os.path.exists(manifest):
        load_manifest(engine, manifest)

    server = FastMCP("sma-1")

    @server.tool()
    def list_ontologies() -> dict:
        """List registered ontologies, their concept counts, indexed-case counts, and supported source formats."""
        return engine.list_ontologies()

    @server.tool()
    def mount_ontology(name: str, path: str, format: str = "auto") -> dict:
        """Register and mount a curated ontology as a structural lattice.

        `format` is auto-detected for OBO/OWL; pass one of the supported formats
        (stix, cpc, xbrl, cwe, capec, mitre_xml, rdf, ...) otherwise. Returns the
        concept count once mounted."""
        return engine.mount_ontology(name, path, format)

    @server.tool()
    def index_cases(ontology: str, cases: list[dict]) -> dict:
        """Add cases to an ontology's memory. Each case is {"key","term_ids",["text"]}.

        `key` is what gets cited back; `term_ids` are the ontology term-ids that
        encode the case's structure. Returns the total number of indexed cases."""
        total = engine.index_cases(ontology, cases)
        return {"ontology": ontology, "indexed_cases": total}

    @server.tool()
    def encode_text(ontology: str, text: str, max_terms: int = 20) -> dict:
        """Deterministically map free text to ontology term-ids via term-name matching (no LLM).

        Use to turn an abstract/finding into `term_ids` before retrieve/novelty.
        This is a starter encoder; swap in a domain encoder for better recall."""
        return engine.encode_text(ontology, text, max_terms)

    @server.tool()
    def retrieve(
        ontology: str,
        text: str = "",
        term_ids: list[str] | None = None,
        k: int = 5,
        ground_threshold: float | None = None,
    ) -> dict:
        """Retrieve STRUCTURALLY-ANALOGOUS prior cases by logical structure (not surface similarity).

        Provide `term_ids` (preferred) or `text` (encoded on the fly). Returns ranked
        citations {id, score, confidence, rank}, a calibrated cite-or-abstain decision
        (`abstain`=True means no structural precedent -- do NOT fabricate), and a
        `novelty` score (high = a case unlike anything indexed)."""
        gt = _SENTINEL if ground_threshold is None else ground_threshold
        return engine.retrieve(ontology, term_ids=term_ids, text=text, k=k, ground_threshold=gt)

    @server.tool()
    def novelty(ontology: str, text: str = "", term_ids: list[str] | None = None) -> dict:
        """Score how novel a case is vs. the indexed memory (expectation-violation).

        High `novelty` / `novelty_flag`=True means the case is structurally unlike
        anything seen -- a candidate genuinely-new pattern."""
        return engine.novelty(ontology, term_ids=term_ids, text=text)

    @server.tool()
    def list_catalog(category: str = "") -> dict:
        """Browse the bundled catalog of public ontologies (no download).

        ~200+ open ontologies across science, medicine, chemistry, cyber, finance,
        legal, geoscience and more. Filter by `category` substring. Use this to
        discover what SMA can specialise in, then `mount_from_catalog`."""
        return engine.list_catalog(category or None)

    @server.tool()
    def search_catalog(query: str, limit: int = 25) -> dict:
        """Search the ontology catalog by id/name/category substring (no download)."""
        return engine.search_catalog(query, limit)

    @server.tool()
    def mount_from_catalog(ontology_id: str) -> dict:
        """Fetch (cache once) and mount a public ontology from the catalog by its id.

        e.g. mount_from_catalog("hp") for the Human Phenotype Ontology. The download
        is lazy and cached under SMA_CACHE_DIR (default ~/.cache/sma). After mounting,
        index your cases and use retrieve/novelty."""
        return engine.mount_from_catalog(ontology_id)

    return server, engine


def main() -> None:
    server, _ = build_server()
    server.run()


if __name__ == "__main__":
    main()
