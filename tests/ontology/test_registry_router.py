"""Tests for the ontology registry and domain router."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from sma.ontology.registry import OntologyEntry, OntologyRegistry, _infer_format
from sma.ontology.router import DomainRouter

REAL_HPO = Path("/mnt/zephyr27/Documents/SMA-1/data/raw/hpo/hp.obo")

OBO_FIXTURE = """\
format-version: 1.2
data-version: tiny/releases/2026-01-01
ontology: tiny

[Term]
id: TST:0000001
name: Root

[Term]
id: TST:0000002
name: Child
is_a: TST:0000001 ! Root
"""


# --------------------------------------------------------------------------- #
# Format inference
# --------------------------------------------------------------------------- #
def test_register_infers_format_from_extension(tmp_path: Path) -> None:
    reg = OntologyRegistry()
    e_obo = reg.register("a", str(tmp_path / "x.obo"))
    e_owl = reg.register("b", str(tmp_path / "x.owl"))
    e_owlxml = reg.register("c", str(tmp_path / "x.owl.xml"))
    assert e_obo.format == "obo"
    assert e_owl.format == "owl"
    assert e_owlxml.format == "owl"


def test_explicit_format_overrides_inference(tmp_path: Path) -> None:
    reg = OntologyRegistry()
    entry = reg.register("a", str(tmp_path / "weird.dat"), fmt="obo")
    assert entry.format == "obo"


def test_infer_format_rejects_unknown_extension() -> None:
    with pytest.raises(ValueError):
        _infer_format("/tmp/whatever.txt")


def test_list_preserves_registration_order(tmp_path: Path) -> None:
    reg = OntologyRegistry()
    reg.register("z", str(tmp_path / "z.obo"))
    reg.register("a", str(tmp_path / "a.obo"))
    reg.register("m", str(tmp_path / "m.owl"))
    assert [e.name for e in reg.list()] == ["z", "a", "m"]
    assert all(isinstance(e, OntologyEntry) for e in reg.list())


def test_get_unknown_name_raises(tmp_path: Path) -> None:
    reg = OntologyRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


# --------------------------------------------------------------------------- #
# get() loads, mounts, and caches
# --------------------------------------------------------------------------- #
def _install_stub_mount(monkeypatch: pytest.MonkeyPatch) -> list:
    """Install a lightweight ``sma.ontology.mount`` stub recording calls.

    The router/registry only rely on the mount contract (a returned object with
    ``.canon`` and ``.build_case``), so a stub lets us assert load+mount+cache
    behaviour without depending on the real (parallel-authored) mount internals.
    """
    calls: list = []

    class _StubMounted:
        def __init__(self, graph, config):
            self.graph = graph
            self.canon = object()

        def build_case(self, term_ids, subject="subject", metadata=None):
            return ("case", tuple(term_ids))

    def _stub_mount(graph, config=None):
        calls.append((graph, config))
        return _StubMounted(graph, config)

    mod = types.ModuleType("sma.ontology.mount")
    mod.mount = _stub_mount  # type: ignore[attr-defined]
    mod.MountedOntology = _StubMounted  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sma.ontology.mount", mod)
    return calls


def test_get_loads_mounts_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_stub_mount(monkeypatch)

    obo = tmp_path / "tiny.obo"
    obo.write_text(OBO_FIXTURE, encoding="utf-8")

    reg = OntologyRegistry()
    reg.register("tiny", str(obo))

    first = reg.get("tiny")
    second = reg.get("tiny")

    # Mount-shaped: has the contract attributes.
    assert hasattr(first, "canon")
    assert hasattr(first, "build_case")
    # Cached: identical object, and mount invoked exactly once.
    assert first is second
    assert len(calls) == 1
    # The graph passed to mount carries the loaded version, propagated to entry.
    assert reg.list()[0].version == "tiny/releases/2026-01-01"


def test_reregister_invalidates_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_stub_mount(monkeypatch)

    obo = tmp_path / "tiny.obo"
    obo.write_text(OBO_FIXTURE, encoding="utf-8")

    reg = OntologyRegistry()
    reg.register("tiny", str(obo))
    first = reg.get("tiny")
    reg.register("tiny", str(obo))  # re-register same name
    second = reg.get("tiny")

    assert first is not second
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def test_route_by_prefix() -> None:
    router = DomainRouter(OntologyRegistry())
    router.register_prefix("HP:", "hpo")
    router.register_prefix("GO:", "go")
    assert router.route(term_ids=["HP:0001250"]) == ["hpo"]
    assert router.route(term_ids=["GO:0005634", "HP:0001250"]) == ["go", "hpo"]


def test_route_by_domain() -> None:
    router = DomainRouter(OntologyRegistry())
    router.register_domain("medicine", "hpo")
    assert router.route(domain="medicine") == ["hpo"]
    assert router.route(domain="unknown") == []


def test_route_longest_prefix_wins() -> None:
    router = DomainRouter(OntologyRegistry())
    router.register_prefix("HP:", "hpo")
    router.register_prefix("HP:00012", "hpo_specific")
    # The longer matching prefix takes precedence.
    assert router.route(term_ids=["HP:0001250"]) == ["hpo_specific"]
    assert router.route(term_ids=["HP:0009999"]) == ["hpo"]


def test_route_dedup_and_order_stable() -> None:
    router = DomainRouter(OntologyRegistry())
    router.register_prefix("HP:", "hpo")
    router.register_prefix("GO:", "go")
    router.register_domain("medicine", "hpo")
    result = router.route(
        term_ids=["HP:0001", "GO:0001", "HP:0002", "GO:0002"],
        domain="medicine",
    )
    # Domain first, then first-seen term ontologies, each once.
    assert result == ["hpo", "go"]


def test_route_nothing_matches_returns_empty() -> None:
    router = DomainRouter(OntologyRegistry())
    router.register_prefix("HP:", "hpo")
    assert router.route(term_ids=["ZZ:0001"]) == []
    assert router.route() == []


# --------------------------------------------------------------------------- #
# Real-HPO end-to-end (skipped when data unavailable)
# --------------------------------------------------------------------------- #
def test_real_hpo_get_is_mount_shaped() -> None:
    if not REAL_HPO.exists():
        pytest.skip("real HPO data not available")
    try:
        import sma.ontology.mount  # noqa: F401
    except Exception:
        pytest.skip("mount.py not available yet")

    reg = OntologyRegistry()
    reg.register("hpo", str(REAL_HPO))
    mounted = reg.get("hpo")
    assert hasattr(mounted, "canon")
    assert hasattr(mounted, "build_case")
    # Cached identity holds for the heavy real ontology too.
    assert reg.get("hpo") is mounted
