"""Test collection guard.

The ``tests/agentic`` and ``tests/agentic_qa`` suites exercise the evaluation
harness, which depends on the optional ``eval`` extra (networkx for the HippoRAG
baseline, rank-bm25, sentence-transformers) and on external benchmark data that is
not tracked in the repository. In a base-install environment (such as the CI
``gates`` job, which installs only the core package) those modules cannot even be
imported, which would otherwise turn the whole collection red. We skip collecting
them when their dependencies are absent so the core gate suite still runs; a full
install (``pip install -e ".[eval,encoders]"``) collects and runs everything.
"""
import importlib.util

_EVAL_DEPS = ("networkx", "rank_bm25", "sentence_transformers")

collect_ignore_glob = []
if any(importlib.util.find_spec(_m) is None for _m in _EVAL_DEPS):
    collect_ignore_glob += ["agentic/*", "agentic/**", "agentic_qa/*", "agentic_qa/**"]
