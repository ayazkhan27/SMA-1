# ADR-001: Stack and MVP Scope

## Context

The blueprint specifies a CPU-only Python 3.11 artifact with optional heavy dependencies for
encoders, baselines, and UI. The local workspace initially contained only the blueprint file and no
git repository.

## Decision

Use a standard-library-friendly implementation path and optional runtime imports for heavy
libraries. The blueprint target remains Python 3.11; the local laptop currently provides Python
3.10, so package metadata accepts Python 3.10+ while CI should still prefer 3.11. The tracked MVP
includes deterministic IR, store, matcher, retrieval, SAGE, FastAPI, Gradio, dataset manifests,
and report generation. Logs and code/bugs are mandatory MVP domains. ARN remains a flagged Tier-1
benchmark.

## Consequences

Gate tests can run without downloading external datasets. Full paper runs require the acquisition
scripts and optional extras.
