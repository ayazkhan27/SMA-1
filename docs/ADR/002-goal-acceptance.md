# ADR-002: Goal Acceptance

## Context

The user requested implementation of the SMA-1 MVP from
`structure_mapping_agentic_memory_blueprint.md`, including exact dataset/material definitions,
paper artifacts, and a Gradio MVP UI.

## Decision

Adopt `GOALS.md` as the repository termination contract and implement the blueprint gate by gate.

## Consequences

Progress is evaluated through pytest gate markers and `make report`; negative scientific outcomes
are valid if reported according to the blueprint kill criteria.

