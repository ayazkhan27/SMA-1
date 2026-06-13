# ADR-007: Dynamic-adapter governance (production access control)

Status: Accepted (2026-06-12) · Owner: maintainer · Applies to: `sma.agent.adapter_draft`, `sma.encoders.draft_adapter`, `sma.encoders.coverage`

## Context

The dynamic-adapter loop lets an LLM **propose** new deterministic encoding
rules when a query/corpus shows low structural coverage (blueprint §4.1, 12-R3;
coverage tripwire `COVERAGE_WARN_THRESHOLD = 0.4`). The rules are data, not
facts; encoding stays byte-deterministic; drafted rules are content-addressed
and carry an "LLM-proposed, unreviewed" taint. This is powerful — the memory can
extend itself to new domains — and therefore dangerous if anyone can mutate the
encoding contract. In production, an unreviewed rule change silently alters how
every future case is encoded and retrieved.

## Decision (maintainer requirement)

**In production, adapters are FROZEN by default.** Requesting the LLM to draft a
new adapter rule, and giving the final sign-off to promote it, are **admin-only**
operations. Non-admin users get the frozen adapters and may *see* a low-coverage
warning, but cannot trigger drafting or approve rules.

Concretely:
1. **RBAC gate.** Two capabilities — `adapter:draft` (request a proposal) and
   `adapter:approve` (promote to active) — are admin-only. Default user role has
   neither; it gets read-only frozen encoding.
2. **Frozen-by-default.** The active adapter set is immutable for normal
   operation. The base ontology (`ontology-v1`) is never editable at all; drafts
   are *additive residual rules* only (they may not override frozen keywords —
   already enforced by frozen-keyword dedup).
3. **Quarantine namespace.** A drafted rule is applied only in a quarantine
   adapter version until approved; it never silently merges into the active set.
4. **Human sign-off required.** Promotion `proposed → active` always requires an
   explicit `adapter:approve` action by an admin. The coverage tripwire may
   *suggest* drafting; it must never *auto-apply* a rule.
5. **Audit trail.** Every draft records {proposer, corpus hash, prompt, model,
   content hash, timestamp}; every approval records {approver, timestamp,
   diff}. The "unreviewed" taint propagates to any retrieval result that used an
   unapproved rule, so downstream consumers see the provenance.

## Maintainer-additional suggestions (agreed, to implement in Phase 5/6 hardening)

- **Separation of duties:** prefer proposer ≠ approver where org size allows; at
  minimum log when they are the same identity.
- **Versioned + reversible:** every adapter version is content-addressed and
  rollback-able to the immutable `ontology-v1` baseline; a bad rule is one
  revert away, never a data-loss event.
- **Determinism guard at promotion:** CI re-runs the byte-determinism golden
  test (G3) on the candidate adapter before it can be approved; a rule that
  introduces any non-determinism is rejected automatically.
- **Cost / rate limits:** drafting calls (the only LLM use in the encode path's
  vicinity) are rate-limited and budgeted per admin; logged for cost audit.
- **Coverage-improvement gate:** a drafted rule is only *eligible* for approval
  if it measurably raises structural coverage on a held-out slice without
  redundantly re-covering existing rules (extends the existing dedup check into
  an effectiveness check).
- **Scoped blast radius:** approval can be scoped to a single adapter/domain/
  tenant, not global, so a finance rule never affects the logs encoder.
- **Tamper-evident log:** the audit trail is append-only and hash-chained
  (consistent with the WAL/content-addressing discipline already in the store).
- **Kill switch:** a single config flag (`adapters.frozen = true`) hard-disables
  all drafting org-wide for incident response, independent of RBAC.

## Consequences

- Production posture is safe-by-default: the system behaves as a frozen,
  reproducible encoder unless an admin deliberately, auditable-ly extends it.
- The dynamic-adapter capability becomes a *governed* feature (the selling point
  for regulated domains: finance, healthcare, legal) rather than a liability.
- Implementation work lands in Phase 5/6 (hardening + release); the research
  evaluation of *whether* drafting helps (Phase 4b) is separate from this
  *who-may-do-it* control and does not require it.
