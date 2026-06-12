"""Deterministic fix-pattern categories for BugsInPy patches.

Categories are assigned by ORDERED rules over the unified diff (first match
wins). They are the ground-truth labels for the T3 retrieval metric
(fix-category-hit@k): did retrieval surface a past bug fixed by the SAME
KIND of change, not just a textually similar file?

Rule order (per the T3 specification):
  1. add-null-check          adds ``is None`` / ``is not None`` / ``if not x``
  2. exception-handling      adds ``try:`` / ``except`` / ``raise``
  3. boundary                comparison-operator swap or +/-1 on an otherwise
                             identical line (off-by-one / boundary fixes)
  4. type-coercion           adds a builtin cast (int()/str()/list()/...)
  5. api-substitution        small hunk replaces one call with another
  6. default-arg-change      a ``def`` signature line changes its defaults/args
  7. condition-strengthening an if/elif/while gains and/or clauses
  8. other

"Net added" means the pattern occurs more often in added lines than in
removed lines, so pure code motion does not trigger a rule.
"""

from __future__ import annotations

import re

from .bugsinpy import Hunk, PatchFacts, _calls

CATEGORIES = (
    "add-null-check",
    "exception-handling",
    "boundary",
    "type-coercion",
    "api-substitution",
    "default-arg-change",
    "condition-strengthening",
    "other",
)

_NULL_CHECK = re.compile(r"\bis\s+(?:not\s+)?None\b|^\s*(?:el)?if\s+not\s+\w")
_EXC_HANDLING = re.compile(r"^\s*(?:try\s*:|except[\s:(]|raise\b)")
_CAST = re.compile(
    r"(?<![\w.])(?:int|str|float|bool|list|tuple|set|dict|frozenset|bytes)\(|\.astype\("
)
_CMP_OPS = re.compile(r"<=|>=|==|!=|<|>")
_PM_ONE = re.compile(r"[+-]\s*1\b")
_COND_LINE = re.compile(r"^\s*((?:el)?if|while)\b")
_BOOL_OP = re.compile(r"\b(?:and|or)\b")
_DEF_LINE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\((.*)$")


def _strip_code(line: str) -> str:
    return line.split("#", 1)[0]


def _net(pattern: re.Pattern[str], added: list[str], removed: list[str]) -> bool:
    n_add = sum(len(pattern.findall(_strip_code(l))) for l in added)
    n_rem = sum(len(pattern.findall(_strip_code(l))) for l in removed)
    return n_add > n_rem


def _paired_swap(hunk: Hunk, pattern: re.Pattern[str]) -> bool:
    """True if some removed/added line pair is identical once `pattern`
    occurrences are masked out, while the raw lines differ (i.e. the ONLY
    change is in the matched operator/constant)."""
    def norm(line: str) -> str:
        return re.sub(r"\s+", " ", pattern.sub("\x00", _strip_code(line))).strip()

    removed = {norm(l): _strip_code(l).strip() for l in hunk.removed if pattern.search(_strip_code(l))}
    for line in hunk.added:
        code = _strip_code(line)
        if not pattern.search(code):
            continue
        key = norm(line)
        if key in removed and removed[key] != code.strip() and "\x00" in key:
            return True
    return False


def _is_null_check(facts: PatchFacts) -> bool:
    return _net(_NULL_CHECK, facts.added_lines, facts.removed_lines)


def _is_exception_handling(facts: PatchFacts) -> bool:
    return _net(_EXC_HANDLING, facts.added_lines, facts.removed_lines)


def _is_boundary(facts: PatchFacts) -> bool:
    for hunk in facts.hunks:
        if _paired_swap(hunk, _CMP_OPS) or _paired_swap(hunk, _PM_ONE):
            return True
    return False


def _is_type_coercion(facts: PatchFacts) -> bool:
    return _net(_CAST, facts.added_lines, facts.removed_lines)


def _is_api_substitution(facts: PatchFacts) -> bool:
    for hunk in facts.hunks:
        if not (1 <= len(hunk.removed) <= 3 and 1 <= len(hunk.added) <= 3):
            continue
        removed_calls = set(_calls(hunk.removed))
        added_calls = set(_calls(hunk.added))
        if (removed_calls - added_calls) and (added_calls - removed_calls):
            return True
    return False


def _is_default_arg_change(facts: PatchFacts) -> bool:
    for hunk in facts.hunks:
        removed_defs = {}
        for line in hunk.removed:
            m = _DEF_LINE.match(_strip_code(line))
            if m:
                removed_defs[m.group(1)] = m.group(2).strip()
        for line in hunk.added:
            m = _DEF_LINE.match(_strip_code(line))
            if not m:
                continue
            name, args = m.group(1), m.group(2).strip()
            old = removed_defs.get(name)
            if old is not None and old != args and ("=" in old or "=" in args):
                return True
    return False


def _is_condition_strengthening(facts: PatchFacts) -> bool:
    for hunk in facts.hunks:
        removed_conds: dict[str, int] = {}
        for line in hunk.removed:
            m = _COND_LINE.match(_strip_code(line))
            if m:
                kw = m.group(1)
                n = len(_BOOL_OP.findall(_strip_code(line)))
                removed_conds[kw] = max(removed_conds.get(kw, -1), n)
        for line in hunk.added:
            m = _COND_LINE.match(_strip_code(line))
            if not m:
                continue
            kw = m.group(1)
            n = len(_BOOL_OP.findall(_strip_code(line)))
            if kw in removed_conds and n > removed_conds[kw]:
                return True
    return False


_RULES = (
    ("add-null-check", _is_null_check),
    ("exception-handling", _is_exception_handling),
    ("boundary", _is_boundary),
    ("type-coercion", _is_type_coercion),
    ("api-substitution", _is_api_substitution),
    ("default-arg-change", _is_default_arg_change),
    ("condition-strengthening", _is_condition_strengthening),
)


def categorize(facts: PatchFacts) -> str:
    """Ordered first-match-wins fix-pattern category for one patch."""
    for name, rule in _RULES:
        if rule(facts):
            return name
    return "other"
