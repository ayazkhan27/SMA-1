"""BugsInPy metadata parsing, patch-structure extraction, and case assembly.

Blueprint T3 (bug-fix memory): a case is the deterministic structure of a
buggy program state -- the unified diff of the fix (which files/functions
were touched, what calls/keywords/exceptions were added or removed, how big
the change is) plus the failing-test context from ``run_test.sh``. Everything
here is Tier-0: regex/diff parsing only, no models.

Layout of the dataset (github.com/soarsmu/BugsInPy, cloned into
``data/raw/bugsinpy``)::

    projects/<project>/bugs/<id>/bug.info        # commit ids, test_file
    projects/<project>/bugs/<id>/bug_patch.txt   # unified diff buggy->fixed
    projects/<project>/bugs/<id>/run_test.sh     # failing test invocation
"""

from __future__ import annotations

import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field

from sma.encoders import get_encoder
from sma.ir.schema import Case, Statement, entity, make_case, stmt

# ---------------------------------------------------------------------------
# Discovery / loading
# ---------------------------------------------------------------------------


def discover_bug_metadata(root: str | pathlib.Path) -> list[pathlib.Path]:
    return sorted(pathlib.Path(root).glob("projects/*/bugs/*/bug.info"))


@dataclass
class BugRecord:
    project: str
    bug_id: str
    patch_text: str
    test_files: tuple[str, ...] = ()
    test_names: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.project}/{self.bug_id}"


def _parse_bug_info(path: pathlib.Path) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r'^(\w+)="(.*)"\s*$', line.strip())
        if m:
            info[m.group(1)] = m.group(2)
    return info


def _parse_run_test(path: pathlib.Path) -> tuple[str, ...]:
    """Extract failing-test function names from run_test.sh (deterministic)."""
    if not path.exists():
        return ()
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # pytest path/to/file.py::Class::test_name  (possibly several)
        for token in line.split():
            if "::" in token:
                names.append(token.split("::")[-1])
        # python -m unittest [-q] pkg.mod.Class.test_name
        if "unittest" in line and "::" not in line:
            tail = line.split()[-1]
            comp = tail.split(".")[-1]
            if comp.startswith("test"):
                names.append(comp)
    out, seen = [], set()
    for n in names:
        n = re.sub(r"\[.*\]$", "", n)  # strip parametrize ids
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def load_bugs(root: str | pathlib.Path) -> list[BugRecord]:
    """Load every bug with a non-empty patch, sorted by (project, bug id)."""
    records: list[BugRecord] = []
    for info_path in discover_bug_metadata(root):
        bug_dir = info_path.parent
        project = bug_dir.parent.parent.name
        bug_id = bug_dir.name
        patch_path = bug_dir / "bug_patch.txt"
        if not patch_path.exists():
            continue
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        if not patch_text.strip():
            continue  # e.g. keras bug 12 ships an empty patch
        info = _parse_bug_info(info_path)
        test_files = tuple(
            t.strip() for t in info.get("test_file", "").split(";") if t.strip()
        )
        records.append(
            BugRecord(
                project=project,
                bug_id=bug_id,
                patch_text=patch_text,
                test_files=test_files,
                test_names=_parse_run_test(bug_dir / "run_test.sh"),
            )
        )
    records.sort(key=lambda r: (r.project, int(r.bug_id) if r.bug_id.isdigit() else 0))
    return records


# ---------------------------------------------------------------------------
# Unified-diff parsing
# ---------------------------------------------------------------------------


@dataclass
class Hunk:
    file: str
    header: str  # text after the second @@ (enclosing-scope context)
    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)


@dataclass
class PatchFacts:
    files: list[str]
    hunks: list[Hunk]
    functions: list[tuple[str, str]]  # (file basename, function) pairs
    added_lines: list[str]  # non-blank code lines added (markers stripped)
    removed_lines: list[str]
    exceptions: list[str]  # CamelCase *Error/*Exception/*Warning mentioned

    @property
    def n_added(self) -> int:
        return len(self.added_lines)

    @property
    def n_removed(self) -> int:
        return len(self.removed_lines)


_EXC_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception|Warning))\b")
_DEF_RE = re.compile(r"\bdef\s+([A-Za-z_]\w*)")


def parse_patch(patch_text: str) -> PatchFacts:
    files: list[str] = []
    hunks: list[Hunk] = []
    current_file = ""
    hunk: Hunk | None = None
    for raw in patch_text.splitlines():
        if raw.startswith("diff --git"):
            m = re.search(r" b/(\S+)$", raw)
            current_file = m.group(1) if m else raw.split()[-1]
            files.append(current_file)
            hunk = None
            continue
        if raw.startswith("+++") or raw.startswith("---") or raw.startswith("index "):
            continue
        if raw.startswith("@@"):
            m = re.match(r"^@@[^@]*@@\s?(.*)$", raw)
            hunk = Hunk(file=current_file, header=m.group(1) if m else "")
            hunks.append(hunk)
            continue
        if hunk is None:
            continue
        if raw.startswith("+"):
            line = raw[1:]
            if line.strip():
                hunk.added.append(line)
        elif raw.startswith("-"):
            line = raw[1:]
            if line.strip():
                hunk.removed.append(line)
        else:
            line = raw[1:] if raw.startswith(" ") else raw
            if line.strip():
                hunk.context.append(line)

    added = [l for h in hunks for l in h.added]
    removed = [l for h in hunks for l in h.removed]
    context = [l for h in hunks for l in h.context]

    functions: list[tuple[str, str]] = []
    seen_fn: set[tuple[str, str]] = set()
    for h in hunks:
        base = pathlib.PurePosixPath(h.file).name
        names = _DEF_RE.findall(h.header)
        # defs appearing inside the changed lines are also "modified functions"
        for l in h.removed + h.added:
            names.extend(_DEF_RE.findall(l))
        if not names:
            names = ["<module>"]
        for name in names:
            key = (base, name)
            if key not in seen_fn:
                seen_fn.add(key)
                functions.append(key)

    exceptions = sorted(
        {e for l in added + context for e in _EXC_RE.findall(l)}
    )
    return PatchFacts(
        files=files,
        hunks=hunks,
        functions=functions,
        added_lines=added,
        removed_lines=removed,
        exceptions=exceptions,
    )


# ---------------------------------------------------------------------------
# Case assembly (Tier-0)
# ---------------------------------------------------------------------------

_PY_KEYWORDS = {
    "if", "elif", "else", "for", "while", "try", "except", "finally", "raise",
    "return", "with", "assert", "not", "and", "or", "in", "is", "def", "class",
    "lambda", "yield", "del", "pass", "import", "from", "as", "global", "print",
}
_CALL_RE = re.compile(r"([A-Za-z_][\w\.]*)\s*\(")
_KEYWORD_TOKENS = (
    "if", "elif", "else", "for", "while", "try", "except", "finally", "raise",
    "return", "with", "assert", "not", "and", "or", "in", "is", "None",
)
_KEYWORD_RES = {k: re.compile(rf"\b{k}\b") for k in _KEYWORD_TOKENS}

_MAX_CALLS = 12
_MAX_EXCEPTIONS = 8
_MAX_FUNCTIONS = 16
_MAX_ENCODER_STMTS = 24


def _calls(lines: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lines:
        for name in _CALL_RE.findall(line):
            tail = name.split(".")[-1]
            if tail in _PY_KEYWORDS or name in _PY_KEYWORDS:
                continue
            counts[tail] += 1
    return counts


def _keyword_counts(lines: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lines:
        code = line.split("#", 1)[0]
        for kw, rx in _KEYWORD_RES.items():
            counts[kw] += len(rx.findall(code))
    return counts


def size_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 7:
        return "4-7"
    if n <= 15:
        return "8-15"
    if n <= 31:
        return "16-31"
    return "32plus"


def _encoder_statements(added_lines: list[str]) -> list[Statement]:
    """Run the existing code adapter on the added hunks (AST if it parses,
    regex fallback otherwise); keep its non-placeholder statements."""
    block = "\n".join(l.strip() for l in added_lines)
    if not block.strip():
        return []
    encoder = get_encoder("code")
    result = encoder.encode(block, language="python")
    functors = {s.functor for s in result.case.statements}
    if "syntaxError" in functors:  # hunk fragments rarely parse; regex fallback
        result = encoder.encode(block, language="text")
    keep = [
        s
        for s in result.case.statements
        if s.functor not in {"syntaxError", "emptyCode", "rawCode"}
    ]
    return keep[:_MAX_ENCODER_STMTS]


def bug_case(record: BugRecord, facts: PatchFacts | None = None) -> Case:
    """Deterministic Tier-0 case for one bug: diff structure + failing test."""
    facts = facts or parse_patch(record.patch_text)
    statements: set[Statement] = set()

    for base, fn in facts.functions[:_MAX_FUNCTIONS]:
        statements.add(stmt("modifies", entity(base, "file"), entity(fn, "function")))
    statements.add(stmt("addsLines", entity(size_bucket(facts.n_added), "count")))
    statements.add(stmt("removesLines", entity(size_bucket(facts.n_removed), "count")))
    statements.add(
        stmt("touchesFiles", entity(size_bucket(len(facts.files)), "count"))
    )
    if any("test" in f.lower() for f in facts.files):
        statements.add(stmt("touchesTestFile", entity("patch", "scope")))

    for exc in facts.exceptions[:_MAX_EXCEPTIONS]:
        statements.add(stmt("mentionsException", entity(exc, "exception")))

    added_calls = _calls(facts.added_lines)
    removed_calls = _calls(facts.removed_lines)
    net_added = sorted(
        (c for c in added_calls if added_calls[c] > removed_calls.get(c, 0)),
        key=lambda c: (-added_calls[c], c),
    )[:_MAX_CALLS]
    net_removed = sorted(
        (c for c in removed_calls if removed_calls[c] > added_calls.get(c, 0)),
        key=lambda c: (-removed_calls[c], c),
    )[:_MAX_CALLS]
    for name in net_added:
        statements.add(stmt("addsCall", entity("patch", "scope"), entity(name, "callable")))
    for name in net_removed:
        statements.add(stmt("removesCall", entity("patch", "scope"), entity(name, "callable")))

    kw_added = _keyword_counts(facts.added_lines)
    kw_removed = _keyword_counts(facts.removed_lines)
    for kw in _KEYWORD_TOKENS:
        if kw_added[kw] > kw_removed[kw]:
            statements.add(stmt("addsKeyword", entity("patch", "scope"), entity(kw, "keyword")))
        elif kw_removed[kw] > kw_added[kw]:
            statements.add(stmt("removesKeyword", entity("patch", "scope"), entity(kw, "keyword")))

    for name in record.test_names[:4]:
        statements.add(stmt("failingTest", entity(name, "test")))
    for tf in record.test_files[:4]:
        statements.add(
            stmt("testModule", entity(pathlib.PurePosixPath(tf).name, "file"))
        )

    statements.update(_encoder_statements(facts.added_lines))

    if not statements:
        statements.add(stmt("emptyPatch", entity(record.key, "bug")))
    return make_case(
        sorted(statements, key=repr),
        {
            "adapter": "code",
            "tier": 0,
            "dataset": "bugsinpy",
            "project": record.project,
            "bug": record.bug_id,
        },
        case_id=f"bugsinpy:{record.key}",
    )
