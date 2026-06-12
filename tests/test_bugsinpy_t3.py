"""Unit tests for the T3 BugsInPy adapter (diff parsing, case assembly,
fix-pattern categories). Synthetic diffs only -- no dataset required."""

from __future__ import annotations

from sma.eval.bugsinpy import BugRecord, bug_case, parse_patch, size_bucket
from sma.eval.bugsinpy_families import CATEGORIES, categorize

NULL_CHECK_DIFF = """\
diff --git a/pkg/core.py b/pkg/core.py
index 111..222 100644
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -10,4 +10,6 @@ def resolve(item):
     value = lookup(item)
+    if value is None:
+        return default_value()
     return transform(value)
"""

EXC_DIFF = """\
diff --git a/pkg/io.py b/pkg/io.py
index 111..222 100644
--- a/pkg/io.py
+++ b/pkg/io.py
@@ -5,3 +5,6 @@ def read(path):
-    data = parse(path)
+    try:
+        data = parse(path)
+    except ValueError:
+        raise ConfigError(path)
"""

BOUNDARY_DIFF = """\
diff --git a/pkg/win.py b/pkg/win.py
index 111..222 100644
--- a/pkg/win.py
+++ b/pkg/win.py
@@ -7,3 +7,3 @@ def clamp(i, n):
-    if i < n:
+    if i <= n:
         return i
"""

API_SUB_DIFF = """\
diff --git a/pkg/api.py b/pkg/api.py
index 111..222 100644
--- a/pkg/api.py
+++ b/pkg/api.py
@@ -3,2 +3,2 @@ def fetch(url):
-    return urlopen(url)
+    return requests_get(url)
"""


def test_parse_patch_structure():
    facts = parse_patch(NULL_CHECK_DIFF)
    assert facts.files == ["pkg/core.py"]
    assert ("core.py", "resolve") in facts.functions
    assert facts.n_added == 2 and facts.n_removed == 0


def test_categories_ordered_rules():
    assert categorize(parse_patch(NULL_CHECK_DIFF)) == "add-null-check"
    assert categorize(parse_patch(EXC_DIFF)) == "exception-handling"
    assert categorize(parse_patch(BOUNDARY_DIFF)) == "boundary"
    assert categorize(parse_patch(API_SUB_DIFF)) == "api-substitution"
    for diff in (NULL_CHECK_DIFF, EXC_DIFF, BOUNDARY_DIFF, API_SUB_DIFF):
        assert categorize(parse_patch(diff)) in CATEGORIES


def test_bug_case_deterministic_and_structured():
    record = BugRecord(
        project="demo",
        bug_id="1",
        patch_text=NULL_CHECK_DIFF,
        test_files=("tests/test_core.py",),
        test_names=("test_resolve_none",),
    )
    case_a = bug_case(record)
    case_b = bug_case(record)
    assert case_a.case_id == "bugsinpy:demo/1"
    assert case_a.statements == case_b.statements  # deterministic
    functors = {s.functor for s in case_a.statements}
    assert "modifies" in functors
    assert "failingTest" in functors
    assert "addsLines" in functors
    assert case_a.metadata["tier"] == 0


def test_size_bucket_monotone():
    assert size_bucket(0) == "0"
    assert size_bucket(1) == "1"
    assert size_bucket(3) == "2-3"
    assert size_bucket(100) == "32plus"
