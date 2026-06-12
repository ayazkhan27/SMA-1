"""Deterministic failure-family labels for LogHub sessions.

The binary Anomaly/Normal label is too coarse to tell whether retrieval
surfaced the *right kind* of failure. This module derives deterministic
root-cause "families" so the eval can score family-hit@k: did the top-k
retrieved sessions share the query's failure family, not merely its
binary class.

HDFS family inventory
---------------------
Derived by inspecting all 2,500 anomalous sessions in
``data/processed/ui_corpus_hdfs.jsonl`` (5,000 sessions, 50/50 split).
Distinct WARN/ERROR/exception line signatures and their session counts
(a session may contain several signatures; the rule order below resolves
co-occurrence by putting rare, specific causes before generic catch-alls):

==============================  =====  ==========================================
signature                       count  family assigned
==============================  =====  ==========================================
BlockInfo not found in            737  DeleteBlockNotFound
  volumeMap (delete error)
writeBlock ... Could not read     494  StreamReadFailure
  from stream (IOException)
Got exception while serving       462  ServeBlockException
Redundant addStoredBlock          138  RedundantAddStoredBlock
Connection reset by peer           11  ConnectionReset
PendingReplicationMonitor           8  ReplicationMonitorTimeout
  timed out
EOFException                        7  EOFException
Interrupt family (Interrupted-      8  Interrupt
  IOException, ClosedByInterrupt-
  Exception, Interrupted receive)
SocketTimeoutException              3  SocketTimeout
Broken pipe                         2  BrokenPipe
No route to host                    1  NoRouteToHost
(no failure line at all)          832  other_anomaly  (sequence-shape anomalies)
==============================  =====  ==========================================

Note: a bare ``java\\.[...](\\w+Exception|\\w+Error)`` regex labels 498 of
these sessions "IOException", which is a useless catch-all (it covers
ConnectionReset, BrokenPipe, StreamReadFailure, ...). The rules below
therefore split IOException by its message and only fall back to the raw
exception class for classes not already covered.

BGL families come from the alert-category column of the raw BGL.log
(first whitespace token, e.g. KERNDTLB, APPSEV). That column is stripped
from session *text* by the sampler (label-leak fix), so ground-truth
families must be read from the raw log, keyed by the sampler's
``bgl_<node>_<window>`` scheme.
"""

from __future__ import annotations

import pathlib
import re
import zipfile
from collections import Counter, defaultdict

# Generic Java exception/error class fallback, e.g. "EOFException".
_JAVA_EXC_RE = re.compile(r"java\.[a-zA-Z.]*\b(\w+Exception|\w+Error)")

# Ordered (family, predicate-substring(s)) rules. First match wins.
# Rare, specific root causes come first; broad catch-alls last, so a
# session showing "Connection reset by peer" inside an IOException is
# labelled ConnectionReset rather than the generic class name.
_HDFS_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ConnectionReset", ("Connection reset",)),
    ("BrokenPipe", ("Broken pipe",)),
    ("NoRouteToHost", ("No route to host", "NoRouteToHostException")),
    ("ReplicationMonitorTimeout", ("PendingReplicationMonitor timed out",)),
    ("SocketTimeout", ("SocketTimeoutException", "millis timeout")),
    ("EOFException", ("EOFException",)),
    (
        "Interrupt",
        (
            "InterruptedIOException",
            "ClosedByInterruptException",
            "Interrupted receiveBlock",
            "interrupt",
        ),
    ),
    ("Checksum", ("checksum", "Checksum")),
    ("StreamReadFailure", ("Could not read from stream",)),
    ("DeleteBlockNotFound", ("BlockInfo not found in volumeMap",)),
    ("ServeBlockException", ("Got exception while serving",)),
    ("RedundantAddStoredBlock", ("Redundant addStoredBlock",)),
)


def hdfs_family(session_text: str, label: str = "Anomaly") -> str:
    """Deterministic failure family for an HDFS block session.

    Rules (ordered, first match wins):
      1. Specific failure markers / refined exception messages from the
         table in the module docstring.
      2. Any remaining ``java.*Exception|Error`` class name (regex), for
         classes the explicit rules do not cover.
      3. ``other_anomaly`` for anomalous sessions with no failure text
         (HDFS labels many sessions anomalous purely on event-sequence
         shape; they contain only INFO lines).

    ``label`` is the binary ground-truth label; normal sessions always
    return ``"normal"`` regardless of text (a handful of normal sessions
    contain benign warning lines, and family metrics only score
    anomalies).
    """
    if label == "Normal":
        return "normal"
    for family, needles in _HDFS_RULES:
        for needle in needles:
            if needle in session_text:
                return family
    m = _JAVA_EXC_RE.search(session_text)
    if m:
        return m.group(1)
    return "other_anomaly"


def bgl_family(
    zip_path: pathlib.Path | str, session_keys: set[str] | list[str]
) -> dict[str, str]:
    """Ground-truth families for BGL sessions from the alert-category column.

    Streams the raw ``BGL.log`` inside ``zip_path`` and, for every line
    belonging to one of ``session_keys``, collects the alert-category
    column (first whitespace token; ``-`` means non-alert). The session's
    family is the most frequent non-``-`` category in its window, ties
    broken alphabetically; sessions whose lines are all ``-`` get
    ``"normal"``.

    Keys use the exact scheme of ``sample_bgl_stratified`` in
    ``sma.eval.loghub_eval``: ``bgl_<node>_<window>`` with
    ``window = unix_timestamp // 60`` and the same line-parsing filter
    (``split(maxsplit=5)``, >= 5 fields, integer timestamp).

    Returns a dict mapping every key in ``session_keys`` to its family
    (keys with no matching log lines are omitted, mirroring the sampler,
    which also drops empty sessions).
    """
    wanted = set(session_keys)
    cat_counts: dict[str, Counter] = defaultdict(Counter)
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("BGL.log") as fh:
            for line_bytes in fh:
                line = line_bytes.decode("utf-8", errors="ignore")
                parts = line.split(maxsplit=5)
                if len(parts) < 5:
                    continue
                try:
                    timestamp = int(parts[1])
                except ValueError:
                    continue
                node_id = parts[3]
                window = timestamp // 60
                session_key = f"bgl_{node_id}_{window}"
                if session_key in wanted:
                    cat_counts[session_key][parts[0]] += 1

    families: dict[str, str] = {}
    for key, counts in cat_counts.items():
        alerts = {c: n for c, n in counts.items() if c != "-"}
        if not alerts:
            families[key] = "normal"
        else:
            # Most frequent alert category; alphabetical tie-break.
            families[key] = min(alerts, key=lambda c: (-alerts[c], c))
    return families
