"""Symbolic canonicalization and minimal ascension support."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .schema import safe_symbol


@dataclass
class PredicateLattice:
    parents: dict[str, set[str]] = field(default_factory=dict)

    def add(self, child: str, parent: str) -> None:
        self.parents.setdefault(safe_symbol(child), set()).add(safe_symbol(parent))

    def ancestors(self, symbol: str, max_depth: int = 2) -> dict[str, int]:
        symbol = safe_symbol(symbol)
        out = {symbol: 0}
        queue = deque([(symbol, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for parent in self.parents.get(current, ()):
                if parent not in out or depth + 1 < out[parent]:
                    out[parent] = depth + 1
                    queue.append((parent, depth + 1))
        return out

    def minimal_ascension(self, left: str, right: str, delta: int) -> tuple[str, int] | None:
        left_anc = self.ancestors(left, delta)
        right_anc = self.ancestors(right, delta)
        overlap = set(left_anc).intersection(right_anc)
        if not overlap:
            return None
        best = min(overlap, key=lambda s: (left_anc[s] + right_anc[s], s))
        dist = left_anc[best] + right_anc[best]
        if dist <= delta:
            return best, dist
        return None


@dataclass
class Canonicalizer:
    aliases: dict[str, str] = field(default_factory=dict)
    lattice: PredicateLattice = field(default_factory=PredicateLattice)

    def canonical(self, symbol: str) -> str:
        symbol = safe_symbol(symbol)
        for prefix in ("far_", "near_"):
            if symbol.startswith(prefix):
                symbol = symbol.removeprefix(prefix)
        seen: set[str] = set()
        while symbol in self.aliases and symbol not in seen:
            seen.add(symbol)
            symbol = safe_symbol(self.aliases[symbol])
        return symbol

    def compatible(
        self, left: str, right: str, delta: int = 0, rho: float = 1.0
    ) -> tuple[bool, float, str | None, int]:
        left_c = self.canonical(left)
        right_c = self.canonical(right)
        if left_c == right_c:
            return True, 1.0, left_c, 0
        if delta <= 0:
            return False, 0.0, None, 0
        asc = self.lattice.minimal_ascension(left_c, right_c, delta)
        if asc is None:
            return False, 0.0, None, 0
        ancestor, dist = asc
        return True, rho**dist, ancestor, dist


def default_canonicalizer() -> Canonicalizer:
    canon = Canonicalizer(
        aliases={
            "connTimeout": "timeout",
            "connectionTimeout": "timeout",
            "retryStorm": "retry",
            "blockReceiveError": "ioError",
            "pressure": "intensity",
            "temperature": "intensity",
            "waterFlow": "flow",
            "heatFlow": "flow",
            "sun": "centralBody",
            "nucleus": "centralBody",
            "planet": "orbitingBody",
            "electron": "orbitingBody",
            "attractsGravity": "attracts",
            "attractsElectrostatic": "attracts",
        }
    )
    for child, parent in (
        ("timeout", "failureEvent"),
        ("ioError", "failureEvent"),
        ("exception", "failureEvent"),
        ("retry", "recoveryAction"),
        ("restart", "recoveryAction"),
        ("saturation", "resourcePressure"),
    ):
        canon.lattice.add(child, parent)
    return canon
