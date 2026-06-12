"""Published-style greedy kernel merge fallback."""

from __future__ import annotations

from .conflicts import kernels_conflict
from .types import Kernel


def greedy_merge(kernels: tuple[Kernel, ...]) -> tuple[Kernel, ...]:
    selected: list[Kernel] = []
    for kernel in sorted(kernels, key=lambda k: (-k.weight, k.root.base_key, k.root.target_key)):
        if all(not kernels_conflict(kernel, existing) for existing in selected):
            selected.append(kernel)
    return tuple(selected)

