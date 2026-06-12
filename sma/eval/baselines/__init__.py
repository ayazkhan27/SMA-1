"""Baseline retrieval implementations for the LogHub evaluation ladder.

Modules here are deliberately self-contained: each baseline owns its model
loading and scoring so `scripts/baseline_ladder.py` can compose them under the
exact protocol of `sma.eval.loghub_eval` without touching that file.
"""
