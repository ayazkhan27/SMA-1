# SMA-1 release plan (Phase 6)

Scaffolding for the public artifact. **Nothing here is published until the
paper is done and the maintainer approves.** The build is reproducible via
`scripts/build_release.py`, which assembles a clean tree under `dist/sma-public/`
containing only public-safe files.

## Identity
- GitHub / public commits: **Ayaz Khan** `<khanayaz2727@gmail.com>` (the
  private working repo uses a different committer; the release build re-stamps
  authorship to this identity — see `build_release.py`).
- License: **Apache-2.0** (`LICENSE` at repo root; declared in `pyproject.toml`).

## What ships vs what NEVER ships

| Ships (public) | NEVER ships |
|---|---|
| `sma/` package (IR, match, index, encoders, sage, agent, ui) | `.env` (secrets / DeepSeek key) |
| `scripts/` (eval, figures, calibration, release) | `data/raw/`, `data/processed/` (LogHub/BugsInPy — license + size) |
| `tests/`, `configs/`, `docs/` (blueprint, ADRs, STATUS, prereg) | `paper/reference_papers/` (copyrighted PDFs) |
| `paper/figures/`, `paper/manuscript/` (our figures + LaTeX) | `models/` (downloaded weights) |
| `LICENSE`, `README.md`, `pyproject.toml`, `Makefile` | `reports/*.html`, local LMDB/sqlite, caches |
| `data/manifests/` (checksums + fetch instructions, NOT data) | private transcripts, `.claude/` |

Data is distributed by **checksum manifest + fetch script**, never by bundling
the raw corpora (`make datasets` reconstructs them from the original sources).

## Distribution targets
1. **GitHub** (`ayazkhan27/sma-1` or similar): the canonical source, Apache-2.0,
   CI, reproducibility (`make gates`, `make paper`). Primary home for code.
2. **Hugging Face Space** (Gradio): the live demo. The `sma/ui/app.py` toggle
   UI runs as a Space; `release/hf_space/` holds the Space `README.md`
   (with the HF YAML header) + a thin `app.py` shim importing the package.
   Needs a CPU Space; no GPU. Secrets (if any LLM backend) via Space secrets,
   never committed.
3. **Hugging Face Dataset** (optional): the SSB generator + a frozen sample of
   generated triples as a reproducible analogical-retrieval benchmark, with a
   dataset card. (The de-circularized SSB is itself a contribution.)
4. **arXiv**: preprint (cs.AI, cross-list cs.IR/cs.LG), `[preprint]` template.
5. **Zenodo**: archival DOI for the artifact (cite in the paper).
6. **Docker** (`docker/`): CPU-only image; `docker run sma:paper make all`
   reproduces every table/figure (blueprint G8).

## Pre-publication checklist (run before any push)
- [ ] `scripts/build_release.py --check` reports zero private files staged
- [ ] No secrets: `git -C dist/sma-public grep -rE "sk-[A-Za-z0-9]{20,}"` empty
- [ ] All DOIs in `references.bib` resolve; `\nocite{*}` removed
- [ ] `make gates` green on the clean tree
- [ ] LICENSE present; headers consistent
- [ ] README has install + reproduce + cite; HF Space README has YAML header
- [ ] Rotate the DeepSeek key (it lived in `.env`; treat as compromised on publish)
- [ ] Confirmatory CSVs + prereg tag included so results are reproducible
- [ ] Figures regenerate from CSVs (no hand-typed numbers)

## Status
Scaffolding only. Build + publish are the final step of Phase 6, after the
manuscript is written and the maintainer signs off.

## Production governance
Dynamic-adapter access control (admin-only drafting + sign-off, frozen by default) is specified in `docs/ADR/007-dynamic-adapter-governance.md`; implement during Phase 5/6 hardening.
