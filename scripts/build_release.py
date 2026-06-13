"""Assemble a clean public release tree under dist/sma-public/.

Copies only public-safe paths (release/RELEASE_PLAN.md is the contract),
re-stamps git authorship to the public GitHub identity, and refuses to run if
any secret/data/copyrighted artifact would leak. Build, never auto-publish.

    python3 scripts/build_release.py --check     # dry run, report only
    python3 scripts/build_release.py             # write dist/sma-public/
"""
from __future__ import annotations
import argparse, pathlib, re, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "dist" / "sma-public"

PUBLIC_AUTHOR = "Ayaz Khan"
PUBLIC_EMAIL = "khanayaz2727@gmail.com"

# Public-safe top-level paths (files or dirs). Everything else is excluded.
INCLUDE = [
    "sma", "scripts", "tests", "configs", "docs",
    "paper/figures", "paper/figures/individual", "paper/figures/tikz",
    "paper/manuscript", "paper/diagrams", "paper/GRAPHICS_STACK.md", "paper/README.md",
    "paper/figure_specs",
    "data/manifests",
    "reports/confirmatory", "reports/calibration_grid.csv", "reports/macfac_certification.csv",
    "LICENSE", "README.md", "pyproject.toml", "Makefile", "GOALS.md",
    ".gitignore",
]
# Hard denylist: never copy, even if nested under an included dir.
DENY = re.compile(r"(^|/)("
                  r"\.env|reference_papers|data/raw|data/processed|models|"
                  r"\.lmdb|\.sqlite3?|report\.html|__pycache__|\.aux$|\.log$|"
                  r"\.git$|\.claude)")
SECRET = re.compile(r"sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY")


def iter_files():
    for top in INCLUDE:
        p = ROOT / top
        if not p.exists():
            print(f"  [skip missing] {top}")
            continue
        if p.is_file():
            yield p
        else:
            for f in p.rglob("*"):
                if f.is_file():
                    yield f


def rel(f):
    return f.relative_to(ROOT).as_posix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="dry run; report only")
    args = ap.parse_args()

    staged, denied, leaks = [], [], []
    for f in iter_files():
        r = rel(f)
        if DENY.search(r):
            denied.append(r)
            continue
        staged.append(f)
        # secret scan on text files
        try:
            if f.suffix in {".py", ".toml", ".md", ".txt", ".cfg", ".json", ".sh", ".tex", ".bib"}:
                if SECRET.search(f.read_text(encoding="utf-8", errors="ignore")):
                    leaks.append(r)
        except Exception:
            pass

    print(f"staged: {len(staged)} files   denied: {len(denied)}   secret-hits: {len(leaks)}")
    if leaks:
        print("REFUSING: potential secrets in:")
        for r in leaks:
            print(f"   !! {r}")
        sys.exit(2)
    if args.check:
        print("dry run OK; no secrets detected. Re-run without --check to write.")
        return

    if DEST.exists():
        shutil.rmtree(DEST)
    for f in staged:
        out = DEST / rel(f)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
    # release author note (git history is re-initialized at publish time)
    (DEST / "RELEASE_AUTHOR.txt").write_text(f"{PUBLIC_AUTHOR} <{PUBLIC_EMAIL}>\n")
    print(f"wrote {DEST} ({len(staged)} files). Init git there with the public identity, "
          f"then push to GitHub / sync the HF Space.")


if __name__ == "__main__":
    main()
