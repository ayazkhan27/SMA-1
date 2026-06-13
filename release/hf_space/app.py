"""Hugging Face Space entrypoint — thin shim over the package UI.

The Space build copies the `sma/` package alongside this file (see
scripts/build_release.py). Keep this shim minimal; all UI logic lives in
sma/ui/app.py so the Space and the local `make ui` stay identical.
"""
from sma.ui.app import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
