#!/usr/bin/env bash
# Render helper: SVG -> PDF + PNG via cairosvg (inkscape unavailable in env).
set -e
f="$1"          # path without extension, e.g. .../fig1_overview_v2
w="${2:-2400}"  # png width
cairosvg "$f.svg" -o "$f.pdf"
cairosvg "$f.svg" -o "$f.png" --output-width "$w"
echo "rendered: $f.pdf  $f.png (w=$w)"
