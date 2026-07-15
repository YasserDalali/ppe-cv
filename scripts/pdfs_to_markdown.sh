#!/usr/bin/env bash
# Convert content/*.pdf → content/md/*.md for token-efficient AI reading.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/content"
OUT="$ROOT/content/md"
mkdir -p "$OUT"

if ! command -v pdftotext >/dev/null 2>&1; then
  echo "error: pdftotext not found (install poppler-utils)" >&2
  exit 1
fi

# Stable, readable slugs for known papers (others keep basename).
slug_for() {
  local base="${1%.pdf}"
  case "$base" in
    661251) echo "tongue-diagnosis-mobilenet" ;;
    661252) echo "eco-track-trash-classification" ;;
    661256) echo "traffic-flow-management" ;;
    661260) echo "tuber-ensemble-xray" ;;
    661266) echo "ia-med-disease-prediction" ;;
    662647) echo "ct-semantic-segmentation" ;;
    "Drone (16)") echo "drone-aerial-accident-detection" ;;
    PPE_Systeme_FR) echo "ppe-systeme-pitch-fr" ;;
    *) echo "$base" | tr ' ()' '---' | tr -s '-' | tr '[:upper:]' '[:lower:]' ;;
  esac
}

shopt -s nullglob
pdfs=("$SRC"/*.pdf)
if ((${#pdfs[@]} == 0)); then
  echo "No PDFs in $SRC" >&2
  exit 1
fi

for pdf in "${pdfs[@]}"; do
  base="$(basename "$pdf")"
  slug="$(slug_for "$base")"
  md="$OUT/${slug}.md"
  pages="$(pdfinfo "$pdf" 2>/dev/null | awk -F: '/^Pages/ {gsub(/ /,"",$2); print $2}')"
  pages="${pages:-?}"

  {
    echo "---"
    echo "source_pdf: ${base}"
    echo "pages: ${pages}"
    echo "converted_with: pdftotext -layout"
    echo "---"
    echo
    echo "# ${base}"
    echo
    echo "> Auto-converted from PDF. Prefer this file over the PDF for AI reading."
    echo
    echo "---"
    echo
    # Form-feed page breaks → markdown horizontal rules
    pdftotext -layout "$pdf" - | sed 's/\x0c/\n\n---\n\n/g'
  } > "$md"

  bytes="$(wc -c < "$md" | tr -d ' ')"
  echo "OK  $base → md/${slug}.md  (${pages} pages, ${bytes} bytes)"
done

echo
echo "Wrote ${#pdfs[@]} markdown file(s) to $OUT"
