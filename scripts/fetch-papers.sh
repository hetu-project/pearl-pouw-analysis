#!/usr/bin/env bash
# Download the papers referenced by the notes and extract text (requires pdftotext from poppler).
# Papers are not redistributed with this repository.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/papers" && cd "$ROOT/papers"
fetch() { curl -sSL -o "$1" "$2"; pdftotext -layout "$1" "${1%.pdf}.txt"; echo "fetched $1"; }
fetch cupow_2504.09971.pdf         https://arxiv.org/pdf/2504.09971
fetch pearl_whitepaper_fp8.pdf     https://pearlresearch.ai/Pearl_Whitepaper.pdf
fetch usefulness_gap_2606.04819.pdf https://arxiv.org/pdf/2606.04819
fetch economics_2606.06700.pdf     https://arxiv.org/pdf/2606.06700
