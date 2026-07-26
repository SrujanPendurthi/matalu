#!/usr/bin/env bash
# Download the two ONNX models diarize.py needs into sidecar/models/diarization/.
# Idempotent: skips files already present. Models come from the sherpa-onnx
# GitHub releases (ungated, no token). ~34 MB total.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models/diarization"
SEG_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
# English speaker-embedding model (TitaNet-small). Override by dropping a
# different *.onnx from the speaker-recongition-models release as embedding.onnx.
EMB_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx"

mkdir -p "$DIR"
cd "$DIR"

if [[ -f segmentation.onnx ]]; then
  echo "segmentation.onnx already present — skipping"
else
  echo "downloading segmentation model…"
  curl -fL --progress-bar "$SEG_URL" -o seg.tar.bz2
  tar xjf seg.tar.bz2
  mv sherpa-onnx-pyannote-segmentation-3-0/model.onnx segmentation.onnx
  rm -rf seg.tar.bz2 sherpa-onnx-pyannote-segmentation-3-0
fi

if [[ -f embedding.onnx ]]; then
  echo "embedding.onnx already present — skipping"
else
  echo "downloading embedding model…"
  curl -fL --progress-bar "$EMB_URL" -o embedding.onnx
fi

echo "done:"
ls -lh "$DIR"/segmentation.onnx "$DIR"/embedding.onnx
