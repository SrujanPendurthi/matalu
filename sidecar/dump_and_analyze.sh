#!/usr/bin/env bash
# One-command dictation-accuracy diagnostic.
#
# Launches the app with the sidecar's audio dump enabled, waits while you do ONE
# bad-accuracy dictation, then (when you quit the app) analyzes exactly what the
# model heard — level/noise stats + an offline transcription — to tell a
# mic/capture problem from a streaming one.
#
# Usage:  sidecar/dump_and_analyze.sh
#   1. The app launches. Dictate a sentence you KNOW (e.g. read this line aloud).
#   2. Quit the app (tray → Quit, or Ctrl-C in this terminal).
#   3. Read the analysis printed here.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
wav="${1:-$root/matalu_heard.wav}"

cd "$root"
rm -f "$wav"

echo "════════════════════════════════════════════════════════════════"
echo " Dumping model-input audio to: $wav"
echo " → Dictate a sentence you know, then QUIT the app (tray Quit / Ctrl-C)."
echo "════════════════════════════════════════════════════════════════"

# Run the app with the dump enabled. `|| true` so a Ctrl-C / non-zero app exit
# still falls through to analysis (set -e would otherwise abort here).
MATALU_DUMP_WAV="$wav" cargo tauri dev || true

echo
echo "════════════════════════════════════════════════════════════════"
echo " App exited — analyzing $wav"
echo "════════════════════════════════════════════════════════════════"
if [ ! -s "$wav" ]; then
  echo "No audio captured. The gate only feeds the model while you're dictating —"
  echo "make sure you held/toggled the hotkey and spoke before quitting."
  exit 1
fi
python3 "$here/analyze_dump.py" "$wav"
