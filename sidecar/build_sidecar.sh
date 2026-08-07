#!/usr/bin/env bash
# Build both self-contained PyInstaller sidecars and stage them where Tauri's
# `externalBin` expects them (src-tauri/binaries/<name>-<triple>).
#
# Run before `cargo tauri build`. Requires pyinstaller plus both sidecars'
# Python deps in the active environment:
#     python3 -m pip install pyinstaller parakeet-mlx mlx-lm
#
# This replaces the committed placeholder stubs with the real binaries. Do NOT
# commit the results — they are large (each bundles MLX) and machine/arch specific.
#
# Both are required. The app degrades differently for each: without the ASR
# sidecar it cannot transcribe at all (loud), without the cleaner it transcribes
# and silently pastes raw text (quiet). The quiet one is why this script builds
# both rather than leaving the cleaner optional.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
triple="aarch64-apple-darwin"   # Apple Silicon; matalu is macOS-only by construction

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "error: pyinstaller not found — python3 -m pip install pyinstaller" >&2
  exit 1
fi

cd "$here"
mkdir -p "$root/src-tauri/binaries"

for name in matalu-sidecar matalu-cleaner; do
  echo "=== building $name ==="
  pyinstaller --clean --noconfirm "$name.spec"
  dest="$root/src-tauri/binaries/$name-$triple"
  cp "dist/$name" "$dest"
  chmod +x "$dest"
  echo "staged -> $dest"
done

# Cheap check that the cleaner actually carries the adapter: a bundle that runs
# the base model looks fine except for one log line, so catch it here instead.
echo "=== verifying the cleaner bundle loads the adapter ==="
if printf '%s\n' '{"text":"so um i think this is a test"}' \
   | "$root/src-tauri/binaries/matalu-cleaner-$triple" 2>&1 >/dev/null \
   | grep -q "short (tuned)"; then
  echo "ok: cleaner reports 'system prompt: short (tuned)'"
else
  echo "WARNING: cleaner did not report the tuned prompt — it may be running the" >&2
  echo "         base model, which captures 12.6% of the cleanup instead of 76.1%." >&2
  exit 1
fi

echo
echo "both sidecars staged; now run: cargo tauri build"
