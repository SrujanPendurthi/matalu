#!/usr/bin/env bash
# Build the self-contained PyInstaller sidecar and stage it where Tauri's
# `externalBin` expects it (src-tauri/binaries/matalu-sidecar-<triple>).
#
# Run before `cargo tauri build`. Requires pyinstaller + the sidecar's Python
# deps (parakeet-mlx, mlx) in the active environment:
#     python3 -m pip install pyinstaller parakeet-mlx
#
# This replaces the committed placeholder stub with the real binary. Do NOT
# commit the result — it is large (bundles MLX) and machine/arch specific.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
triple="aarch64-apple-darwin"   # Apple Silicon; matalu is macOS-only by construction
dest="$root/src-tauri/binaries/matalu-sidecar-$triple"

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "error: pyinstaller not found — python3 -m pip install pyinstaller" >&2
  exit 1
fi

cd "$here"
pyinstaller --clean --noconfirm matalu-sidecar.spec

mkdir -p "$root/src-tauri/binaries"
cp "dist/matalu-sidecar" "$dest"
chmod +x "$dest"
echo "staged sidecar -> $dest"
echo "now run: cargo tauri build"
