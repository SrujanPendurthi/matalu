# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the matalu ASR sidecar (a self-contained executable so the
# packaged Tauri app doesn't depend on a system Python / pip environment).
#
# Build with:  pyinstaller --clean --noconfirm matalu-sidecar.spec   (from sidecar/)
# Output:      dist/matalu-sidecar   (onefile) — staged by build_sidecar.sh.
#
# NOTE: mlx / parakeet-mlx pull in native Metal libraries and data files, which
# PyInstaller doesn't discover automatically — hence collect_all() below. This
# is the fragile part of bundling; verify the produced binary actually loads the
# model on a clean machine (no MLX installed) before shipping.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("mlx", "parakeet_mlx"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["matalu_sidecar.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

# onefile: everything is embedded in EXE (no COLLECT step).
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="matalu-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
