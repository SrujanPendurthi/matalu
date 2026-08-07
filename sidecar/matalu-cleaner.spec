# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the matalu *cleanup* sidecar — the QLoRA-tuned Qwen that
# strips fillers and false starts. Sibling of matalu-sidecar.spec (ASR).
#
# Build with:  pyinstaller --clean --noconfirm matalu-cleaner.spec   (from sidecar/)
# Output:      dist/matalu-cleaner   (onefile) — staged by build_sidecar.sh.
#
# Two things differ from the ASR spec:
#
#  1. It collects `mlx_lm` rather than `parakeet_mlx`, which drags in transformers
#     and its tokenizer data — the heaviest and most hook-sensitive part.
#  2. **It bundles the adapter.** Without it the app loads the base model, which
#     captures 12.6% of the achievable cleanup instead of 76.1%, and the only
#     symptom is one line in the startup log. Shipping the binary without the
#     adapter is the quiet failure this spec exists to prevent.
#
# Verify the produced binary loads on a machine with no MLX installed before
# shipping; the collect_all() hooks are the fragile part.

import os
from PyInstaller.utils.hooks import collect_all

# SPECPATH (injected by PyInstaller) rather than getcwd(), so the build does not
# depend on which directory it was launched from.
here = os.path.abspath(SPECPATH)  # noqa: F821
adapter = os.path.join(here, "adapters", "cleanup")
if not os.path.isdir(adapter):
    raise SystemExit(
        f"error: no cleanup adapter at {adapter}\n"
        "The adapter is committed to the repo; if it is missing, retrain with\n"
        "  python3 -m mlx_lm lora --train -c training/lora_config.yaml\n"
        "Refusing to build a cleaner that would silently run the base model."
    )

datas, binaries, hiddenimports = [], [], []
for pkg in ("mlx", "mlx_lm"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Mirrors resolve_adapter()'s layout: <resource dir>/adapters/cleanup.
# clean_sidecar.py resolves that dir via sys._MEIPASS when frozen.
datas += [(adapter, os.path.join("adapters", "cleanup"))]

a = Analysis(
    [os.path.join(here, "clean_sidecar.py")],
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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="matalu-cleaner",
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
