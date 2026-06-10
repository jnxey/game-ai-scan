# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

easyocr_datas, easyocr_binaries, easyocr_hiddenimports = collect_all("easyocr")

a = Analysis(
    ["ai-scan.py"],
    pathex=[],
    binaries=easyocr_binaries,
    datas=[
        ("easyocr_models", "easyocr_models"),
        ("certs", "certs"),
        ("templates", "templates"),
        ("poker-best8m.pt", "."),
        ("majiang-best8m.pt", "."),
        ("chips-best8m.pt", "."),
        ("prerun.png", "."),
        ("data-chips.yaml", "."),
        ("data-majiang.yaml", "."),
        ("data-poker.yaml", "."),
        *easyocr_datas,
    ],
    hiddenimports=easyocr_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["hooks/runtime_ssl.py"],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ai-scan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
