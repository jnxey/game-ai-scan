import os
import sys

import easyocr
import torch

_reader = None


def get_model_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "easyocr_models")
    return os.environ.get("EASYOCR_MODEL_DIR", os.path.expanduser("~/.EasyOCR/model"))


def get_reader():
    global _reader
    if _reader is None:
        frozen = getattr(sys, "frozen", False)
        _reader = easyocr.Reader(
            ["en"],
            gpu=torch.cuda.is_available(),
            verbose=False,
            model_storage_directory=get_model_dir(),
            download_enabled=not frozen,
        )
    return _reader
