from __future__ import annotations

from models.stage1 import MaiGenerator


def load_model(**kwargs):
    return MaiGenerator(**kwargs)

