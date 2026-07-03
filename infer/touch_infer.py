from __future__ import annotations

from models.touch_stage import TouchRefiner


def load_model(**kwargs):
    return TouchRefiner(**kwargs)

