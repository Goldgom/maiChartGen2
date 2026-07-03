from __future__ import annotations

from models.slide_stage import SlidePathGenerator


def load_model(**kwargs):
    return SlidePathGenerator(**kwargs)

