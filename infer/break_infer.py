from __future__ import annotations

from models.break_stage import BreakClassifier


def load_model(**kwargs):
    return BreakClassifier(**kwargs)

