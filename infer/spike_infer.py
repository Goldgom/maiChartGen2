from __future__ import annotations

from models.spike_stage import SpikeClassifier


def load_model(**kwargs):
    return SpikeClassifier(**kwargs)

