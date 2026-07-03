from __future__ import annotations

import argparse

from models.break_stage import BreakClassifier


def build_model(**kwargs):
    return BreakClassifier(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()
    print(build_model(hidden_dim=args.hidden_dim).__class__.__name__)


if __name__ == "__main__":
    main()
