from __future__ import annotations

import argparse

from models.stage1 import MaiGenerator


def build_model(hidden_dim: int = 512, num_layers: int = 8, num_heads: int = 8):
    return MaiGenerator(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    args = parser.parse_args()
    model = build_model(args.hidden_dim, args.num_layers, args.num_heads)
    print(model.__class__.__name__)


if __name__ == "__main__":
    main()
