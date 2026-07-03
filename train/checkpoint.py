from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device | str) -> None:
    device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 原子写入：先写临时文件再重命名，避免进程被杀死导致文件损坏
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # 原子 rename（同磁盘）


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def pack_config(cfg: Any) -> Any:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return cfg

