"""
Phase 1: 音频特征提取 → cache/_audio/

从 datasets/{song_id}/track.mp3 提取 onset/chroma/centroid，
独立存储到 cache/_audio/，供所有 Stage 共享。

用法:
  python scripts/preprocess_audio.py                      # 全部
  python scripts/preprocess_audio.py --limit 10            # 前10首
  python scripts/preprocess_audio.py --num-workers 4       # 4线程
  python scripts/preprocess_audio.py --skip-existing       # 跳过已有
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_audio")

_RE_BPM = re.compile(r"\((\d+(?:\.\d+)?)\)")


def parse_bpm(text: str) -> float:
    for line in text.split("\n")[:20]:
        m = _RE_BPM.search(line)
        if m:
            return float(m.group(1))
    return 150.0


def extract_one(folder: Path, subdiv: int = 64, sr: int = 22050, encodec_layers: int = 1) -> dict[str, Any]:
    """提取单曲音频特征 + EnCodec tokens。"""
    name = folder.name
    audio_path = folder / "track.mp3"
    maidata_path = folder / "maidata.txt"

    if not audio_path.exists():
        return {"folder": str(folder), "error": "missing_audio"}
    if not maidata_path.exists():
        return {"folder": str(folder), "error": "missing_maidata"}

    text = maidata_path.read_text(encoding="utf-8")
    bpm = parse_bpm(text)

    # --- Spectral features ---
    try:
        from utils.audio_features import extract_features
        feats = extract_features(str(audio_path), bpm=bpm, subdiv=subdiv, sr=sr)
    except Exception as e:
        return {"folder": str(folder), "error": f"extract: {e}"}

    # --- EnCodec tokens (Stream A) ---
    audio_tokens = None
    try:
        from Tokenizer.MaiTrackTokenizer import MaiTrackTokenizer
        etok = MaiTrackTokenizer(n_layers=encodec_layers, device="cpu")
        audio_tokens = etok.encode(str(audio_path), n_layers=encodec_layers,
                                   add_bos=False, add_eos=False, interleave=False)
    except Exception:
        pass  # EnCodec 失败不影响其他特征

    return {
        "folder": str(folder),
        "onset": torch.from_numpy(feats.onset.copy()).float(),
        "chroma": torch.from_numpy(feats.chroma.copy()).float(),
        "centroid": torch.from_numpy(feats.centroid.copy()).float(),
        "num_slots": feats.num_slots,
        "bpm": bpm,
        "audio_tokens": torch.tensor(audio_tokens, dtype=torch.long) if audio_tokens else torch.zeros(0, dtype=torch.long),
    }


def save_audio(result: dict, cache_root: Path) -> None:
    name = Path(result["folder"]).name
    (cache_root / f"{name}.pt").parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "onset": result["onset"],
        "chroma": result["chroma"],
        "centroid": result["centroid"],
        "num_slots": result["num_slots"],
        "bpm": result["bpm"],
        "audio_tokens": result.get("audio_tokens", torch.zeros(0, dtype=torch.long)),
    }, cache_root / f"{name}.pt")


def main():
    p = argparse.ArgumentParser(description="Phase 1: 音频特征提取 → cache/_audio/")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--cache-root", default="/data/maiG_v2/cache/_audio")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--subdiv", type=int, default=64)
    p.add_argument("--force", action="store_true", help="强制重新处理")
    p.add_argument("--encodec-layers", type=int, default=1, help="EnCodec 层数 (1/2)")
    args = p.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted(
        [d for d in data_root.iterdir() if d.is_dir() and (d / "maidata.txt").exists()],
        key=lambda x: x.name,
    )
    logger.info(f"找到 {len(folders)} 首歌曲")

    if args.limit:
        folders = folders[:args.limit]

    if not args.force:
        before = len(folders)
        folders = [f for f in folders if not (cache_dir / f"{f.name}.pt").exists()]
        logger.info(f"跳过 {before - len(folders)} 首已有，需处理 {len(folders)} 首")

    if not folders:
        logger.info("无需处理")
        return

    ok = fail = 0
    if args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            fut = {ex.submit(extract_one, f, args.subdiv, 22050, args.encodec_layers): f for f in folders}
            for fu in as_completed(fut):
                r = fu.result()
                if "error" in r:
                    logger.warning(f"  ✗ {fut[fu].name}: {r['error']}"); fail += 1
                else:
                    save_audio(r, cache_dir); ok += 1
                if (ok + fail) % 50 == 0:
                    logger.info(f"进度: {ok}✓ / {fail}✗ / {len(folders)}")
    else:
        for i, f in enumerate(folders):
            logger.info(f"[{i+1}/{len(folders)}] {f.name}")
            r = extract_one(f, args.subdiv, 22050, args.encodec_layers)
            if "error" in r:
                logger.warning(f"  ✗ {r['error']}"); fail += 1
            else:
                save_audio(r, cache_dir); ok += 1

    logger.info(f"完成! 成功: {ok}, 失败: {fail}")
    (cache_dir / "manifest.json").write_text(json.dumps(
        {"ok": ok, "fail": fail, "subdiv": args.subdiv}, indent=2))


if __name__ == "__main__":
    main()
