"""
server_pipeline_stage3.py — 服务器端完整流程: 重建 slide 词表 → 重训 Stage 3

用法:
  python server_pipeline_stage3.py [--data_dir /data/maiG_v2/preprocessed]
                                   [--datasets_dir datasets]
                                   [--ckpt_dir /data/maiG_v2/checkpoints]
                                   [--device cuda]

步骤:
  1. 扫描 datasets/ 中所有 maidata.txt, 用 SimaiPaser 提取 slide path+timing → slide_vocab_with_timing.json
  2. 用新词表训练 Stage3SlideModel
  3. 保存 checkpoint (含新 vocab)
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.common import StageConfig
from models.stage3_slide import Stage3SlideModel
from SimaiPaser import (
    SimaiData, SimaiChart,
    extract_slide_tokens_from_file, extract_slide_tokens_from_chart,
    DIFFICULTY_NAME_TO_NUM, _split_slide_path,
)

# ═══════════════════════════════════════════════════════════
# Step 1: 重建 slide_vocab_with_timing.json (使用 SimaiPaser)
# ═══════════════════════════════════════════════════════════

_TIMING_RE = re.compile(r"\[([^\]]+)\]")


def _slide_token_from_obj(obj: dict) -> list[str]:
    """Extract path+timing tokens from a preprocessed frame object."""
    if obj.get("type") != "slide":
        return []

    path = obj.get("path", "")
    if not path:
        return []

    timing = ""
    raw = obj.get("raw", "")
    matches = _TIMING_RE.findall(raw)
    if matches:
        timing = matches[-1]
    elif obj.get("dur"):
        timing = str(obj["dur"])

    tokens = []
    for seg in _split_slide_path(path):
        tokens.append(f"{seg}[{timing}]" if timing else seg)
    return tokens


def extract_slide_frame_labels_from_preprocessed(
    meta_path: Path,
    slide_vocab: dict | None = None,
) -> tuple[dict[int, list[int]], list[str]]:
    """Read preprocessed frame_objects and return absolute-frame slide labels."""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, []

    frame_objects = meta.get("frame_objects", {})
    if not isinstance(frame_objects, dict):
        return {}, []

    labels: dict[int, list[int]] = {}
    token_strings: list[str] = []
    for frame_str, objects in frame_objects.items():
        if not isinstance(objects, list):
            continue
        try:
            frame = int(frame_str)
        except (TypeError, ValueError):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            for token_str in _slide_token_from_obj(obj):
                token_strings.append(token_str)
                if slide_vocab is None:
                    continue
                tid = slide_vocab.get(token_str, 0)
                if tid > 0:
                    labels.setdefault(frame, []).append(tid)

    return labels, token_strings


def build_slide_vocab(datasets_dir: str, output_path: str, data_dir: str | None = None) -> dict:
    """扫描训练数据, 构建含 timing 的 slide 词表.

    Prefer preprocessed frame_objects because many local maidata.txt files are empty
    while the preprocessed charts still contain raw slide notes.
    """
    ds_dir = Path(datasets_dir)
    all_tokens = Counter()

    preprocessed_with_slides = 0
    if data_dir is not None:
        json_files = sorted(Path(data_dir).glob("*.json"))
        json_files = [
            p for p in json_files
            if p.name not in {"vocab.json", "tag_vocab.json", "slide_vocab.json",
                              "slide_vocab_with_timing.json"}
        ]
        print(f"Scanning {len(json_files)} preprocessed chart JSON files...")
        for i, jp in enumerate(json_files):
            _, tokens = extract_slide_frame_labels_from_preprocessed(jp)
            if tokens:
                preprocessed_with_slides += 1
                all_tokens.update(tokens)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(json_files)}, {preprocessed_with_slides} with slides, "
                      f"{len(all_tokens)} unique tokens...")

    maidata_files = sorted(ds_dir.glob("*/maidata.txt"))
    print(f"Scanning {len(maidata_files)} maidata.txt files for fallback/additional tokens...")

    for i, mp in enumerate(maidata_files):
        try:
            tokens = extract_slide_tokens_from_file(mp)
        except Exception:
            continue
        all_tokens.update(tokens)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(maidata_files)}, {len(all_tokens)} unique tokens...")

    vocab = {"<PAD>": 0, "<EOS>": 1}
    for token, _ in all_tokens.most_common():
        vocab[token] = len(vocab)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Slide vocab: {len(vocab)} tokens → {output_path}")
    print(f"  Preprocessed charts with slides: {preprocessed_with_slides}")
    print(f"  Top 10: {all_tokens.most_common(10)}")
    return vocab


# ═══════════════════════════════════════════════════════════
# Step 2: 从 SimaiPaser 解析的谱面提取 slide 标签 (按出现顺序)
# ═══════════════════════════════════════════════════════════

def extract_slide_labels_from_chart(chart_id: str, difficulty: int,
                                     datasets_dir: str,
                                     slide_vocab: dict,
                                     debug_first: bool = False) -> dict:
    """
    用 SimaiPaser 解析 maidata.txt, 提取指定难度的 slide path+timing,
    按出现顺序编号。

    参数:
      chart_id: 谱面基础 ID (如 "100018")
      difficulty: 难度编号 (1=Easy ... 5=Master, 6=Re:Master, 7=UTAGE)
      datasets_dir: 数据集根目录
      slide_vocab: slide 词表 {token_str: id}

    返回: {slide_order_index: token_id}
    """
    maidata_path = Path(datasets_dir) / chart_id / "maidata.txt"
    if not maidata_path.exists():
        if debug_first:
            print(f"    [DEBUG] chart_id={chart_id}: maidata.txt not found at {maidata_path}")
        return {}

    try:
        data = SimaiData.load(maidata_path)
    except Exception:
        if debug_first:
            print(f"    [DEBUG] chart_id={chart_id}: SimaiData.load failed")
        return {}

    chart = data.charts.get(difficulty)
    if chart is None:
        if debug_first:
            print(f"    [DEBUG] chart_id={chart_id}: difficulty {difficulty} not found "
                  f"(available: {list(data.charts.keys())})")
        return {}

    # 提取 slide token 字符串 (按谱面顺序)
    slide_tokens = extract_slide_tokens_from_chart(chart)

    # 查词表, 构建 labels
    labels = {}
    for idx, token_str in enumerate(slide_tokens):
        tid = slide_vocab.get(token_str, 0)
        if tid > 0:
            labels[idx] = tid

    if debug_first:
        diff_name = chart.difficulty_name
        print(f"    [DEBUG] chart_id={chart_id}/{diff_name}: "
              f"{len(slide_tokens)} slides extracted, {len(labels)} in vocab")

    return labels


# ═══════════════════════════════════════════════════════════
# Step 3: Dataset + Training
# ═══════════════════════════════════════════════════════════

class Stage3TrainDataset(Dataset):
    def __init__(self, npz_files: list[Path], max_frames: int):
        self.files = npz_files
        self.max_frames = max_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        data = np.load(npz_path)
        T_orig = data["audio_tokens"].shape[0]
        T = min(T_orig, self.max_frames)
        start = np.random.randint(0, max(1, T_orig - T + 1)) if T_orig > T else 0

        audio = torch.from_numpy(data["audio_tokens"][start:start+T].astype(np.int64))
        beat = torch.from_numpy(data["beat_signal"][start:start+T].astype(np.float32))
        chart = torch.from_numpy(data["chart_tokens"][start:start+T].astype(np.int64))
        return {
            "audio": audio, "beat": beat, "chart": chart,
            "sid": npz_path.stem, "start": start, "T": T,
        }


def collate(batch):
    max_T = max(b["T"] for b in batch)
    B = len(batch)
    audio = torch.zeros(B, max_T, 8, dtype=torch.long)
    beat = torch.zeros(B, max_T, 2, dtype=torch.float32)
    chart = torch.zeros(B, max_T, dtype=torch.long)
    for i, b in enumerate(batch):
        T = b["T"]
        audio[i, :T] = b["audio"]
        beat[i, :T] = b["beat"]
        chart[i, :T] = b["chart"]
    return {
        "audio": audio, "beat": beat, "chart": chart,
        "sids": [b["sid"] for b in batch],
        "starts": [b["start"] for b in batch],
    }


def train_stage3(
    data_dir: str,
    datasets_dir: str,
    ckpt_dir: str,
    slide_vocab: dict,
    device: str = "cuda",
):
    data_dir = Path(data_dir)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 加载 chart vocab
    with open(data_dir / "vocab.json", "r", encoding="utf-8") as f:
        chart_vocab = json.load(f)

    # 预计算 slide labels
    print("Precomputing slide labels (frame alignment from preprocessed JSON)...")
    npz_files = sorted(data_dir.glob("*.npz"))
    slide_labels: dict[str, dict[int, list[int]]] = {}  # sid → {absolute_frame → [token_id, ...]}
    for i, npz_path in enumerate(npz_files):
        sid = npz_path.stem
        # 从 metadata JSON 读取 chart_id 和 difficulty
        meta_path = npz_path.with_suffix(".json")
        chart_id = sid  # fallback
        difficulty = 5   # fallback: Master
        if meta_path.exists():
            frame_labels, _ = extract_slide_frame_labels_from_preprocessed(meta_path, slide_vocab)
            if frame_labels:
                slide_labels[sid] = frame_labels
                if i < 3:
                    total_tokens = sum(len(v) for v in frame_labels.values())
                    print(f"    [DEBUG] sid={sid}: {len(frame_labels)} slide frames, "
                          f"{total_tokens} tokens from preprocessed JSON")
                if (i + 1) % 200 == 0:
                    print(f"  {i+1}/{len(npz_files)}... (found {len(slide_labels)} songs with slides)")
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                chart_id = meta.get("metadata", {}).get("chart_id", sid)
                diff_name = meta.get("metadata", {}).get("difficulty_name", "Master")
                difficulty = DIFFICULTY_NAME_TO_NUM.get(
                    diff_name.lower().replace(":", ""), 5)
            except Exception:
                pass
        labels = extract_slide_labels_from_chart(
            chart_id, difficulty, datasets_dir, slide_vocab,
            debug_first=(i < 3),  # 前 3 首显示调试信息
        )
        if labels:
            # Last-resort fallback when no frame_objects exist. These labels are
            # order-only and are aligned later from the start of the full chart.
            slide_labels[sid] = {-idx - 1: [tid] for idx, tid in labels.items()}
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(npz_files)}... (found {len(slide_labels)} songs with slides)")
    print(f"  Songs with slide labels: {len(slide_labels)} / {len(npz_files)}")

    # 数据集
    split = int(len(npz_files) * 0.9)
    train_ds = Stage3TrainDataset(npz_files[:split], max_frames=1024)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              collate_fn=collate, num_workers=2)

    # 模型
    cfg = StageConfig(
        d_model=512, n_head=8, n_layer=6, d_ff=2048, dropout=0.1,
        max_seq_len=4096, audio_codebook_size=1024, audio_num_codebooks=8,
        chart_vocab_size=max(chart_vocab.values()) + 1,
        slide_vocab_size=len(slide_vocab),
        max_slide_slots=8,
    )
    print(f"Model config: d_model={cfg.d_model}, n_layer={cfg.n_layer}, "
          f"slide_vocab={cfg.slide_vocab_size}")

    model = Stage3SlideModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 训练
    EPOCHS = 50
    GRAD_ACCUM = 4
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_slides = 0
        t0 = time.time()
        optimizer.zero_grad()
        step = 0

        for batch in train_loader:
            audio = batch["audio"].to(device)
            beat = batch["beat"].to(device)
            chart = batch["chart"].to(device)
            B, T = chart.shape

            # 构建 targets: 按 slide 序号对齐 (非帧精确, 而是顺序匹配)
            max_slide_slots = cfg.max_slide_slots
            slide_tgt = torch.zeros(B, T, max_slide_slots, dtype=torch.long, device=device)
            slide_mask = torch.zeros(B, T, max_slide_slots, dtype=torch.bool, device=device)
            for b in range(B):
                sid = batch["sids"][b]
                if sid not in slide_labels:
                    continue
                labels = slide_labels[sid]  # {absolute_frame: [token_id, ...]}
                if not labels:
                    continue

                start = batch["starts"][b]
                end = start + T
                if any(frame >= 0 for frame in labels):
                    for abs_frame, token_ids in labels.items():
                        if abs_frame < start or abs_frame >= end:
                            continue
                        rel_frame = abs_frame - start
                        for slot, tid in enumerate(token_ids[:max_slide_slots]):
                            slide_tgt[b, rel_frame, slot] = tid
                            slide_mask[b, rel_frame, slot] = True
                else:
                    # Legacy fallback for charts without frame_objects.
                    slide_ids = {tid for tok, tid in chart_vocab.items()
                                 if any(part.startswith("slide") for part in tok.split("+"))}
                    slide_positions = [t for t in range(T)
                                       if int(chart[b, t].item()) in slide_ids]
                    ordered = [labels[k][0] for k in sorted(labels.keys(), reverse=True)]
                    max_n = min(len(slide_positions), len(ordered))
                    for j in range(max_n):
                        frame = slide_positions[j]
                        slide_tgt[b, frame, 0] = ordered[j]
                        slide_mask[b, frame, 0] = True

            if slide_mask.sum() == 0:
                continue

            diff_t = torch.full((B,), 4, dtype=torch.long, device=device)
            lvl_t = torch.full((B,), 10.0, dtype=torch.float32, device=device)
            tags_t = torch.full((B, 32), -1, dtype=torch.long, device=device)

            out = model(chart, audio, beat, diff_t, lvl_t, tags_t,
                        slide_path_targets=slide_tgt, slide_mask=slide_mask)
            loss = out["loss"] / GRAD_ACCUM
            if loss.item() == 0:
                continue

            loss.backward()
            step += 1
            total_loss += loss.item() * GRAD_ACCUM
            total_slides += slide_mask.sum().item()

            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        print(f"E{epoch:3d} | loss={avg_loss:.4f} | slides={total_slides} | "
              f"{elapsed:.0f}s | lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % 5 == 0:
            ckpt_path = ckpt_dir / f"stage3_v2_e{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "slide_vocab": slide_vocab,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # 最终保存
    final_path = ckpt_dir / "stage3_v2_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "slide_vocab": slide_vocab,
        "epoch": EPOCHS,
    }, final_path)
    print(f"\nFinal: {final_path}")
    return model


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Server pipeline: rebuild slide vocab + retrain Stage 3")
    parser.add_argument("--data_dir", default="/data/maiG_v2/preprocessed",
                        help="预处理数据目录 (含 .npz + vocab.json)")
    parser.add_argument("--datasets_dir", default="datasets",
                        help="原始谱面数据目录 (含 */maidata.txt)")
    parser.add_argument("--ckpt_dir", default="/data/maiG_v2/checkpoints",
                        help="checkpoint 输出目录")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--skip_vocab", action="store_true",
                        help="跳过词表重建 (使用已有文件)")
    args = parser.parse_args()

    print("=" * 60)
    print("Server Pipeline: Slide Vocab + Stage 3 Retrain")
    print("=" * 60)
    print(f"  data_dir:    {args.data_dir}")
    print(f"  datasets_dir: {args.datasets_dir}")
    print(f"  ckpt_dir:    {args.ckpt_dir}")
    print(f"  device:      {args.device}")
    print()

    # ── Step 1: 重建 slide 词表 ──
    vocab_path = Path(args.data_dir) / "slide_vocab_with_timing.json"
    if args.skip_vocab and vocab_path.exists():
        print("[Step 1] Loading existing slide vocab...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            slide_vocab = json.load(f)
    else:
        print("[Step 1] Building slide vocab with timing...")
        slide_vocab = build_slide_vocab(args.datasets_dir, str(vocab_path), data_dir=args.data_dir)

    # 同时覆盖旧词表 (供后续推理使用)
    old_vocab_path = Path(args.data_dir) / "slide_vocab.json"
    with open(old_vocab_path, "w", encoding="utf-8") as f:
        json.dump(slide_vocab, f, ensure_ascii=False, indent=2)
    print(f"  Old vocab replaced: {old_vocab_path}")
    print()

    # ── Step 2: 训练 Stage 3 ──
    print("[Step 2] Training Stage 3...")
    train_stage3(
        data_dir=args.data_dir,
        datasets_dir=args.datasets_dir,
        ckpt_dir=args.ckpt_dir,
        slide_vocab=slide_vocab,
        device=args.device,
    )

    print("\nDone! New checkpoint →", Path(args.ckpt_dir) / "stage3_v2_best.pt")


if __name__ == "__main__":
    main()
