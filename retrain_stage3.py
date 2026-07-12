"""
retrain_stage3.py — 使用含 timing 的新词表重新训练 Stage 3

新词表: slide_vocab_with_timing.json (5509 条, path+timing)
模型参数: 与服务器 checkpoint 一致 (d_model=512, n_head=8, n_layer=6, audio_num_codebooks=8)
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.common import StageConfig
from models.stage3_slide import Stage3SlideModel

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("preprocessed")
DATASETS_DIR = Path("datasets")
NEW_SLIDE_VOCAB_PATH = DATA_DIR / "slide_vocab_with_timing.json"
OUTPUT_CKPT = Path("checkpoints") / "stage3_retrained.pt"

# 模型参数 (与服务器 checkpoint 一致: n_layer=6)
CFG = StageConfig(
    d_model=512,
    n_head=8,
    n_layer=6,           # 服务器实际为 6 层
    d_ff=2048,
    dropout=0.1,
    max_seq_len=4096,
    audio_codebook_size=1024,
    audio_num_codebooks=8,
    chart_vocab_size=5320,
    slide_vocab_size=5509,  # 新词表大小
    max_slide_slots=8,
)

BATCH_SIZE = 2
GRAD_ACCUM = 4            # 有效 batch = 8
EPOCHS = 50
LR = 3e-4
SAVE_EVERY = 5
MAX_FRAMES = 1024

# ═══════════════════════════════════════════════════════════
# 加载新词表
# ═══════════════════════════════════════════════════════════
with open(NEW_SLIDE_VOCAB_PATH, "r", encoding="utf-8") as f:
    NEW_SLIDE_VOCAB = json.load(f)  # {token: id}
assert len(NEW_SLIDE_VOCAB) == CFG.slide_vocab_size, \
    f"Vocab size {len(NEW_SLIDE_VOCAB)} != config {CFG.slide_vocab_size}"

with open(DATA_DIR / "vocab.json", "r", encoding="utf-8") as f:
    CHART_VOCAB = json.load(f)

# slide token ids 在 chart vocab 中
SLIDE_CHART_IDS = {tid for tok, tid in CHART_VOCAB.items() if tok.startswith("slide")}


# ═══════════════════════════════════════════════════════════
# 从原始 maidata.txt 提取 slide path+timing 标签 (按序号对齐)
# ═══════════════════════════════════════════════════════════

SLIDE_RE = re.compile(
    r'(?P<start>\d+)(?P<flags>[bx]*)'
    r'(?P<path>(?:pp|qq|[><^vVpqszw\-])\d*(?:'
    r'(?:pp|qq|[><^vVpqszw\-*])\d*)*)'
    r'\[(?P<timing>[^\]]+)\]'
)


def extract_slide_labels(song_id: str) -> dict[int, int]:
    """
    从 datasets/{song_id}/maidata.txt 全文提取 slide path+timing
    返回: {slide_index: token_id}  (按出现顺序编号)
    """
    maidata_path = DATASETS_DIR / song_id / "maidata.txt"
    if not maidata_path.exists():
        return {}

    try:
        content = maidata_path.read_text(encoding="utf-8")
    except Exception:
        return {}

    # 直接在全文中提取 (与 build_slide_vocab 完全一致)
    labels = {}
    idx = 0
    for m in SLIDE_RE.finditer(content):
        path = m.group("path")
        timing = m.group("timing")
        if not path:
            continue
        token_str = f"{path}[{timing}]"
        tid = NEW_SLIDE_VOCAB.get(token_str, 0)
        if tid > 0:
            labels[idx] = tid
            idx += 1

    return labels


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

class Stage3Dataset(Dataset):
    def __init__(self, npz_files: list[Path]):
        self.files = npz_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        sid = npz_path.stem

        data = np.load(npz_path)
        T_orig = data["audio_tokens"].shape[0]
        T = min(T_orig, MAX_FRAMES)
        start = np.random.randint(0, max(1, T_orig - T + 1)) if T_orig > T else 0

        audio = torch.from_numpy(
            data["audio_tokens"][start:start+T].astype(np.int64)
        )
        beat = torch.from_numpy(
            data["beat_signal"][start:start+T].astype(np.float32)
        )
        chart = torch.from_numpy(
            data["chart_tokens"][start:start+T].astype(np.int64)
        )

        return {
            "audio": audio,
            "beat": beat,
            "chart": chart,
            "sid": sid,
            "start": start,
            "T": T,
        }


def collate_fn(batch):
    """padding to max T in batch"""
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
        "audio": audio,
        "beat": beat,
        "chart": chart,
        "sids": [b["sid"] for b in batch],
        "starts": [b["start"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════

def train():
    print(f"Device: {DEVICE}")
    print(f"Slide vocab size: {len(NEW_SLIDE_VOCAB)}")
    print(f"Chart vocab size: {len(CHART_VOCAB)}")
    print(f"Slide chart token IDs: {len(SLIDE_CHART_IDS)}")
    print()

    # 收集 npz 文件
    npz_files = sorted(DATA_DIR.glob("*.npz"))
    print(f"Total npz files: {len(npz_files)}")

    # 划分 train/val
    split = int(len(npz_files) * 0.9)
    train_files = npz_files[:split]
    val_files = npz_files[split:]
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_dataset = Stage3Dataset(train_files)
    val_dataset = Stage3Dataset(val_files)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    # 预计算 slide labels
    print("Precomputing slide labels...")
    slide_labels: dict[str, dict[int, int]] = {}  # sid → {slide_index → token_id}
    all_files = train_files + val_files
    for i, npz_path in enumerate(all_files):
        sid = npz_path.stem
        labels = extract_slide_labels(sid)
        if labels:
            slide_labels[sid] = labels
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(all_files)}...")
    print(f"  Songs with slide labels: {len(slide_labels)}")

    # 模型
    model = Stage3SlideModel(CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_slides = 0
        t0 = time.time()
        optimizer.zero_grad()
        accum_step = 0

        for batch_idx, batch in enumerate(train_loader):
            audio = batch["audio"].to(DEVICE)
            beat = batch["beat"].to(DEVICE)
            chart = batch["chart"].to(DEVICE)
            B, T = chart.shape

            # 构建 slide targets: 按 slide 序号对齐
            slide_tgt = torch.zeros(B, T, 1, dtype=torch.long, device=DEVICE)
            slide_mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)

            for b in range(B):
                sid = batch["sids"][b]
                if sid not in slide_labels:
                    continue
                labels = slide_labels[sid]  # {slide_index: token_id}
                if not labels:
                    continue

                # 找到当前窗口内所有 slide chart token 位置
                slide_positions = [t for t in range(T)
                                   if int(chart[b, t].item()) in SLIDE_CHART_IDS]

                # 按序号对齐
                max_n = min(len(slide_positions), max(labels.keys()) + 1)
                for i in range(max_n):
                    if i in labels:
                        frame = slide_positions[i]
                        slide_tgt[b, frame, 0] = labels[i]
                        slide_mask[b, frame] = True

            if slide_mask.sum() == 0:
                continue

            # 难度/等级 (简化: 全用 Master 10)
            diff_t = torch.full((B,), 4, dtype=torch.long, device=DEVICE)
            lvl_t = torch.full((B,), 10.0, dtype=torch.float32, device=DEVICE)
            tags_t = torch.full((B, 32), -1, dtype=torch.long, device=DEVICE)

            out = model(chart, audio, beat, diff_t, lvl_t, tags_t,
                        slide_path_targets=slide_tgt,
                        slide_mask=slide_mask)
            loss = out["loss"] / GRAD_ACCUM
            if loss.item() == 0:
                continue

            loss.backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM
            total_slides += slide_mask.sum().item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | "
              f"slides={total_slides} | time={elapsed:.0f}s | lr={scheduler.get_last_lr()[0]:.2e}")

        # 保存
        if epoch % SAVE_EVERY == 0:
            ckpt_path = OUTPUT_CKPT.parent / f"stage3_retrained_e{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": CFG,
                "slide_vocab": NEW_SLIDE_VOCAB,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # 最终保存
    OUTPUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": CFG,
        "slide_vocab": NEW_SLIDE_VOCAB,
        "epoch": EPOCHS,
    }, OUTPUT_CKPT)
    print(f"\nFinal model saved: {OUTPUT_CKPT}")


if __name__ == "__main__":
    train()
