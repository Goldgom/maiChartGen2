"""
train_stage1.py — Stage 1 训练 + 推理验证脚本

用少量预处理样本跑通完整训练/推理流程。
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.common import StageConfig
from models.stage1_chart import Stage1ChartModel


# ============================================================
# Dataset
# ============================================================

class PreprocessedDataset(Dataset):
    """加载预处理 .npz 文件的 Dataset"""

    def __init__(self, data_dir: str, max_frames: int = 4096):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        self.max_frames = max_frames

        # 加载 vocab
        vocab_path = self.data_dir / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.chart_vocab = json.load(f)
        else:
            self.chart_vocab = {}

        tag_path = self.data_dir / "tag_vocab.json"
        if tag_path.exists():
            with open(tag_path, "r", encoding="utf-8") as f:
                self.tag_vocab = json.load(f)
        else:
            self.tag_vocab = {}

        print(f"Dataset: {len(self.files)} files, "
              f"chart_vocab={len(self.chart_vocab)}, "
              f"tag_vocab={len(self.tag_vocab)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        meta_path = self.files[idx].with_suffix(".json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        audio = data["audio_tokens"].astype(np.int64)      # (T, C)
        beat = data["beat_signal"].astype(np.float32)       # (T, 2)
        chart = data["chart_tokens"].astype(np.int64)       # (T,)
        tag_ids = data.get("tag_ids", np.array([], dtype=np.int64))

        # 截断/填充到 max_frames
        T = audio.shape[0]
        if T > self.max_frames:
            start = np.random.randint(0, T - self.max_frames + 1)
            audio = audio[start:start + self.max_frames]
            beat = beat[start:start + self.max_frames]
            chart = chart[start:start + self.max_frames]
            T = self.max_frames

        # 补齐 tag_ids 到 max_tags=32
        max_tags = 32
        padded_tags = np.full(max_tags, -1, dtype=np.int64)
        n_tags = min(len(tag_ids), max_tags)
        padded_tags[:n_tags] = tag_ids[:n_tags]

        # 难度 ID (1~6 → 0~5)
        diff_name = meta["metadata"].get("difficulty_name", "Master")
        diff_map = {"Easy": 0, "Basic": 1, "Advanced": 2, "Expert": 3, "Master": 4, "Re:Master": 5, "UTAGE": 6}
        diff_id = diff_map.get(diff_name, 3)

        level = float(meta["metadata"].get("level", 10.0))

        return {
            "audio": audio,
            "beat": beat,
            "chart": chart,
            "difficulty": np.array(diff_id, dtype=np.int64),
            "level": np.array(level, dtype=np.float32),
            "tags": padded_tags,
        }


def collate_fn(batch):
    """补齐到 batch 内最大长度"""
    max_t = max(item["audio"].shape[0] for item in batch)
    C = batch[0]["audio"].shape[1]

    audio_b = torch.zeros(len(batch), max_t, C, dtype=torch.long)
    beat_b = torch.zeros(len(batch), max_t, 2)
    chart_b = torch.zeros(len(batch), max_t, dtype=torch.long)
    diff_b = torch.zeros(len(batch), dtype=torch.long)
    level_b = torch.zeros(len(batch))
    tags_b = torch.zeros(len(batch), batch[0]["tags"].shape[0], dtype=torch.long)

    for i, item in enumerate(batch):
        t = item["audio"].shape[0]
        audio_b[i, :t] = torch.from_numpy(item["audio"])
        beat_b[i, :t] = torch.from_numpy(item["beat"])
        chart_b[i, :t] = torch.from_numpy(item["chart"])
        diff_b[i] = int(item["difficulty"])
        level_b[i] = float(item["level"])
        tags_b[i] = torch.from_numpy(item["tags"])

    return audio_b, beat_b, chart_b, diff_b, level_b, tags_b


# ============================================================
# Training
# ============================================================

def train_stage1(
    data_dir: str = "preprocessed",
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 1e-4,
    device: str = "cpu",
):
    """训练 Stage 1 模型"""

    # Config
    cfg = StageConfig(
        d_model=256, n_head=4, n_layer=3, d_ff=1024,
        max_seq_len=4096, audio_num_codebooks=4,
        chart_vocab_size=128,  # 略大于实际 vocab
        ahpe_householder_order=2,
    )

    # Data
    dataset = PreprocessedDataset(data_dir, max_frames=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    # Model
    model = Stage1ChartModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"Training Stage 1: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Device: {device}, Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"Data: {len(dataset)} samples")
    print(f"{'='*60}\n")

    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        for audio, beat, chart, diff, lvl, tags in loader:
            audio = audio.to(device)
            beat = beat.to(device)
            chart = chart.to(device)
            diff = diff.to(device)
            lvl = lvl.to(device)
            tags = tags.to(device)

            optimizer.zero_grad()
            output = model(audio, beat, diff, lvl, tags, chart_tokens=chart)
            loss = output["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

    print(f"\nTraining complete!")
    return model, cfg


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def inference_stage1(model, cfg, dataset, num_samples: int = 2, device: str = "cpu"):
    """在验证集上推理并展示结果"""
    from SimaiToken import SimaiTokenType

    model.eval()
    # 加载 vocab 反向映射
    with open(Path(dataset.data_dir) / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

    print(f"\n{'='*60}")
    print("Inference (Stage 1)")
    print(f"{'='*60}")

    for idx in range(min(num_samples, len(dataset))):
        item = dataset[idx]

        audio = torch.from_numpy(item["audio"]).unsqueeze(0).to(device).long()
        beat = torch.from_numpy(item["beat"]).unsqueeze(0).to(device)
        diff = torch.tensor([item["difficulty"].item()]).to(device)
        lvl = torch.tensor([item["level"].item()]).to(device)
        tags = torch.from_numpy(item["tags"]).unsqueeze(0).to(device)

        # 生成
        pred = model.generate(audio, beat, diff, lvl, tags, temperature=0.8, top_k=50)
        pred = pred[0].cpu().numpy()
        target = item["chart"]

        # 统计
        pred_notes = (pred > 0).sum()
        tgt_notes = (target > 0).sum()

        # 样本对比
        print(f"\n  Sample {idx+1}: pred_notes={pred_notes}, target_notes={tgt_notes}")
        print(f"  First 15 predicted tokens:")
        count = 0
        for i in range(len(pred)):
            if pred[i] > 0:
                tok = id_to_token.get(int(pred[i]), f"<unk:{pred[i]}>")
                tgt_tok = id_to_token.get(int(target[i]), f"<unk:{target[i]}>")
                print(f"    f{i:5d}: pred={tok:12s}  target={tgt_tok:12s}")
                count += 1
                if count >= 15:
                    break

    return pred


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 训练
    model, cfg = train_stage1(
        data_dir="preprocessed",
        epochs=20,
        batch_size=2,
        lr=1e-4,
        device=device,
    )

    # 推理
    dataset = PreprocessedDataset("preprocessed", max_frames=512)
    inference_stage1(model, cfg, dataset, num_samples=2, device=device)

    # 保存 checkpoint
    ckpt_path = Path("preprocessed/stage1_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model": model.state_dict(),
        "config": cfg,
        "cfg": cfg,
        "stage": 1,
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")
