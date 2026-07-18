"""
train_stage1_v2.py — 增强版 Stage 1 训练

改进:
  - 50 样本, 200 epochs
  - 空拍 (ID 0) 类别加权
  - train/val 分割
  - 定期 checkpoint + 推理验证
"""

import json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from models.common import StageConfig
from models.stage1_chart import Stage1ChartModel


# ============================================================
class PreprocessedDataset(Dataset):
    def __init__(self, data_dir: str, max_frames: int = 1024):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        self.max_frames = max_frames

        vocab_path = self.data_dir / "vocab.json"
        self.chart_vocab = json.loads(vocab_path.read_text("utf-8")) if vocab_path.exists() else {}
        tag_path = self.data_dir / "tag_vocab.json"
        self.tag_vocab = json.loads(tag_path.read_text("utf-8")) if tag_path.exists() else {}

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        meta = json.loads(self.files[idx].with_suffix(".json").read_text("utf-8"))

        audio  = data["audio_tokens"].astype(np.int64)
        beat   = data["beat_signal"].astype(np.float32)
        chart  = data["chart_tokens"].astype(np.int64)
        tag_ids = data.get("tag_ids", np.array([], dtype=np.int64))

        T = audio.shape[0]
        if T > self.max_frames:
            start = np.random.randint(0, T - self.max_frames + 1)
            audio, beat, chart = audio[start:start+self.max_frames], beat[start:start+self.max_frames], chart[start:start+self.max_frames]
            T = self.max_frames

        padded_tags = np.full(32, -1, dtype=np.int64)
        n_tags = min(len(tag_ids), 32)
        padded_tags[:n_tags] = tag_ids[:n_tags]

        diff_map = {"Easy":0,"Basic":1,"Advanced":2,"Expert":3,"Master":4,"Re:Master":5,"UTAGE":6}
        diff_id = diff_map.get(meta["metadata"].get("difficulty_name","Master"), 3)
        level = float(meta["metadata"].get("level", 10.0))

        return {
            "audio": audio, "beat": beat, "chart": chart,
            "difficulty": np.array(diff_id, dtype=np.int64),
            "level": np.array(level, dtype=np.float32),
            "tags": padded_tags,
        }


def collate_fn(batch):
    max_t = max(item["audio"].shape[0] for item in batch)
    C = batch[0]["audio"].shape[1]
    audio_b  = torch.zeros(len(batch), max_t, C, dtype=torch.long)
    beat_b   = torch.zeros(len(batch), max_t, 2)
    chart_b  = torch.zeros(len(batch), max_t, dtype=torch.long)
    diff_b   = torch.zeros(len(batch), dtype=torch.long)
    level_b  = torch.zeros(len(batch))
    tags_b   = torch.zeros(len(batch), batch[0]["tags"].shape[0], dtype=torch.long)
    for i, item in enumerate(batch):
        t = item["audio"].shape[0]
        audio_b[i,:t] = torch.from_numpy(item["audio"])
        beat_b[i,:t]  = torch.from_numpy(item["beat"])
        chart_b[i,:t] = torch.from_numpy(item["chart"])
        diff_b[i]  = int(item["difficulty"])
        level_b[i] = float(item["level"])
        tags_b[i]  = torch.from_numpy(item["tags"])
    return audio_b, beat_b, chart_b, diff_b, level_b, tags_b


# ============================================================
def train(
    data_dir: str = "preprocessed",
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 3e-4,
    device: str = "cpu",
    no_note_weight: float = 0.15,
):
    cfg = StageConfig(
        d_model=256, n_head=4, n_layer=4, d_ff=1024,
        max_seq_len=2048, audio_num_codebooks=4,
        chart_vocab_size=256,
        ahpe_householder_order=2, dropout=0.1,
    )

    # Dataset: 80/20 split
    full_ds = PreprocessedDataset(data_dir, max_frames=1024)
    n_train = int(0.8 * len(full_ds))
    n_val = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # Class weights: 0=no_note 占 ~95%, 降低其权重
    class_weights = torch.ones(cfg.chart_vocab_size)
    class_weights[0] = no_note_weight  # no-note 降权
    class_weights = class_weights.to(device)

    model = Stage1ChartModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Stage 1 Training: {n_params:,} params | {n_train} train / {n_val} val")
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"No-note weight: {no_note_weight} (vocab={cfg.chart_vocab_size})")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    stats = []

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0; n_batches = 0
        for audio, beat, chart, diff, lvl, tags in train_loader:
            audio, beat, chart = audio.to(device), beat.to(device), chart.to(device)
            diff, lvl, tags = diff.to(device), lvl.to(device), tags.to(device)

            optimizer.zero_grad()
            logits = model(audio, beat, diff, lvl, tags, chart_tokens=chart)["logits"]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, cfg.chart_vocab_size),
                chart.reshape(-1).long(),
                weight=class_weights,
                ignore_index=-100,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item(); n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        # --- Val ---
        model.eval()
        val_loss = 0.0; n_val_b = 0
        with torch.no_grad():
            for audio, beat, chart, diff, lvl, tags in val_loader:
                audio, beat, chart = audio.to(device), beat.to(device), chart.to(device)
                diff, lvl, tags = diff.to(device), lvl.to(device), tags.to(device)
                logits = model(audio, beat, diff, lvl, tags, chart_tokens=chart)["logits"]
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, cfg.chart_vocab_size),
                    chart.reshape(-1).long(),
                    weight=class_weights,
                )
                val_loss += loss.item(); n_val_b += 1

        avg_val = val_loss / max(n_val_b, 1)
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"E{epoch+1:4d} | train={avg_train:.4f} val={avg_val:.4f} | lr={lr_now:.2e}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({"model_state_dict": model.state_dict(), "config": cfg},
                       Path(data_dir) / "stage1_best.pt")

        stats.append((avg_train, avg_val))

    # Final save
    torch.save({"model_state_dict": model.state_dict(), "config": cfg},
               Path(data_dir) / "stage1_final.pt")
    print(f"\nBest val loss: {best_val_loss:.4f}")

    return model, cfg, stats


# ============================================================
@torch.no_grad()
def infer_sample(model, cfg, mp3_path: str, output_path: str,
                 difficulty: str = "Master", level: float = 10.0,
                 designer: str = "AI", device: str = "cpu"):
    """对单个 mp3 自回归推理完整歌曲，停止于音频末尾

    Args:
        difficulty: Easy/Basic/Advanced/Expert/Master/Re:Master/UTAGE
        level: 谱面等级
        designer: 谱师名
    """
    from AudioTokenizer import AudioTokenizer
    from BeatTokenizer import BeatTokenizer
    from SimaiToken import SimaiToken

    DIFF_MAP = {"Easy": 1, "Basic": 2, "Advanced": 3, "Expert": 4,
                "Master": 5, "Re:Master": 6, "UTAGE": 7}
    DIFF_ID = {"Easy": 0, "Basic": 1, "Advanced": 2, "Expert": 3,
               "Master": 4, "Re:Master": 5, "UTAGE": 6}

    diff_num = DIFF_MAP.get(difficulty, 4)
    diff_id = DIFF_ID.get(difficulty, 3)

    with open("preprocessed/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

    # 1. 音频 + 节拍
    at = AudioTokenizer(num_codebooks=4)
    ad = at.encode_file(mp3_path)
    bt = BeatTokenizer(method="librosa", target_bpm=None, quantize_beats=True)
    bl = bt.analyse(mp3_path)

    fr = ad.frame_rate  # 75 Hz
    nf = ad.num_frames
    bpm = bl.bpm
    subdiv = 4
    # 每小节时长 (秒)
    measure_dur = subdiv * 60.0 / bpm  # 4/4: 240/bpm

    audio_t = torch.from_numpy(ad.tokens).unsqueeze(0).long().to(device)

    beat_s = np.zeros((nf, 2), dtype=np.float32)
    for b in bl.beats:
        fi = round(b.time * fr)
        if 0 <= fi < nf:
            beat_s[fi, 0] = max(beat_s[fi, 0], 0.5)
            if b.is_downbeat:
                beat_s[fi, 1] = 1.0
    beat_t = torch.from_numpy((beat_s > 0.3).astype(np.float32)).unsqueeze(0).to(device)

    diff_t = torch.tensor([diff_id], device=device)
    lvl_t = torch.tensor([level], device=device)
    tags_t = torch.full((1, 32), -1, dtype=torch.long, device=device)

    # 2. 自回归推理整个序列，到音频末尾停止
    model.eval()
    pred = model.generate(audio_t, beat_t, diff_t, lvl_t, tags_t, temperature=0.8, top_k=50)
    pn = pred[0].cpu().numpy()

    note_count = int((pn > 0).sum())
    print(f"  Notes: {note_count}/{nf} ({100*note_count/nf:.1f}%)")

    # 3. Token → simai (正确的时间→小节映射)
    from collections import defaultdict
    from SimaiToken import SimaiToken
    from SimaiToken import _token_to_simai_note as token_to_simai_note

    measures: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))

    for f in range(nf):
        tid = int(pn[f])
        if tid <= 0:
            continue
        tok_str = id_to_token.get(tid)
        if tok_str is None:
            continue

        # 解析 token → simai 音符字符串
        st = SimaiToken.from_string(tok_str)
        if st is None:
            continue
        simai_note = token_to_simai_note(st)

        t_sec = f / fr
        m = int(t_sec / measure_dur)
        beat_in_measure = (t_sec % measure_dur) / measure_dur
        bi = min(round(beat_in_measure * subdiv), subdiv - 1)
        measures[m][bi].append(simai_note)

    # 构建 simai (含难度头部)
    lines = [f"&title={Path(mp3_path).parent.name}",
             f"&artist={designer}",
             f"&wholebpm={bpm:.1f}",
             f"&lv_{diff_num}={level:.1f}",
             f"&des_{diff_num}={designer}",
             f"&inote_{diff_num}="]
    max_m = max(measures.keys()) if measures else 0
    for m in range(max_m + 1):
        beats = measures.get(m, {})
        parts = []
        for bi in range(subdiv):
            if bi in beats:
                parts.append("/".join(beats[bi]))
            else:
                parts.append("")
        if m == 0:
            lines.append(f"({bpm:.1f}){{{subdiv}}}{','.join(parts)}")
        else:
            lines.append(f"{{{subdiv}}}{','.join(parts)}")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  Measures: {max_m + 1} | Saved: {output_path}")


# ============================================================
if __name__ == "__main__":
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg, stats = train(
        data_dir="preprocessed", epochs=200, batch_size=4,
        lr=3e-4, device=device, no_note_weight=0.15,
    )

    # 推理
    print(f"\n{'='*60}")
    print("Inference on validation sample")
    infer_sample(model, cfg, "samples/人是猫/track.mp3",
                 "samples/人是猫/maidata_ai.txt",
                 difficulty="Master", level=10.0, designer="AI", device=device)
