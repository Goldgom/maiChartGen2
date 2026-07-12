"""
train_all_stages.py — Stage 2~5 训练脚本

直接从原始 maidata.txt 读取 hold dur / slide path 真值,
与预处理数据对齐后训练各 Stage。
"""

import json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.common import StageConfig
from models.stage2_hold import Stage2HoldModel
from models.stage3_slide import Stage3SlideModel, build_slide_vocab
from models.stage4_break import Stage4BreakModel
from models.stage5_ex import Stage5ExModel


# ═══════════════════════════════════════════════════════════
# Stage 2: Hold Duration
# ═══════════════════════════════════════════════════════════

def train_stage2(device="cpu", epochs=50, data_dir="preprocessed"):
    """Stage 2: 自回归预测 hold 持续时间"""
    from SimaiPaser import SimaiData

    cfg = StageConfig(d_model=256, n_head=4, n_layer=3, d_ff=1024,
                      max_seq_len=2048, audio_num_codebooks=4,
                      chart_vocab_size=128, hold_dur_bins=32)

    # 收集 hold 持续时间真值
    print("Loading hold duration labels from raw charts...")
    hold_labels = {}
    npz_files = sorted(Path(data_dir).glob("*.npz"))
    for npz_path in npz_files:
        sid = npz_path.stem
        maidata = Path(f"datasets/{sid}/maidata.txt")
        if not maidata.exists():
            continue
        data = SimaiData.parse(maidata.read_text(encoding="utf-8"),
                               target_subdiv=4)
        diffs = data.available_difficulties
        if not diffs:
            continue
        chart = data.charts[max(diffs)]
        durs = []
        for t in chart.tokens:
            if t.token_type.value == "hold" and t.is_note:
                durs.append(_encode_dur(t.params.get("dur", "")))
        hold_labels[sid] = durs

    print(f"  Loaded hold labels for {len(hold_labels)} charts")
    # Simplified training loop
    model = Stage2HoldModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; n = 0
        for npz_path in npz_files:
            sid = npz_path.stem
            if sid not in hold_labels:
                continue
            data = np.load(npz_path)
            T = min(512, data["audio_tokens"].shape[0])
            audio = torch.from_numpy(data["audio_tokens"][:T]).unsqueeze(0).long().to(device)
            beat  = torch.from_numpy(data["beat_signal"][:T]).unsqueeze(0).to(device)

            # 读取 flat chart tokens + 构建 hold mask
            chart = torch.from_numpy(data["chart_tokens"][:T]).unsqueeze(0).long().to(device)
            # 从 vocab 判断哪些 ID 是 hold
            with open(f"{data_dir}/vocab.json") as f:
                vocab = json.load(f)
            hold_ids = set()
            for tok, tid in vocab.items():
                if tok.startswith("hold"):
                    hold_ids.add(tid)
            hold_mask = torch.zeros_like(chart, dtype=torch.bool)
            for hid in hold_ids:
                hold_mask = hold_mask | (chart == hid)

            if hold_mask.sum() == 0:
                continue

            # hold duration targets: 简化为 bin1
            hold_dur_tgt = torch.ones_like(chart) * hold_mask.long()

            optimizer.zero_grad()
            out = model(chart, audio, beat,
                        torch.tensor([3], device=device),
                        torch.tensor([10.], device=device),
                        torch.full((1, 32), -1, dtype=torch.long, device=device),
                        hold_dur_targets=hold_dur_tgt, hold_mask=hold_mask)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1

        if (epoch+1) % 10 == 0:
            print(f"  Stage2 E{epoch+1:3d}: loss={total_loss/max(n,1):.4f}")

    torch.save({"model": model.state_dict(), "cfg": cfg}, f"{data_dir}/stage2.pt")
    print(f"  Stage2 saved")
    return model


# ═══════════════════════════════════════════════════════════
# Stage 3: Slide Path
# ═══════════════════════════════════════════════════════════

def train_stage3(device="cpu", epochs=50, data_dir="preprocessed"):
    """Stage 3: 自回归补全 slide 路径"""
    from SimaiPaser import SimaiData

    # Build slide vocab from all charts
    slide_vocab = {"<PAD>": 0, "<EOS>": 1}
    for npz_path in sorted(Path(data_dir).glob("*.npz")):
        sid = npz_path.stem
        maidata = Path(f"datasets/{sid}/maidata.txt")
        if not maidata.exists():
            continue
        data = SimaiData.parse(maidata.read_text(encoding="utf-8"), target_subdiv=4)
        for chart in data.charts.values():
            v = build_slide_vocab(chart.tokens)
            for k, vid in v.items():
                if k not in slide_vocab:
                    slide_vocab[k] = len(slide_vocab)

    cfg = StageConfig(d_model=256, n_head=4, n_layer=3, d_ff=1024,
                      max_seq_len=2048, audio_num_codebooks=4,
                      chart_vocab_size=128, slide_vocab_size=max(256, len(slide_vocab)))

    print(f"Stage3 slide vocab: {len(slide_vocab)} segments")
    model = Stage3SlideModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; n = 0
        for npz_path in sorted(Path(data_dir).glob("*.npz")):
            data = np.load(npz_path)
            T = min(512, data["audio_tokens"].shape[0])
            audio = torch.from_numpy(data["audio_tokens"][:T]).unsqueeze(0).long().to(device)
            chart_1d = torch.from_numpy(data["chart_tokens"][:T]).long().to(device)

            with open(f"{data_dir}/vocab.json") as f:
                vocab = json.load(f)
            slide_ids = {tid for tok, tid in vocab.items() if tok.startswith("slide")}
            slide_mask_1d = torch.zeros(T, dtype=torch.bool, device=device)
            for sid2 in slide_ids:
                slide_mask_1d = slide_mask_1d | (chart_1d == sid2)
            if slide_mask_1d.sum() == 0:
                continue

            slide_path_tgt = torch.zeros(1, T, 1, dtype=torch.long, device=device)
            slide_path_tgt[:, slide_mask_1d, 0] = 1

            optimizer.zero_grad()
            out = model(chart_1d.unsqueeze(0), audio,
                        torch.from_numpy(data["beat_signal"][:T]).unsqueeze(0).to(device),
                        torch.tensor([3], device=device),
                        torch.tensor([10.], device=device),
                        torch.full((1, 32), -1, dtype=torch.long, device=device),
                        slide_path_targets=slide_path_tgt,
                        slide_mask=slide_mask_1d.unsqueeze(0))
            loss = out["loss"]
            if loss.item() == 0:
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1

        if (epoch+1) % 10 == 0:
            print(f"  Stage3 E{epoch+1:3d}: loss={total_loss/max(n,1):.4f}")

    torch.save({"model": model.state_dict(), "cfg": cfg, "slide_vocab": slide_vocab},
               f"{data_dir}/stage3.pt")
    print(f"  Stage3 saved")
    return model


# ═══════════════════════════════════════════════════════════
# Stage 4: Break
# ═══════════════════════════════════════════════════════════

def train_stage4(device="cpu", epochs=30, data_dir="preprocessed"):
    """Stage 4: 双向预测 break"""
    cfg = StageConfig(d_model=256, n_head=4, n_layer=2, d_ff=1024,
                      max_seq_len=2048, audio_num_codebooks=4,
                      chart_vocab_size=128)
    model = Stage4BreakModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    npz_files = sorted(Path(data_dir).glob("*.npz"))
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; n = 0
        for npz_path in npz_files:
            data = np.load(npz_path)
            T = min(512, data["audio_tokens"].shape[0])
            audio = torch.from_numpy(data["audio_tokens"][:T]).unsqueeze(0).long().to(device)
            chart = torch.from_numpy(data["chart_tokens"][:T]).unsqueeze(0).long().to(device)
            beat = torch.from_numpy(data["beat_signal"][:T]).unsqueeze(0).to(device)
            break_tgt = torch.from_numpy(data["break_mask"][:T].astype(np.int64)).unsqueeze(0).to(device)
            if "object_mask" in data:
                note_mask = torch.from_numpy(data["object_mask"][:T].astype(bool)).unsqueeze(0).to(device)
            else:
                note_mask = (chart > 0).to(device)

            if note_mask.sum() == 0:
                continue

            optimizer.zero_grad()
            out = model(chart, audio, beat, torch.tensor([3],device=device),
                        torch.tensor([10.],device=device),
                        torch.full((1,32),-1,dtype=torch.long,device=device),
                        break_targets=break_tgt, note_mask=note_mask)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1

        if (epoch+1) % 10 == 0:
            print(f"  Stage4 E{epoch+1:3d}: loss={total_loss/max(n,1):.4f}")

    torch.save({"model": model.state_dict(), "cfg": cfg}, f"{data_dir}/stage4.pt")
    print(f"  Stage4 saved")
    return model


# ═══════════════════════════════════════════════════════════
# Stage 5: Ex (仅 DX 谱面)
# ═══════════════════════════════════════════════════════════

def train_stage5(device="cpu", epochs=30, data_dir="preprocessed"):
    """Stage 5: 双向预测 ex (仅含 ex-note 的谱面)"""
    cfg = StageConfig(d_model=256, n_head=4, n_layer=2, d_ff=1024,
                      max_seq_len=2048, audio_num_codebooks=4,
                      chart_vocab_size=128)
    model = Stage5ExModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    npz_files = sorted(Path(data_dir).glob("*.npz"))
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; n = 0
        for npz_path in npz_files:
            data = np.load(npz_path)
            ex_mask_data = data.get("ex_mask", np.zeros(data["chart_tokens"].shape[0], dtype=bool))
            if ex_mask_data.sum() == 0:
                continue  # 跳过无 ex 的谱面

            T = min(512, data["audio_tokens"].shape[0])
            audio = torch.from_numpy(data["audio_tokens"][:T]).unsqueeze(0).long().to(device)
            chart = torch.from_numpy(data["chart_tokens"][:T]).unsqueeze(0).long().to(device)
            beat = torch.from_numpy(data["beat_signal"][:T]).unsqueeze(0).to(device)
            ex_tgt = torch.from_numpy(ex_mask_data[:T].astype(np.int64)).unsqueeze(0).to(device)
            if "object_mask" in data:
                note_mask = torch.from_numpy(data["object_mask"][:T].astype(bool)).unsqueeze(0).to(device)
            else:
                note_mask = (chart > 0).to(device)

            optimizer.zero_grad()
            out = model(chart, audio, beat, torch.tensor([3],device=device),
                        torch.tensor([10.],device=device),
                        torch.full((1,32),-1,dtype=torch.long,device=device),
                        ex_targets=ex_tgt, note_mask=note_mask)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1

        if n > 0 and (epoch+1) % 10 == 0:
            print(f"  Stage5 E{epoch+1:3d}: loss={total_loss/max(n,1):.4f} (n={n})")

    torch.save({"model": model.state_dict(), "cfg": cfg}, f"{data_dir}/stage5.pt")
    print(f"  Stage5 saved")
    return model


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _encode_dur(dur_str: str, max_bins: int = 32) -> int:
    """将 hold 持续时间编码为 bin ID (1-based, 0=无效)"""
    if not dur_str or ":" not in dur_str:
        return 1
    try:
        parts = dur_str.split(":", 1)
        x, y = float(parts[0]), float(parts[1])
        if y == 0:
            return 1
        val = x / y
        bin_id = min(int(np.log2(max(val, 0.0625)) + 5), max_bins - 1) + 1
        return max(1, bin_id)
    except (ValueError, ZeroDivisionError):
        return 1


def _encode_slide_path(path_str: str, vocab: dict) -> int:
    """将 slide 路径第一段编码为 vocab ID"""
    import re
    segs = re.findall(r'\*?(?:pp|qq|[-><^vVpqszw])\d+', path_str)
    seg = segs[0] if segs else path_str
    return vocab.get(seg, 0)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def train_stage2(device="cpu", epochs=50, data_dir="preprocessed"):
    """Stage 2: train hold duration from preprocessed true dur targets."""
    npz_files = sorted(Path(data_dir).glob("*.npz"))
    with open(Path(data_dir) / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    chart_vocab_size = max(vocab.values()) + 1 if vocab else 128

    cfg = StageConfig(d_model=256, n_head=4, n_layer=3, d_ff=1024,
                      max_seq_len=2048, audio_num_codebooks=4,
                      chart_vocab_size=max(128, chart_vocab_size),
                      hold_dur_bins=32)
    model = Stage2HoldModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for npz_path in npz_files:
            data = np.load(npz_path)
            T = min(512, data["audio_tokens"].shape[0])
            if "hold_dur_targets" not in data:
                continue
            hold_np = data["hold_dur_targets"][:T].astype(np.int64)
            if hold_np.sum() == 0:
                continue

            audio = torch.from_numpy(data["audio_tokens"][:T]).unsqueeze(0).long().to(device)
            beat = torch.from_numpy(data["beat_signal"][:T]).unsqueeze(0).to(device)
            chart = torch.from_numpy(data["chart_tokens"][:T]).unsqueeze(0).long().to(device)
            hold_dur = torch.from_numpy(hold_np).unsqueeze(0).long().to(device)
            hold_mask = hold_dur > 0

            optimizer.zero_grad()
            out = model(chart, audio, beat,
                        torch.tensor([3], device=device),
                        torch.tensor([10.], device=device),
                        torch.full((1, 32), -1, dtype=torch.long, device=device),
                        hold_dur_targets=hold_dur,
                        hold_mask=hold_mask)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Stage2 E{epoch+1:3d}: loss={total_loss/max(n,1):.4f}")

    torch.save({"model": model.state_dict(), "cfg": cfg}, f"{data_dir}/stage2.pt")
    print("  Stage2 saved")
    return model


def train_stage3(device="cpu", epochs=50, data_dir="preprocessed"):
    """Stage 3: train slide paths from preprocessed independent slide vocab targets."""
    npz_files = sorted(Path(data_dir).glob("*.npz"))
    with open(Path(data_dir) / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    slide_vocab_path = Path(data_dir) / "slide_vocab.json"
    if slide_vocab_path.exists():
        with open(slide_vocab_path, "r", encoding="utf-8") as f:
            slide_vocab = json.load(f)
    else:
        slide_vocab = {"<PAD>": 0}

    chart_vocab_size = max(vocab.values()) + 1 if vocab else 128
    slide_vocab_size = max(slide_vocab.values()) + 1 if slide_vocab else 1

    cfg = StageConfig(d_model=256, n_head=4, n_layer=3, d_ff=1024,
                      max_seq_len=2048, audio_num_codebooks=4,
                      chart_vocab_size=max(128, chart_vocab_size),
                      slide_vocab_size=max(256, slide_vocab_size))
    model = Stage3SlideModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for npz_path in npz_files:
            data = np.load(npz_path)
            T = min(512, data["audio_tokens"].shape[0])
            if "slide_path_targets" not in data:
                continue
            slide_np = data["slide_path_targets"][:T].astype(np.int64)
            if slide_np.sum() == 0:
                continue

            audio = torch.from_numpy(data["audio_tokens"][:T]).unsqueeze(0).long().to(device)
            beat = torch.from_numpy(data["beat_signal"][:T]).unsqueeze(0).to(device)
            chart = torch.from_numpy(data["chart_tokens"][:T]).unsqueeze(0).long().to(device)
            slide_path = torch.from_numpy(slide_np).unsqueeze(0).long().to(device)
            slide_mask = slide_path > 0

            optimizer.zero_grad()
            out = model(chart, audio, beat,
                        torch.tensor([3], device=device),
                        torch.tensor([10.], device=device),
                        torch.full((1, 32), -1, dtype=torch.long, device=device),
                        slide_path_targets=slide_path,
                        slide_mask=slide_mask)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Stage3 E{epoch+1:3d}: loss={total_loss/max(n,1):.4f}")

    torch.save({"model": model.state_dict(), "cfg": cfg, "slide_vocab": slide_vocab},
               f"{data_dir}/stage3.pt")
    print("  Stage3 saved")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    print("=== Stage 2: Hold Durations ===")
    train_stage2(device, epochs=20)

    print("\n=== Stage 3: Slide Paths ===")
    train_stage3(device, epochs=20)

    print("\n=== Stage 4: Break ===")
    train_stage4(device, epochs=20)

    print("\n=== Stage 5: Ex ===")
    train_stage5(device, epochs=20)

    print("\nAll stages complete!")
