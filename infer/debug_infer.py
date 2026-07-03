#!/usr/bin/env python3
"""
maiG_v2 Full Inference Debug Script
====================================
对 test/test3 进行完整 5 阶段推理，保存所有中间输出用于 debug。

Pipeline: audio → stage1(生成主谱) → break(分类) → slide(滑条) → spike(星星) → touch(触摸)
"""
from __future__ import annotations

import sys
import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    # ============================================================
    # 配置
    # ============================================================
    TEST_DIR = PROJECT_ROOT / "test" / "test3"
    AUDIO_PATH = TEST_DIR / "track.mp3"
    DEBUG_OUT = TEST_DIR / "debug_outputs"
    CHECKPOINT_DIR = Path("/data/maiG_v2/runs/rotating_4090")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*60}")
    print(f"  maiG_v2 Full Inference Debug")
    print(f"  Test: {TEST_DIR}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {DEBUG_OUT}")
    print(f"{'='*60}")

    # 创建输出目录
    DEBUG_OUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DEBUG_OUT / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Step 1: BPM 检测
    # ============================================================
    print("\n[Step 1] BPM Detection...")
    from utils.bpm_detector import BPMDetector

    detector = BPMDetector()
    bpm_result = detector.detect(str(AUDIO_PATH))
    bpm = bpm_result.bpm
    print(f"  → BPM: {bpm:.1f} (confidence: {bpm_result.confidence:.2f}, method: {bpm_result.method})")

    # 保存 BPM 结果
    with open(run_dir / "01_bpm_result.json", "w") as f:
        json.dump({
            "bpm": bpm,
            "confidence": bpm_result.confidence,
            "method": bpm_result.method,
            "candidates": bpm_result.candidates,
        }, f, indent=2, default=str)

    # ============================================================
    # Step 2: 音频特征提取
    # ============================================================
    print("\n[Step 2] Audio Feature Extraction...")
    from utils.audio_features import extract_features

    features = extract_features(str(AUDIO_PATH), bpm=bpm, subdiv=64)
    onset = torch.from_numpy(features.onset)
    chroma = torch.from_numpy(features.chroma)
    centroid = torch.from_numpy(features.centroid)
    print(f"  → Onset: {onset.shape}, Chroma: {chroma.shape}, Centroid: {centroid.shape}")
    print(f"  → Num slots: {features.num_slots}")

    # 保存原始特征
    np.save(run_dir / "02_onset.npy", features.onset)
    np.save(run_dir / "02_chroma.npy", features.chroma)
    np.save(run_dir / "02_centroid.npy", features.centroid)

    # ============================================================
    # Step 3: Stage 1 — 主谱生成
    # ============================================================
    print("\n[Step 3] Stage 1 — Chart Generation...")

    from models.stage1 import MaiGenerator

    # 构建模型配置 (与 rotating_4090.yaml 一致)
    stage1 = MaiGenerator(
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        vocab_size=161512,
        subdiv=64,
        beats_per_bar=4,
        dropout=0.1,
        audio_stream_layers=4,
        audio_stream_heads=12,
        use_checkpoint=False,
        global_stride=8,
        local_window_s=5.0,
        local_slots_per_sec=184,
        local_dilation_base=4,
        max_spectral_len=16384,
        use_spectral_sliding_window=False,
    ).to(DEVICE)

    # 加载权重
    ckpt_path = CHECKPOINT_DIR / "stage1" / "last.pt"
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        stage1.load_state_dict(ckpt, strict=False)
    stage1.eval()

    # 推理参数
    LEVEL = 10.0   # 难度等级
    GENRE = 0      # 流派

    max_gen_steps = 16384  # 覆盖整首歌 (audio_len * 0.55 + margin)
    print(f"  Generating (bpm={bpm:.1f}, level={LEVEL}, max_steps={max_gen_steps}, window=1024)...")
    with torch.no_grad():
        generated_tokens = stage1.generate(
            onset=onset,
            chroma=chroma,
            centroid=centroid,
            bpm=bpm,
            level=LEVEL,
            genre=GENRE,
            max_steps=max_gen_steps,
            temperature=1.0,
            top_k=50,
            window_size=1024,
        )

    print(f"  → Generated {len(generated_tokens)} tokens")
    print(f"  → First 50: {generated_tokens[:50]}")

    # 保存 stage1 输出
    with open(run_dir / "03_stage1_tokens.json", "w") as f:
        json.dump({"num_tokens": len(generated_tokens), "tokens": generated_tokens}, f)

    # 解码 stage1 tokens → 合法 maidata.txt
    from Tokenizer.MaiChartTokenizer import MaiChartTokenizer
    tokenizer = MaiChartTokenizer()
    try:
        raw_text = tokenizer.decode(generated_tokens)
        print(f"  → Decoded raw text ({len(raw_text)} chars)")
        print(raw_text[:300])

        # 构建合法 maidata.txt
        # 去掉 BOS/EOS 占位
        clean_notes = raw_text.strip()
        if clean_notes.endswith(",EOS"):
            clean_notes = clean_notes[:-4]
        if clean_notes.startswith("BOS,"):
            clean_notes = clean_notes[4:]

        # 每 64 个逗号 (1拍) 换行，方便阅读
        parts = clean_notes.split(",")
        lines = []
        for i in range(0, len(parts), 64):
            lines.append(",".join(parts[i:i+64]))
        formatted_notes = ",\n".join(lines)

        maidata = f"""&title=Generated Chart
&wholebpm={bpm:.1f}
&artist=MaiGenerator
&lv_4=13.0
&inote_4=
({bpm:.0f}){{4}}
{formatted_notes}
E"""

        with open(run_dir / "03_stage1_decoded.txt", "w") as f:
            f.write(raw_text)

        with open(run_dir / "03_maidata.txt", "w") as f:
            f.write(maidata)

        print(f"  → maidata.txt written ({len(maidata)} chars)")
        print(maidata[:500])
    except Exception as e:
        print(f"  → Decode failed: {e}")
        with open(run_dir / "03_stage1_decoded_error.txt", "w") as f:
            f.write(f"Decode error: {e}\nTokens: {generated_tokens}")

    # 截断到各模型 pos_embed 安全范围 (max_pos=16384)
    MAX_POS = 16380
    if len(generated_tokens) > MAX_POS:
        print(f"  → Truncating tokens from {len(generated_tokens)} to {MAX_POS}")
        generated_tokens = generated_tokens[:MAX_POS]
        if generated_tokens[-1] != 2:  # EOS
            generated_tokens.append(2)

    # ============================================================
    # Step 4: Stage 1 Hidden States (滑动窗口分块提取避免 OOM)
    # ============================================================
    print("\n[Step 4] Stage 1 Hidden States Extraction (windowed)...")

    tokens_t = torch.tensor([generated_tokens], device=DEVICE)
    with torch.no_grad():
        inp = tokens_t[:, :-1]
        T_in = inp.size(1)

        bpm_t = torch.tensor([[bpm]], device=DEVICE, dtype=torch.float32)
        level_t = torch.tensor([[LEVEL]], device=DEVICE, dtype=torch.float32)
        genre_t = torch.tensor([[GENRE]], device=DEVICE, dtype=torch.float32)
        cond_in = torch.cat([bpm_t, level_t, genre_t], dim=-1)
        cond = stage1.cond_embed(cond_in)

        onset_d = onset.unsqueeze(0).to(DEVICE)
        chroma_d = chroma.unsqueeze(0).to(DEVICE)
        centroid_d = centroid.unsqueeze(0).to(DEVICE)

        audio_pack = stage1.audio(onset_d, chroma_d, centroid_d)
        full_audio_memory = audio_pack.fused_memory  # [1, T_audio, D]

        # 滑动窗口提取 hidden states
        WINDOW = 2048
        STRIDE = 1024
        from models.stage1 import compute_relative_distances

        all_hidden = []
        for win_start in range(0, T_in, STRIDE):
            win_end = min(win_start + WINDOW, T_in)
            if win_end <= win_start:
                break
            if win_start > 0 and win_end == T_in and win_end - win_start < STRIDE:
                # 最后一段太短，合并到前一段
                break

            win_inp = inp[:, win_start:win_end]
            win_len = win_inp.size(1)
            win_dist = compute_relative_distances(win_inp)
            win_pos = torch.arange(win_start, win_end, device=DEVICE).unsqueeze(0)
            win_x = stage1.token_embed(win_inp) + stage1.pos_embed(win_pos) + stage1.timing_embed(win_dist)

            mem_end = min(full_audio_memory.size(1), win_end)
            mem_start = max(0, win_start)
            mem = full_audio_memory[:, mem_start:mem_end, :]
            if mem.size(1) < win_len:
                mem = torch.cat([mem, mem[:, -1:, :].expand(1, win_len - mem.size(1), -1)], dim=1)
            mem = mem[:, :win_len, :]

            mask = torch.triu(torch.full((win_len, win_len), float("-inf"), device=DEVICE), diagonal=1)
            win_x = stage1._run_decoder(win_x, mem, cond)

            # 只保留非重叠部分 (第一个窗口全部保留)
            if win_start == 0:
                all_hidden.append(win_x.cpu())
            else:
                overlap = STRIDE  # 每个窗口前半部分与上一窗口重叠
                all_hidden.append(win_x[:, overlap:, :].cpu())

            print(f"    [hidden] window {win_start}-{win_end}, total hidden so far: {sum(h.size(1) for h in all_hidden)}", flush=True)

        stage1_hidden = torch.cat(all_hidden, dim=1)  # [1, T_hidden, 768]

    print(f"  → Stage1 hidden: {stage1_hidden.shape}")
    torch.save(stage1_hidden, run_dir / "04_stage1_hidden.pt")

    # ============================================================
    # Step 5: Break Stage — 区分 TAP/BREAK
    # ============================================================
    print("\n[Step 5] Break Stage — Classifying TAP vs BREAK...")

    from models.break_stage import BreakClassifier

    break_model = BreakClassifier(
        hidden_dim=384,
        num_layers=4,
        num_heads=6,
        num_positions=8,
        vocab_size=161512,
        stage1_dim=768,
    ).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / "break" / "last.pt"
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        break_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        break_model.load_state_dict(ckpt, strict=False)
    break_model.eval()

    with torch.no_grad():
        break_logits = break_model(
            tokens_t.to(DEVICE),
            stage1_hidden.to(DEVICE),
        )  # [1, T, 8, 2]

    break_preds = break_logits.argmax(dim=-1).cpu()  # [1, T, 8]
    print(f"  → Break logits: {break_logits.shape}")
    print(f"  → Break predictions (first 10 positions):\n{break_preds[0, :10, :]}")

    torch.save(break_logits.cpu(), run_dir / "05_break_logits.pt")
    torch.save(break_preds, run_dir / "05_break_preds.pt")

    # ============================================================
    # Step 6: Slide Stage — 滑条路径生成
    # ============================================================
    print("\n[Step 6] Slide Stage — Generating Slide Paths...")

    from models.slide_stage import SlidePathGenerator

    slide_model = SlidePathGenerator(
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        stage1_dim=768,
    ).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / "slide" / "last.pt"
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        slide_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        slide_model.load_state_dict(ckpt, strict=False)
    slide_model.eval()

    # 找到 slide token 位置，使用 generate 生成每个 slide 的路径
    from Tokenizer.MaiChartTokenizer import (
        SLD_BASE, SLD_END_TOKEN_BASE, SLD_BEG, SLD_END,
        SLD_BEG_BASE, SLD_BEG_END, SLD_END_POS_BASE, SLD_END_POS_END,
        DUR, ID_TO_DUR_NUM, ID_TO_DUR_DEN,
    )

    gen_arr = torch.tensor(generated_tokens, device=DEVICE)

    # 解析 slide 起始/结束位置和时长
    slide_regions = []  # [(start_pos, end_pos, duration, token_idx_range)]
    i = 0
    while i < len(generated_tokens):
        tid = generated_tokens[i]
        # 检查是否是 EOS 开始的 slide
        if SLD_BEG_BASE <= tid < SLD_BEG_END:
            start_pos = tid - SLD_BEG_BASE + 1
            end_pos = None
            dur = None
            j = i + 1
            while j < len(generated_tokens) and j < i + 20:
                t2 = generated_tokens[j]
                if SLD_END_POS_BASE <= t2 < SLD_END_POS_END:
                    end_pos = t2 - SLD_END_POS_BASE + 1
                elif t2 == DUR and j + 2 < len(generated_tokens):
                    dur = (ID_TO_DUR_NUM.get(generated_tokens[j+1], 4),
                           ID_TO_DUR_DEN.get(generated_tokens[j+2], 4))
                    j += 2
                j += 1
                if end_pos is not None and dur is not None:
                    break
            if end_pos is not None and dur is not None:
                slide_regions.append({
                    "position": i,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "duration": dur,
                })
            i = j
        else:
            i += 1

    slide_outputs = []
    if slide_regions:
        print(f"  → Found {len(slide_regions)} slide regions")
        with torch.no_grad():
            for sr in slide_regions[:10]:  # 限制前10个
                try:
                    path = slide_model.generate(
                        audio_memory=stage1_hidden.to(DEVICE),
                        start_pos=sr["start_pos"],
                        end_pos=sr["end_pos"],
                        duration=sr["duration"],
                        max_steps=8,
                        temperature=0.8,
                        top_k=10,
                    )
                    slide_outputs.append({
                        **sr,
                        "generated_path": path,
                        "path_len": len(path),
                    })
                except Exception as e:
                    slide_outputs.append({**sr, "error": str(e)})

        with open(run_dir / "06_slide_outputs.json", "w") as f:
            json.dump(slide_outputs, f, indent=2, default=str)
        print(f"  → Slide results saved ({len(slide_outputs)} entries)")
    else:
        print("  → No slide regions found in generated tokens")
        with open(run_dir / "06_slide_outputs.json", "w") as f:
            json.dump({"message": "No slide regions found"}, f)

    # ============================================================
    # Step 7: Spike Stage — 星星分类
    # ============================================================
    print("\n[Step 7] Spike Stage — Classifying Touch Spikes...")

    from models.spike_stage import SpikeClassifier

    spike_model = SpikeClassifier(
        hidden_dim=384,
        num_layers=4,
        num_heads=6,
        num_zones=33,
        vocab_size=161512,
        stage1_dim=768,
    ).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / "spike" / "last.pt"
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        spike_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        spike_model.load_state_dict(ckpt, strict=False)
    spike_model.eval()

    with torch.no_grad():
        spike_logits = spike_model(
            tokens_t.to(DEVICE),
            stage1_hidden.to(DEVICE),
        )  # [1, T, 33, 2]

    spike_preds = spike_logits.argmax(dim=-1).cpu()  # [1, T, 33]
    print(f"  → Spike logits: {spike_logits.shape}")

    torch.save(spike_logits.cpu(), run_dir / "07_spike_logits.pt")
    torch.save(spike_preds, run_dir / "07_spike_preds.pt")

    # ============================================================
    # Step 8: Touch Stage — 触摸区域精细化
    # ============================================================
    print("\n[Step 8] Touch Stage — Refining Touch Regions...")

    from models.touch_stage import TouchRefiner

    touch_model = TouchRefiner(
        hidden_dim=768,
        num_layers=6,
        num_heads=12,
        num_zones=33,
        num_states=3,
        vocab_size=161512,
        dropout=0.1,
    ).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / "touch" / "last.pt"
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        touch_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        touch_model.load_state_dict(ckpt, strict=False)
    touch_model.eval()

    with torch.no_grad():
        touch_logits = touch_model(
            tokens_t.to(DEVICE),
            stage1_hidden.to(DEVICE),
            audio_pack.fused_memory.to(DEVICE),
        )  # [1, T, 33, 3]

    touch_preds = touch_logits.argmax(dim=-1).cpu()  # [1, T, 33]
    print(f"  → Touch logits: {touch_logits.shape}")

    torch.save(touch_logits.cpu(), run_dir / "08_touch_logits.pt")
    torch.save(touch_preds, run_dir / "08_touch_preds.pt")

    # ============================================================
    # 汇总输出
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  推理完成！所有输出保存到：")
    print(f"  {run_dir}")
    print(f"{'='*60}")

    # 生成汇总清单
    summary = {
        "timestamp": timestamp,
        "test_dir": str(TEST_DIR),
        "device": DEVICE,
        "bpm": bpm,
        "level": LEVEL,
        "genre": GENRE,
        "stage1_num_tokens": len(generated_tokens),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "output_files": sorted([f.name for f in run_dir.iterdir()]),
    }
    with open(run_dir / "00_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n输出文件清单：")
    for fn in sorted(run_dir.iterdir()):
        size = fn.stat().st_size
        if size > 1024 * 1024:
            print(f"  {fn.name:40s}  {size / 1024 / 1024:.1f} MB")
        elif size > 1024:
            print(f"  {fn.name:40s}  {size / 1024:.1f} KB")
        else:
            print(f"  {fn.name:40s}  {size} B")


if __name__ == "__main__":
    main()
