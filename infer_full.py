"""
infer_full.py — 5-Stage 完整推理流水线

Stage1: 音频+节拍 → 谱面骨架
Stage2: +骨架 → hold 持续时间 (自回归)
Stage3: +骨架 → slide 路径 (自回归)
Stage4: +骨架 → break 标记 (双向)
Stage5: +骨架 → ex 标记 (双向, 仅DX)

输出: 完整 simai 谱面文件
"""

import json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models.common import StageConfig
from models.stage1_chart import Stage1ChartModel
from models.stage2_hold import Stage2HoldModel
from models.stage3_slide import Stage3SlideModel
from models.stage4_break import Stage4BreakModel
from models.stage5_ex import Stage5ExModel
from SimaiToken import SimaiToken, SimaiTokenType, _token_to_simai_note as note_to_simai


@torch.no_grad()
def infer_full(
    mp3_path: str,
    output_path: str,
    difficulty: str = "Master",
    level: float = 12.0,
    designer: str = "AI",
    data_dir: str = "preprocessed",
    ckpt_dir: str = "checkpoints",
    device: str = "cpu",
):
    DIFF_MAP = {"Easy": 1, "Basic": 2, "Advanced": 3, "Expert": 4,
                "Master": 5, "Re:Master": 6, "UTAGE": 7}
    DIFF_ID = {"Easy": 0, "Basic": 1, "Advanced": 2, "Expert": 3,
               "Master": 4, "Re:Master": 5, "UTAGE": 6}
    diff_num = DIFF_MAP.get(difficulty, 5)
    diff_id = DIFF_ID.get(difficulty, 4)

    # ── 加载 vocab ──
    with open(f"{data_dir}/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

    # ── 1. 音频 + 节拍 ──
    from AudioTokenizer import AudioTokenizer
    from BeatTokenizer import BeatTokenizer

    print("Encoding audio...")
    at = AudioTokenizer(num_codebooks=4)
    ad = at.encode_file(mp3_path)
    bt = BeatTokenizer(method="librosa", target_bpm=None, quantize_beats=True)
    bl = bt.analyse(mp3_path)

    fr = ad.frame_rate
    nf = ad.num_frames
    bpm = bl.bpm
    subdiv = 4
    measure_dur = subdiv * 60.0 / bpm

    audio = torch.from_numpy(ad.tokens).unsqueeze(0).long().to(device)

    beat_s = np.zeros((nf, 2), dtype=np.float32)
    for b in bl.beats:
        fi = round(b.time * fr)
        if 0 <= fi < nf:
            beat_s[fi, 0] = max(beat_s[fi, 0], 0.5)
            if b.is_downbeat:
                beat_s[fi, 1] = 1.0
    beat = torch.from_numpy((beat_s > 0.3).astype(np.float32)).unsqueeze(0).to(device)

    diff_t = torch.tensor([diff_id], device=device)
    lvl_t = torch.tensor([level], device=device)
    tags_t = torch.full((1, 32), -1, dtype=torch.long, device=device)

    ckpt_root = Path(ckpt_dir)

    def load_stage_checkpoint(stage: int):
        candidates = [
            ckpt_root / f"stage{stage}_best.pt",
            Path(data_dir) / f"stage{stage}_best.pt",
            Path(data_dir) / f"stage{stage}.pt",
        ]
        for path in candidates:
            if path.exists():
                ckpt = torch.load(path, map_location=device, weights_only=False)
                cfg = ckpt.get("config", ckpt.get("cfg"))
                state = ckpt.get("model_state_dict", ckpt.get("model"))
                if cfg is None or state is None:
                    raise KeyError(f"Checkpoint {path} missing config/cfg or model_state_dict/model")
                return ckpt, cfg, state
        raise FileNotFoundError(f"No checkpoint found for stage {stage}: {candidates}")

    # ── Stage 1: 谱面骨架 ──
    print("Stage 1: chart skeleton...")
    ckpt1, cfg1, state1 = load_stage_checkpoint(1)
    m1 = Stage1ChartModel(cfg1).to(device).eval()
    m1.load_state_dict(state1)
    pred1 = m1.generate(audio, beat, diff_t, lvl_t, tags_t, temperature=0.8, top_k=50)
    chart = pred1  # (1, T)
    T = chart.shape[1]
    print(f"  frames={T}, notes={(chart>0).sum().item()}")

    # ── Stage 2: Hold 持续时间 ──
    print("Stage 2: hold durations...")
    ckpt2, cfg2, state2 = load_stage_checkpoint(2)
    m2 = Stage2HoldModel(cfg2).to(device).eval()
    m2.load_state_dict(state2)
    hold_ids = {tid for tok, tid in vocab.items() if tok.startswith("hold")}
    hold_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
    for hid in hold_ids:
        hold_mask = hold_mask | (chart == hid)
    dur_pred = m2.generate(chart, audio, beat, diff_t, lvl_t, tags_t, hold_mask)
    hold_durs = dur_pred[0].cpu().numpy()
    if hold_durs.ndim == 2:
        hold_durs = hold_durs[:, 0]  # legacy renderer uses one hold dur per frame
    print(f"  holds with duration: {(hold_durs > 0).sum()}")

    # ── Stage 3: Slide 路径 ──
    print("Stage 3: slide paths...")
    ckpt3, cfg3, state3 = load_stage_checkpoint(3)
    m3 = Stage3SlideModel(cfg3).to(device).eval()
    m3.load_state_dict(state3)
    slide_vocab_path = Path(data_dir) / "slide_vocab.json"
    if "slide_vocab" in ckpt3:
        slide_vocab = ckpt3["slide_vocab"]
    elif slide_vocab_path.exists():
        slide_vocab = json.loads(slide_vocab_path.read_text("utf-8"))
    else:
        slide_vocab = {"<PAD>": 0}
    slide_vocab_inv = {v: k for k, v in slide_vocab.items()}
    # 简化: 使用模型输出的第一个路径 token
    out3 = m3(chart, audio, beat, diff_t, lvl_t, tags_t)
    slide_paths = out3["logits"].argmax(dim=-1)[0].cpu().numpy()  # (T,)
    if slide_paths.ndim == 2:
        slide_paths = slide_paths[:, 0]  # legacy renderer uses one slide path per frame
    print(f"  slides with path: {(slide_paths > 0).sum()}")

    # ── Stage 4: Break ──
    print("Stage 4: break flags...")
    ckpt4, cfg4, state4 = load_stage_checkpoint(4)
    m4 = Stage4BreakModel(cfg4).to(device).eval()
    m4.load_state_dict(state4)
    note_mask = (chart > 0).bool()
    break_pred = m4.predict(chart, audio, beat, diff_t, lvl_t, tags_t)[0].cpu().numpy()
    if break_pred.ndim == 2:
        break_pred = break_pred[:, 0]
    print(f"  break notes: {int(break_pred.sum())}")

    # ── Stage 5: Ex ──
    print("Stage 5: ex flags...")
    ckpt5, cfg5, state5 = load_stage_checkpoint(5)
    m5 = Stage5ExModel(cfg5).to(device).eval()
    m5.load_state_dict(state5)
    ex_pred = m5.predict(chart, audio, beat, diff_t, lvl_t, tags_t)[0].cpu().numpy()
    if ex_pred.ndim == 2:
        ex_pred = ex_pred[:, 0]
    print(f"  ex notes: {int(ex_pred.sum())}")

    # ── 构建 simai 输出 ──
    print("Building simai output...")
    chart_np = chart[0].cpu().numpy()
    measures: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))

    for f in range(T):
        tid = int(chart_np[f])
        if tid <= 0:
            continue
        tok_str = id_to_token.get(tid)
        if tok_str is None:
            continue

        st = SimaiToken.from_string(tok_str)
        if st is None:
            continue

        # 注入 hold 持续时间
        if st.token_type == SimaiTokenType.HOLD and hold_durs[f] > 0:
            dur_bin = int(hold_durs[f])
            secs = 2.0 ** (dur_bin - 5)  # 粗略反量化
            dur_str = f"{max(1, round(secs * 4))}:1"
            st.params["dur"] = dur_str

        # 注入 slide 路径 (过滤 <EOS>/<PAD>)
        if st.token_type == SimaiTokenType.SLIDE and slide_paths[f] > 0:
            pid = int(slide_paths[f])
            seg = slide_vocab_inv.get(pid, "")
            if seg and seg not in ("<PAD>", "<EOS>"):
                st.params["path"] = seg

        # 注入 break/ex
        if break_pred[f]:
            st.params["break"] = ""
        if ex_pred[f]:
            st.params["ex"] = ""

        simai_note = note_to_simai(st)

        t_sec = f / fr
        m = int(t_sec / measure_dur)
        beat_in_m = (t_sec % measure_dur) / measure_dur
        bi = min(round(beat_in_m * subdiv), subdiv - 1)
        measures[m][bi].append(simai_note)

    # 写文件
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
    total_notes = sum(len(b) for b in measures.values() for b in b.values())
    print(f"\nDone! {total_notes} notes, {max_m+1} measures → {output_path}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    infer_full(
        mp3_path="samples/人是猫/track.mp3",
        output_path="samples/人是猫/maidata_full.txt",
        difficulty="Master", level=12.0, designer="AI",
        device=device,
    )
