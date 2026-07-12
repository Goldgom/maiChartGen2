"""
webui.py — maiChartGen3 Gradio WebUI

提供可视化界面，手动调整生成参数、标签等，一键推理生成谱面。
"""

import json
from collections import defaultdict
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from models.stage1_chart import Stage1ChartModel
from models.stage2_hold import Stage2HoldModel
from models.stage3_slide import Stage3SlideModel
from SimaiToken import SimaiToken, SimaiTokenType, _token_to_simai_note as note_to_simai

# ═══════════════════════════════════════════════════════════
# 全局常量
# ═══════════════════════════════════════════════════════════
DATA_DIR = "preprocessed"
CKPT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DIFFICULTIES = ["Easy", "Basic", "Advanced", "Expert", "Master", "Re:Master", "UTAGE"]
DIFF_MAP = {d: i + 1 for i, d in enumerate(DIFFICULTIES)}
DIFF_ID = {d: i for i, d in enumerate(DIFFICULTIES)}

# 加载词表
with open(f"{DATA_DIR}/vocab.json", "r", encoding="utf-8") as f:
    VOCAB = json.load(f)
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

with open(f"{DATA_DIR}/tag_vocab.json", "r", encoding="utf-8") as f:
    TAG_VOCAB = json.load(f)  # {tag_string: id}

# 加载 slide vocab
slide_vocab_path = Path(DATA_DIR) / "slide_vocab.json"
if slide_vocab_path.exists():
    SLIDE_VOCAB = json.loads(slide_vocab_path.read_text("utf-8"))
else:
    SLIDE_VOCAB = {"<PAD>": 0}
SLIDE_VOCAB_INV = {v: k for k, v in SLIDE_VOCAB.items()}

# 加载 path → best_timing 映射 (从训练数据统计)
_timing_map_path = Path(DATA_DIR) / "slide_path_timing_map.json"
if _timing_map_path.exists():
    PATH_BEST_TIMING = json.loads(_timing_map_path.read_text("utf-8"))
else:
    PATH_BEST_TIMING = {}
print(f"Loaded path→timing map: {len(PATH_BEST_TIMING)} paths")

# 提取 collection 标签
COLLECTION_TAGS = sorted([
    k.replace("collection:", "") for k in TAG_VOCAB
    if k.startswith("collection:")
], key=lambda x: (0 if x == "Original" else 1, x))

# 提取 designer 标签
DESIGNER_TAGS = sorted([
    k.replace("designer:", "") for k in TAG_VOCAB
    if k.startswith("designer:")
])

# ═══════════════════════════════════════════════════════════
# Token 类型掩码 (用于生成偏置)
# ═══════════════════════════════════════════════════════════
VOCAB_SIZE = len(VOCAB) + 1  # +1 for implicit id=0 (empty)
EMPTY_ID = 0

# 预计算各类 token 的 id 集合
_TAP_IDS = set(v for k, v in VOCAB.items() if k.startswith("tap"))
_HOLD_IDS = set(v for k, v in VOCAB.items() if k.startswith("hold"))
_SLIDE_IDS = set(v for k, v in VOCAB.items() if k.startswith("slide"))
_TOUCH_IDS = set(v for k, v in VOCAB.items() if k.startswith("touch"))
_NOTE_IDS = _TAP_IDS | _HOLD_IDS | _SLIDE_IDS | _TOUCH_IDS

# 构建偏置掩码张量 (device 无关, 使用时 .to(device))
def _build_mask(id_set: set, vocab_size: int) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.float32)
    for i in id_set:
        mask[i] = 1.0
    return mask

BIAS_EMPTY_MASK = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
BIAS_EMPTY_MASK[EMPTY_ID] = 1.0
BIAS_NOTE_MASK = _build_mask(_NOTE_IDS, VOCAB_SIZE)
BIAS_TAP_MASK = _build_mask(_TAP_IDS, VOCAB_SIZE)
BIAS_HOLD_MASK = _build_mask(_HOLD_IDS, VOCAB_SIZE)
BIAS_SLIDE_MASK = _build_mask(_SLIDE_IDS, VOCAB_SIZE)
BIAS_TOUCH_MASK = _build_mask(_TOUCH_IDS, VOCAB_SIZE)

# ═══════════════════════════════════════════════════════════
# 模型加载 (延迟加载)
# ═══════════════════════════════════════════════════════════
_models_cache: dict = {}


def _load_model(stage: int):
    """加载指定 stage 的模型"""
    if stage in _models_cache:
        return _models_cache[stage]

    ckpt_path = Path(CKPT_DIR) / f"stage{stage}_best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(DATA_DIR) / f"stage{stage}_best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(DATA_DIR) / f"stage{stage}.pt"

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", ckpt.get("cfg"))
    state = ckpt.get("model_state_dict", ckpt.get("model"))

    if stage == 1:
        model = Stage1ChartModel(cfg).to(DEVICE).eval()
    elif stage == 2:
        model = Stage2HoldModel(cfg).to(DEVICE).eval()
    elif stage == 3:
        model = Stage3SlideModel(cfg).to(DEVICE).eval()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    model.load_state_dict(state)
    _models_cache[stage] = model
    return model


# ═══════════════════════════════════════════════════════════
# 带偏置的采样函数
# ═══════════════════════════════════════════════════════════

def _biased_sample(
    logits: torch.Tensor,      # (B, T, V)
    temperature: float,
    top_k: int,
    density: float,            # 密度偏置 (-5~+5)
    tap_bias: float,           # Tap 偏置
    hold_bias: float,          # Hold 偏置
    slide_bias: float,         # Slide 偏置
    touch_bias: float,         # Touch 偏置
) -> torch.Tensor:
    """对 logits 施加类型偏置后采样"""
    device = logits.device

    # 构建偏置向量: density 提升所有音符 / 降低空位
    bias = (BIAS_NOTE_MASK.to(device) - BIAS_EMPTY_MASK.to(device)) * density
    bias += BIAS_TAP_MASK.to(device) * tap_bias
    bias += BIAS_HOLD_MASK.to(device) * hold_bias
    bias += BIAS_SLIDE_MASK.to(device) * slide_bias
    bias += BIAS_TOUCH_MASK.to(device) * touch_bias

    logits = logits + bias.view(1, 1, -1)

    # Temperature
    if temperature > 0:
        logits = logits / temperature

    # Top-K
    if top_k > 0 and top_k < logits.shape[-1]:
        topk_vals, _ = torch.topk(logits, top_k, dim=-1)
        min_topk = topk_vals[:, :, -1:]
        logits = torch.where(logits < min_topk,
                             torch.full_like(logits, float("-inf")),
                             logits)

    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(
        probs.reshape(-1, logits.shape[-1]), 1
    ).reshape(logits.shape[0], -1)

    return tokens


# ═══════════════════════════════════════════════════════════
# Slide 路径校验
# ═══════════════════════════════════════════════════════════

def _validate_slide_path(start_pos: str, path_str: str) -> bool:
    """检查 slide 路径是否合法 (不产生相邻位置)"""
    import re
    try:
        start = int(start_pos)
    except ValueError:
        return True  # 非数字位置 (如触摸), 跳过检查

    # 提取路径的第一个目标位置
    # 匹配模式: -X, >X, <X, ^X, vX, wX, pX, qX, sX, zX, VXX(取第二个数字)
    m = re.match(r'[><^v\-wpqsz]([1-8])', path_str)
    if not m:
        # VXX 格式: V28 → 目标 = 8
        m = re.match(r'V[1-8]([1-8])', path_str)
    if m:
        target = int(m.group(1))
    else:
        return True  # 无法解析, 保守放行

    # 相邻检查: |a-b| == 1 或 == 7 (环形)
    diff = abs(start - target)
    if diff == 1 or diff == 7:
        return False  # 相邻, 无效

    return True


# ═══════════════════════════════════════════════════════════
# 推理核心
# ═══════════════════════════════════════════════════════════
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def generate_chart(
    mp3_path: str,
    difficulty: str,
    level: float,
    designer: str,
    collections: list[str],
    temperature: float,
    top_k: int,
    bpm_override: float,
    density: float,
    tap_bias: float,
    hold_bias: float,
    slide_bias: float,
    touch_bias: float,
    progress=gr.Progress(),
) -> tuple[str, str]:
    """核心推理函数，返回 (simai文本, 状态信息)"""
    from AudioTokenizer import AudioTokenizer
    from BeatTokenizer import BeatTokenizer

    diff_num = DIFF_MAP.get(difficulty, 5)
    diff_id = DIFF_ID.get(difficulty, 4)

    # ── 1. 音频编码 ──
    progress(0.05, desc="正在编码音频...")
    at = AudioTokenizer(num_codebooks=8)
    ad = at.encode_file(mp3_path)
    bt = BeatTokenizer(method="librosa", target_bpm=None, quantize_beats=True)
    bl = bt.analyse(mp3_path)

    fr = ad.frame_rate
    nf = ad.num_frames
    bpm = bl.bpm if bpm_override <= 0 else bpm_override
    subdiv = 32
    measure_dur = 4 * 60.0 / bpm

    audio = torch.from_numpy(ad.tokens).unsqueeze(0).long().to(DEVICE)

    # 节拍信号
    beat_s = np.zeros((nf, 2), dtype=np.float32)
    for b in bl.beats:
        fi = round(b.time * fr)
        if 0 <= fi < nf:
            beat_s[fi, 0] = max(beat_s[fi, 0], 0.5)
            if b.is_downbeat:
                beat_s[fi, 1] = 1.0
    beat = torch.from_numpy((beat_s > 0.3).astype(np.float32)).unsqueeze(0).to(DEVICE)

    diff_t = torch.tensor([diff_id], device=DEVICE)
    lvl_t = torch.tensor([level], device=DEVICE)

    # 构建 tag tensor (最多32个标签)
    tag_ids = [-1] * 32
    tag_idx = 0
    # difficulty tag (始终添加)
    diff_tag = f"difficulty:{difficulty}"
    if diff_tag in TAG_VOCAB and tag_idx < 32:
        tag_ids[tag_idx] = TAG_VOCAB[diff_tag]
        tag_idx += 1
    # collection tags (可多选)
    if collections:
        for col in collections:
            if not col or col == "无" or tag_idx >= 32:
                continue
            col_tag = f"collection:{col}"
            if col_tag in TAG_VOCAB:
                tag_ids[tag_idx] = TAG_VOCAB[col_tag]
                tag_idx += 1
    # designer tag
    if designer and designer != "AI" and tag_idx < 32:
        des_tag = f"designer:{designer}"
        if des_tag in TAG_VOCAB:
            tag_ids[tag_idx] = TAG_VOCAB[des_tag]
            tag_idx += 1
    tags_t = torch.tensor([tag_ids], dtype=torch.long, device=DEVICE)

    # ── Stage 1: 谱面骨架 (带偏置采样) ──
    progress(0.15, desc="Stage 1: 生成谱面骨架...")
    m1 = _load_model(1)
    result1 = m1.forward(audio, beat, diff_t, lvl_t, tags_t)
    logits1 = result1["logits"]  # (B, T, V)
    chart = _biased_sample(logits1, temperature, top_k,
                           density, tap_bias, hold_bias, slide_bias, touch_bias)
    T = chart.shape[1]

    hold_ids = {tid for tok, tid in VOCAB.items() if tok.startswith("hold")}

    # ── Stage 2: Hold 持续时间 ──
    progress(0.40, desc="Stage 2: 预测 Hold 持续时间...")
    m2 = _load_model(2)
    hold_mask = torch.zeros(1, T, dtype=torch.bool, device=DEVICE)
    for hid in hold_ids:
        hold_mask = hold_mask | (chart == hid)
    dur_pred = m2.generate(chart, audio, beat, diff_t, lvl_t, tags_t, hold_mask,
                           temperature=temperature)
    hold_durs = dur_pred[0].cpu().numpy()
    if hold_durs.ndim == 2:
        hold_durs = hold_durs[:, 0]

    # ── Stage 3: Slide 路径 (带采样, 支持多段路径) ──
    progress(0.65, desc="Stage 3: 预测 Slide 路径...")
    m3 = _load_model(3)
    out3 = m3(chart, audio, beat, diff_t, lvl_t, tags_t)

    # slide_logits: (B, T, S, V) 其中 S=max_slide_slots, V=slide_vocab_size
    slide_logits = out3["logits"][0]  # (T, S, V)
    S = slide_logits.shape[1]  # max_slide_slots (通常 8)

    # 对每个 slot 独立采样 (temperature + top_k)
    slide_temp = temperature * 0.7  # slide 略低温度, 更稳定
    slide_topk = max(10, top_k // 2)

    # 采样: (T, S) → 每帧每 slot 一个 path id
    if slide_temp > 0:
        sl = slide_logits / slide_temp
    else:
        sl = slide_logits

    if slide_topk > 0 and slide_topk < sl.shape[-1]:
        topk_vals, _ = torch.topk(sl, slide_topk, dim=-1)
        min_topk = topk_vals[:, :, -1:]
        sl = torch.where(sl < min_topk, torch.full_like(sl, float("-inf")), sl)

    probs = F.softmax(sl, dim=-1)  # (T, S, V)
    flat_probs = probs.reshape(-1, sl.shape[-1])  # (T*S, V)
    slide_paths = torch.multinomial(flat_probs, 1).reshape(T, S).cpu().numpy()  # (T, S)

    # ── 构建 simai ──
    progress(0.80, desc="构建 simai 谱面...")
    chart_np = chart[0].cpu().numpy()
    measures: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))

    note_count = 0
    hold_count = 0
    slide_count = 0
    tap_count = 0

    for f in range(T):
        tid = int(chart_np[f])
        if tid <= 0:
            continue
        tok_str = ID_TO_TOKEN.get(tid)
        if tok_str is None:
            continue

        st = SimaiToken.from_string(tok_str)
        if st is None:
            continue

        # 注入 hold 持续时间
        if st.token_type == SimaiTokenType.HOLD:
            if hold_durs[f] > 0:
                dur_bin = int(hold_durs[f])
                secs = 2.0 ** (dur_bin - 5)
                dur_str = f"{max(1, round(secs * 4))}:1"
                st.params["dur"] = dur_str
            elif "dur" not in st.params or not st.params["dur"]:
                st.params["dur"] = "4:1"
            hold_count += 1

        # 注入 slide 路径 + 持续时间
        # 只取 slot 0: vocab 中已有 2004 个单 token 含多段路径 (如 -5*V26, >8*V28)
        # 多 slot 独立拼接会产生无效语法
        if st.token_type == SimaiTokenType.SLIDE:
            pid = int(slide_paths[f, 0])
            if pid > 1:  # <PAD>=0, <EOS>=1
                seg = SLIDE_VOCAB_INV.get(pid, "")
                if seg and seg not in ("<PAD>", "<EOS>"):
                    # 校验: 防止相邻位置 slide (如 1-2, 8-1)
                    if _validate_slide_path(st.position, seg):
                        st.params["path"] = seg
                    # 相邻路径被丢弃, 模型采样已有足够多样性替补
            # 数据驱动 timing: 查 path→best_timing 映射
            if "dur" not in st.params or not st.params["dur"]:
                path_key = st.params.get("path", "")
                if path_key and path_key in PATH_BEST_TIMING:
                    st.params["dur"] = PATH_BEST_TIMING[path_key]
                elif hold_durs[f] > 0:
                    dur_bin = int(hold_durs[f])
                    secs = 2.0 ** (dur_bin - 5)
                    dur_str = f"{max(1, round(secs * 4))}:1"
                    st.params["dur"] = dur_str
                else:
                    st.params["dur"] = "4:1"  # fallback

        if st.token_type == SimaiTokenType.TAP:
            tap_count += 1

        note_count += 1
        simai_note = note_to_simai(st)

        t_sec = f / fr
        m = int(t_sec / measure_dur)
        beat_in_m = (t_sec % measure_dur) / measure_dur
        bi = min(round(beat_in_m * subdiv), subdiv - 1)
        measures[m][bi].append(simai_note)

    # ── 写 simai 文件 ──
    title = Path(mp3_path).parent.name
    lines = [
        f"&title={title}",
        f"&artist={designer}",
        f"&wholebpm={bpm:.1f}",
        f"&lv_{diff_num}={level:.1f}",
        f"&des_{diff_num}={designer}",
        f"&inote_{diff_num}=",
    ]
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

    simai_text = "\n".join(lines)

    # 统计信息
    info = (
        f"✅ 生成完成！\n\n"
        f"📊 统计信息:\n"
        f"  - 总音符数: {note_count}\n"
        f"  - Tap: {tap_count} | Hold: {hold_count} | Slide: {slide_count}\n"
        f"  - 小节数: {max_m + 1} | 帧数: {T}\n"
        f"  - BPM: {bpm:.1f} | 难度: {difficulty} {level:.1f}\n"
        f"  - 设备: {DEVICE}\n"
        f"\n📝 谱面已保存"
    )

    progress(1.0, desc="完成！")
    return simai_text, info


# ═══════════════════════════════════════════════════════════
# Gradio 界面
# ═══════════════════════════════════════════════════════════

CUSTOM_CSS = """
.generate-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
                color: white !important; font-size: 18px !important; font-weight: bold !important; }
.simai-output textarea { font-family: 'Consolas', 'Courier New', monospace !important; 
                         font-size: 13px !important; line-height: 1.4 !important; }
"""


def build_ui():
    with gr.Blocks(title="maiChartGen3 — AI 谱面生成器") as demo:
        gr.Markdown("""
        # 🎹 maiChartGen3 — AI 谱面生成器
        
        上传 MP3 音频，AI 自动生成 maimai 谱面 (simai 格式)。
        支持 Master 13 难度，使用三阶段 Transformer 模型。
        """)

        with gr.Row():
            # ── 左栏: 输入参数 ──
            with gr.Column(scale=1):
                gr.Markdown("### 🎵 音频输入")
                audio_input = gr.File(
                    label="上传 MP3 文件",
                    file_types=[".mp3", ".wav", ".ogg"],
                    type="filepath",
                )

                gr.Markdown("### ⚙️ 谱面参数")
                with gr.Row():
                    difficulty = gr.Dropdown(
                        choices=DIFFICULTIES,
                        value="Master",
                        label="难度",
                    )
                    level = gr.Slider(
                        minimum=1.0, maximum=15.0, value=13.0, step=0.1,
                        label="等级",
                    )

                designer = gr.Textbox(
                    value="AI", label="谱面作者 (Designer)",
                    placeholder="输入作者名...",
                )

                collection = gr.Dropdown(
                    choices=COLLECTION_TAGS,
                    value=["Original"],
                    label="曲库标签 (Collection) — 可多选",
                    multiselect=True,
                    allow_custom_value=True,
                )

                gr.Markdown("### 🎛️ 生成控制")
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.8, step=0.05,
                        label="Temperature (温度)",
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=200, value=50, step=1,
                        label="Top-K 采样",
                    )

                bpm_override = gr.Number(
                    value=-1, label="BPM 覆盖 (-1=自动检测)",
                    precision=1,
                )

                gr.Markdown("### 🎯 物块偏置 (Bias)")
                gr.Markdown("*正值=更多, 负值=更少, 0=不偏置*")

                density = gr.Slider(
                    minimum=-5.0, maximum=5.0, value=0.0, step=0.5,
                    label="📊 整体密度 (Density)",
                )
                with gr.Row():
                    tap_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.5,
                        label="👆 Tap",
                    )
                    hold_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.5,
                        label="✋ Hold",
                    )
                with gr.Row():
                    slide_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.5,
                        label="⭐ Slide",
                    )
                    touch_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.5,
                        label="🖐 Touch",
                    )

                generate_btn = gr.Button(
                    "🚀 生成谱面",
                    variant="primary",
                    elem_classes=["generate-btn"],
                    size="lg",
                )

            # ── 右栏: 输出 ──
            with gr.Column(scale=2):
                gr.Markdown("### 📝 生成结果")
                status_output = gr.Textbox(
                    label="状态",
                    value="等待生成...",
                    lines=5,
                    interactive=False,
                )
                simai_output = gr.Textbox(
                    label="Simai 谱面",
                    lines=25,
                    max_lines=40,
                    interactive=False,
                    elem_classes=["simai-output"],
                    placeholder="生成的谱面将显示在这里...",
                )

                with gr.Row():
                    download_btn = gr.DownloadButton(
                        label="📥 下载 maidata.txt",
                        value=str(OUTPUT_DIR / "generated.txt"),
                        visible=False,
                    )
                    copy_btn = gr.Button("📋 复制到剪贴板", size="sm")

        # ── 事件绑定 ──
        def on_generate(audio_file, diff, lvl, des, cols, temp, topk, bpm_ov,
                        dens, tb, hb, sb, ttb):
            if audio_file is None:
                return "", "❌ 请先上传 MP3 文件！", gr.DownloadButton(visible=False)

            mp3_path = audio_file if isinstance(audio_file, str) else audio_file.name
            try:
                simai_text, info = generate_chart(
                    mp3_path, diff, lvl, des, cols, temp, topk, bpm_ov,
                    dens, tb, hb, sb, ttb,
                )
                # 写入文件供下载
                out_file = OUTPUT_DIR / "generated.txt"
                out_file.write_text(simai_text, encoding="utf-8")
                return simai_text, info, gr.DownloadButton(visible=True, value=str(out_file))
            except Exception as e:
                import traceback
                traceback.print_exc()
                return "", f"❌ 生成失败: {str(e)}", gr.DownloadButton(visible=False)

        generate_btn.click(
            on_generate,
            inputs=[audio_input, difficulty, level, designer, collection,
                    temperature, top_k, bpm_override,
                    density, tap_bias, hold_bias, slide_bias, touch_bias],
            outputs=[simai_output, status_output, download_btn],
        )

        # 示例
        gr.Markdown("""
        ---
        ### 💡 参数说明
        
        | 参数 | 说明 | 推荐值 |
        |------|------|--------|
        | Temperature | 越高越有创意但可能混乱，越低越保守但可能单调 | Master: 0.7~0.9 |
        | Top-K | 每步只从概率最高的 K 个 token 中采样 | 30~80 |
        | BPM 覆盖 | -1 表示自动检测；手动设置可覆盖检测结果 | -1 |
        | 整体密度 | 正值 → 更密集(更多音符)；负值 → 更稀疏 | 0~2 |
        | Tap/Hold/Slide/Touch | 正值 → 该类型更多；负值 → 更少 | 0 |
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
