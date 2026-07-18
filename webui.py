"""
webui.py — maiChartGen3 Gradio WebUI

提供可视化界面，手动调整生成参数、标签等，一键推理生成谱面。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import re

from models.stage1_chart import Stage1ChartModel
from models.stage2_hold import Stage2HoldModel
from models.stage3_slide import Stage3SlideModel
from models.stage4_break import Stage4BreakModel
from models.stage5_ex import Stage5ExModel
from SimaiToken import SimaiToken, SimaiTokenType, _token_to_simai_note as note_to_simai
from Config import load_config

# ═══════════════════════════════════════════════════════════
# 加载配置
# ═══════════════════════════════════════════════════════════
_config_name = None
_cli_port = None
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        _config_name = sys.argv[i + 1]
    elif arg == "--port" and i + 1 < len(sys.argv):
        _cli_port = int(sys.argv[i + 1])

cfg = load_config(_config_name) if _config_name else load_config()
print(f"[webui] 已加载配置: {cfg.config_name}")

# ═══════════════════════════════════════════════════════════
# 全局常量 (从配置读取)
# ═══════════════════════════════════════════════════════════
DATA_DIR = cfg.preprocess.output_dir
VOCAB_DIR = Path(getattr(cfg.paths, "vocab_dir", "vocab"))
CKPT_DIR = cfg.paths.model_dir
OUTPUT_DIR = Path(cfg.paths.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设备: 优先使用配置中的设置，若配置为 cpu 但 cuda 可用则提示
_cfg_device = cfg.audio.device
if _cfg_device == "cuda" and not torch.cuda.is_available():
    print("[webui] 警告: 配置要求 cuda 但不可用，回退到 cpu")
    DEVICE = "cpu"
elif _cfg_device == "cpu" and torch.cuda.is_available():
    print("[webui] 提示: cuda 可用但配置指定为 cpu，使用 cpu")
    DEVICE = "cpu"
else:
    DEVICE = _cfg_device if _cfg_device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")

DIFFICULTIES = ["Easy", "Basic", "Advanced", "Expert", "Master", "Re:Master", "UTAGE"]
DIFF_MAP = {d: i + 1 for i, d in enumerate(DIFFICULTIES)}
DIFF_ID = {d: i for i, d in enumerate(DIFFICULTIES)}

# 加载词表
with open(VOCAB_DIR / "vocab.json", "r", encoding="utf-8") as f:
    VOCAB = json.load(f)
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

with open(VOCAB_DIR / "tag_vocab.json", "r", encoding="utf-8") as f:
    TAG_VOCAB = json.load(f)  # {tag_string: id}

# 加载 slide vocab
slide_vocab_path = VOCAB_DIR / "slide_vocab.json"
if slide_vocab_path.exists():
    SLIDE_VOCAB = json.loads(slide_vocab_path.read_text("utf-8"))
else:
    SLIDE_VOCAB = {"<PAD>": 0}
SLIDE_VOCAB_INV = {v: k for k, v in SLIDE_VOCAB.items()}


def _is_wifi_slide_vocab_token(token: str) -> bool:
    path = re.sub(r"\[[^\]]+\]$", "", token)
    return re.search(r"(^|\*)w[1-8]", path) is not None


_WIFI_SLIDE_IDS = {
    int(v) for k, v in SLIDE_VOCAB.items()
    if k not in ("<PAD>", "<EOS>") and _is_wifi_slide_vocab_token(k)
}

# 加载 path → best_timing 映射 (从训练数据统计)
_timing_map_path = VOCAB_DIR / "slide_path_timing_map.json"
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
_TOUCHHOLD_IDS = set(
    v for k, v in VOCAB.items()
    if re.match(r"^hold[A-E]\d*$", k)
)
_NOTE_IDS = _TAP_IDS | _HOLD_IDS | _SLIDE_IDS | _TOUCH_IDS


def _count_simultaneous_taps(token_str: str) -> int:
    tap_count = 0
    for part in token_str.split("+"):
        st = SimaiToken.from_string(part)
        if st is None:
            continue
        if st.token_type == SimaiTokenType.TAP:
            tap_count += len(st.position)
        elif st.token_type == SimaiTokenType.SLIDE:
            tap_count += 1
        elif st.token_type == SimaiTokenType.HOLD and re.fullmatch(r"\d+", st.position or ""):
            tap_count += 1
    return tap_count


def _has_touch_note(token_str: str) -> bool:
    for part in token_str.split("+"):
        st = SimaiToken.from_string(part)
        if st is None:
            continue
        if st.token_type == SimaiTokenType.TOUCH:
            return True
        if st.token_type == SimaiTokenType.HOLD and re.fullmatch(r"[A-E]\d*", st.position or ""):
            return True
    return False


_MULTI_TAP_IDS = {
    v for k, v in VOCAB.items()
    if _count_simultaneous_taps(k) >= 3
}
TAP_COUNTS_BY_ID = torch.zeros(VOCAB_SIZE, dtype=torch.long)
HAS_TOUCH_BY_ID = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
for _token, _tid in VOCAB.items():
    if 0 <= _tid < VOCAB_SIZE:
        TAP_COUNTS_BY_ID[_tid] = _count_simultaneous_taps(_token)
        HAS_TOUCH_BY_ID[_tid] = _has_touch_note(_token)

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
BIAS_TOUCHHOLD_MASK = _build_mask(_TOUCHHOLD_IDS, VOCAB_SIZE)
BIAS_MULTI_TAP_MASK = _build_mask(_MULTI_TAP_IDS, VOCAB_SIZE)
BIAS_WIFI_SLIDE_MASK = _build_mask(_WIFI_SLIDE_IDS, max(SLIDE_VOCAB_INV.keys(), default=0) + 1)
print(f"Loaded wifi slide paths: {len(_WIFI_SLIDE_IDS)}")


def _mask_like_logits(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    target = logits.shape[-1]
    mask = mask.to(logits.device)
    if mask.shape[0] == target:
        return mask
    if mask.shape[0] > target:
        return mask[:target]
    return F.pad(mask, (0, target - mask.shape[0]))


def _output_grid_index(frame_idx: int, frame_rate: float, measure_dur: float, subdiv: int) -> tuple[int, int]:
    t_sec = frame_idx / frame_rate
    measure = int(t_sec / measure_dur)
    beat_in_measure = (t_sec % measure_dur) / measure_dur
    beat_idx = min(round(beat_in_measure * subdiv), subdiv - 1)
    return measure, beat_idx


def _filter_multi_tap_chart(
    chart: torch.Tensor,
    frame_rate: float,
    measure_dur: float,
    subdiv: int,
) -> tuple[torch.Tensor, int, int]:
    """把会在最终输出格子形成三押及以上的 Stage1 token 直接置空。"""
    filtered_chart = chart.clone()
    chart_np = filtered_chart[0].detach().cpu().numpy()
    tap_counts_by_grid: dict[tuple[int, int], int] = defaultdict(int)
    touch_by_grid: dict[tuple[int, int], bool] = defaultdict(bool)
    filtered_tokens = 0
    filtered_taps = 0

    for frame_idx, tid in enumerate(chart_np):
        token_str = ID_TO_TOKEN.get(int(tid))
        if not token_str:
            continue

        tap_count = _count_simultaneous_taps(token_str)
        has_touch = _has_touch_note(token_str)
        if tap_count <= 0 and not has_touch:
            continue

        grid_idx = _output_grid_index(frame_idx, frame_rate, measure_dur, subdiv)
        final_taps = tap_counts_by_grid[grid_idx] + tap_count
        final_touch = touch_by_grid[grid_idx] or has_touch
        if final_taps >= 3 or (final_taps >= 2 and final_touch):
            filtered_chart[0, frame_idx] = EMPTY_ID
            filtered_tokens += 1
            filtered_taps += tap_count
            continue

        tap_counts_by_grid[grid_idx] += tap_count
        touch_by_grid[grid_idx] = final_touch

    return filtered_chart, filtered_tokens, filtered_taps

# ═══════════════════════════════════════════════════════════
# 模型加载 (延迟加载)
# ═══════════════════════════════════════════════════════════
_models_cache: dict = {}


def _load_compatible_state(model, state: dict) -> None:
    current = model.state_dict()
    compatible = {}
    skipped = []
    for name, tensor in state.items():
        if name in current and current[name].shape == tensor.shape:
            compatible[name] = tensor
        else:
            skipped.append(name)
    current.update(compatible)
    model.load_state_dict(current)
    if skipped:
        print(f"Skipped {len(skipped)} incompatible tensors: {skipped[:6]}")


def _load_model(stage: int):
    """加载指定 stage 的模型"""
    if stage in _models_cache:
        return _models_cache[stage]

    candidates = [
        Path(CKPT_DIR) / f"stage{stage}_last.pt",
        Path(CKPT_DIR) / f"stage{stage}_best.pt",
        Path(DATA_DIR) / f"stage{stage}_last.pt",
        Path(DATA_DIR) / f"stage{stage}_best.pt",
        Path(DATA_DIR) / f"stage{stage}.pt",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"Stage {stage} checkpoint not found")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", ckpt.get("cfg"))
    state = ckpt.get("model_state_dict", ckpt.get("model"))

    if stage == 1:
        model = Stage1ChartModel(cfg).to(DEVICE).eval()
    elif stage == 2:
        model = Stage2HoldModel(cfg).to(DEVICE).eval()
    elif stage == 3:
        model = Stage3SlideModel(cfg).to(DEVICE).eval()
    elif stage == 4:
        model = Stage4BreakModel(cfg).to(DEVICE).eval()
    elif stage == 5:
        model = Stage5ExModel(cfg).to(DEVICE).eval()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    _load_compatible_state(model, state)
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
    touchhold_bias: float,     # TouchHold 偏置
    filter_multi_tap: bool,    # 过滤三押及以上 Tap token
) -> torch.Tensor:
    """对 logits 施加类型偏置后采样"""
    device = logits.device

    # 构建偏置向量: density 提升所有音符 / 降低空位
    bias = (BIAS_NOTE_MASK.to(device) - BIAS_EMPTY_MASK.to(device)) * density
    bias += BIAS_TAP_MASK.to(device) * tap_bias
    bias += BIAS_HOLD_MASK.to(device) * hold_bias
    bias += BIAS_SLIDE_MASK.to(device) * slide_bias
    bias += BIAS_TOUCH_MASK.to(device) * touch_bias
    bias += BIAS_TOUCHHOLD_MASK.to(device) * touchhold_bias

    logits = logits + bias.view(1, 1, -1)
    if filter_multi_tap:
        logits = logits.masked_fill(
            BIAS_MULTI_TAP_MASK.to(device).view(1, 1, -1).bool(),
            float("-inf"),
        )

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


def _apply_stage1_bias(
    logits: torch.Tensor,
    density: float,
    tap_bias: float,
    hold_bias: float,
    slide_bias: float,
    touch_bias: float,
    touchhold_bias: float,
    filter_multi_tap: bool,
) -> torch.Tensor:
    """Apply Stage1 sampling biases without sampling, for AR generation."""
    device = logits.device
    bias = (BIAS_NOTE_MASK.to(device) - BIAS_EMPTY_MASK.to(device)) * density
    bias += BIAS_TAP_MASK.to(device) * tap_bias
    bias += BIAS_HOLD_MASK.to(device) * hold_bias
    bias += BIAS_SLIDE_MASK.to(device) * slide_bias
    bias += BIAS_TOUCH_MASK.to(device) * touch_bias
    bias += BIAS_TOUCHHOLD_MASK.to(device) * touchhold_bias

    logits = logits + bias.view(1, 1, -1)
    if filter_multi_tap:
        logits = logits.masked_fill(
            BIAS_MULTI_TAP_MASK.to(device).view(1, 1, -1).bool(),
            float("-inf"),
        )
    return logits


def _apply_stage1_constraints(
    logits: torch.Tensor,
    frame_idx: int,
    generated: torch.Tensor,
    frame_rate: float,
    measure_dur: float,
    subdiv: int,
    filter_multi_tap: bool,
) -> torch.Tensor:
    if not filter_multi_tap:
        return logits

    current_grid = _output_grid_index(frame_idx, frame_rate, measure_dur, subdiv)
    tap_counts = _mask_like_logits(TAP_COUNTS_BY_ID, logits).to(logits.device)
    has_touch = _mask_like_logits(HAS_TOUCH_BY_ID.float(), logits).to(logits.device).bool()
    note_candidate = (tap_counts > 0) | has_touch
    masked = logits.clone()

    for batch_idx in range(generated.shape[0]):
        existing_taps = 0
        existing_touch = False
        for prev_frame in range(frame_idx):
            if _output_grid_index(prev_frame, frame_rate, measure_dur, subdiv) != current_grid:
                continue
            tid = int(generated[batch_idx, prev_frame].item())
            if 0 <= tid < TAP_COUNTS_BY_ID.shape[0]:
                existing_taps += int(TAP_COUNTS_BY_ID[tid].item())
                existing_touch = existing_touch or bool(HAS_TOUCH_BY_ID[tid].item())

        final_taps = existing_taps + tap_counts
        invalid = note_candidate & (
            (final_taps >= 3) |
            ((final_taps >= 2) & (existing_touch | has_touch))
        )
        if invalid.any():
            masked[batch_idx, :, invalid] = float("-inf")

    return masked


def _biased_binary_predict(logits: torch.Tensor, positive_bias: float) -> torch.Tensor:
    """对二分类 logits 的正类施加偏置后取 argmax。"""
    if positive_bias:
        logits = logits.clone()
        logits[..., 1] += positive_bias
    return logits.argmax(dim=-1).bool()


# ═══════════════════════════════════════════════════════════
# Slide 路径校验
# ═══════════════════════════════════════════════════════════

def _slide_first_target(path_str: str) -> tuple[str, int] | None:
    """Return the first slide connector and target position."""
    # VXX paths visually end on the second digit.
    m = re.match(r'^(V)([1-8])([1-8])', path_str)
    if m:
        return m.group(1), int(m.group(3))

    m = re.match(r'^(pp|qq|PP|QQ|[><^vVpqszw-])([1-8])', path_str)
    if m:
        return m.group(1), int(m.group(2))

    return None


def _validate_slide_path(start_pos: str, path_str: str) -> bool:
    """检查 slide 路径是否合法 (不产生相邻位置/直线同点)"""
    try:
        start = int(start_pos)
    except ValueError:
        return True  # 非数字位置 (如触摸), 跳过检查

    first = _slide_first_target(path_str)
    if first is None:
        return True  # 无法解析, 保守放行

    connector, target = first
    if connector == "-" and target == start:
        return False  # 1-1[...], 5-5[...] 这类直线同点无效

    # 相邻检查: |a-b| == 1 或 == 7 (环形)
    diff = abs(start - target)
    if diff == 1 or diff == 7:
        return False  # 相邻, 无效

    return True


def _duration_bin_to_str(dur_bin: int) -> str:
    secs = 2.0 ** (int(dur_bin) - 5)
    return f"{max(1, round(secs * 4))}:1"


def _default_slide_path(start_pos: str) -> str:
    try:
        start = int(start_pos)
    except ValueError:
        return "-4"
    target = ((start + 3) % 8) + 1
    return f"-{target}"


def _slide_vocab_token_to_params(token: str) -> tuple[str, str]:
    m = re.match(r"^(.+)\[([^\]]+)\]$", token)
    if m:
        return m.group(1), m.group(2)
    return token, ""


def _invalid_slide_vocab_ids_for_start(start_pos: str) -> list[int]:
    invalid = []
    for pid, token in SLIDE_VOCAB_INV.items():
        if token in ("<PAD>", "<EOS>"):
            continue
        path, _ = _slide_vocab_token_to_params(token)
        if not _validate_slide_path(start_pos, path):
            invalid.append(int(pid))
    return invalid


def _as_slot_array(arr: np.ndarray, length: int, slots: int = 1) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < length:
        pad = np.zeros((length - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:length, :max(slots, arr.shape[1])]


# ═══════════════════════════════════════════════════════════
# 推理核心
# ═══════════════════════════════════════════════════════════


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
    wifi_bias: float,
    touch_bias: float,
    touchhold_bias: float,
    break_bias: float,
    filter_multi_tap: bool,
    skip_stages: list[str] | None = None,
    progress=gr.Progress(),
) -> tuple[str, str]:
    skip_stages = set(skip_stages or [])
    """核心推理函数，返回 (simai文本, 状态信息)"""
    from AudioTokenizer import AudioTokenizer
    from BeatTokenizer import BeatTokenizer

    diff_num = DIFF_MAP.get(difficulty, 5)
    diff_id = DIFF_ID.get(difficulty, 4)

    # ── 1. 音频编码 ──
    progress(0.05, desc="正在编码音频...")
    at = AudioTokenizer(
        num_codebooks=cfg.audio.num_codebooks,
        device=DEVICE,
        local_path=cfg.audio.premodel_path or None,
    )
    ad = at.encode_file(mp3_path)
    bt = BeatTokenizer(
        method=cfg.beat.method,
        target_bpm=None if bpm_override <= 0 else bpm_override,
        quantize_beats=cfg.beat.quantize_beats,
        bpm_min=cfg.beat.bpm_min,
        bpm_max=cfg.beat.bpm_max,
    )
    bl = bt.analyse(mp3_path)

    fr = ad.frame_rate
    nf = ad.num_frames
    bpm = bl.bpm if bpm_override <= 0 else bpm_override
    subdiv = cfg.chart.target_subdiv
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
    chart = m1.generate(
        audio,
        beat,
        diff_t,
        lvl_t,
        tags_t,
        temperature=temperature,
        top_k=top_k,
        logits_processor=lambda logits, t, generated: _apply_stage1_constraints(
            _apply_stage1_bias(
                logits,
                density,
                tap_bias,
                hold_bias,
                slide_bias,
                touch_bias,
                touchhold_bias,
                filter_multi_tap,
            ),
            t,
            generated,
            fr,
            measure_dur,
            subdiv,
            filter_multi_tap,
        ),
    )
    T = chart.shape[1]
    filtered_multi_tap_tokens = 0
    filtered_multi_tap_count = 0
    if filter_multi_tap:
        chart, filtered_multi_tap_tokens, filtered_multi_tap_count = _filter_multi_tap_chart(
            chart,
            fr,
            measure_dur,
            subdiv,
        )

    hold_ids = {tid for tok, tid in VOCAB.items() if tok.startswith("hold")}

    # ── Stage 2: Hold 持续时间 ──
    if "Stage 2" in skip_stages:
        hold_durs = np.zeros((T, 1), dtype=np.int64)
    else:
        progress(0.35, desc="Stage 2: 预测 Hold 持续时间...")
        m2 = _load_model(2)
        hold_mask = torch.zeros(1, T, dtype=torch.bool, device=DEVICE)
        for hid in hold_ids:
            hold_mask = hold_mask | (chart == hid)
        dur_pred = m2.generate(chart, audio, beat, diff_t, lvl_t, tags_t, hold_mask,
                               temperature=temperature)
        hold_durs = _as_slot_array(dur_pred[0].cpu().numpy(), T)

    # ── Stage 3: Slide 路径 (带采样, 支持多段路径) ──
    if "Stage 3" in skip_stages:
        slide_paths = np.zeros((T, 1), dtype=np.int64)
    else:
        progress(0.55, desc="Stage 3: 预测 Slide 路径...")
        m3 = _load_model(3)
        out3 = m3(chart, audio, beat, diff_t, lvl_t, tags_t)

        # slide_logits: (B, T, S, V) 其中 S=max_slide_slots, V=slide_vocab_size
        slide_logits = out3["logits"][0]  # (T, S, V)
        S = slide_logits.shape[1]  # max_slide_slots (通常 8)

        # 对每个 slot 独立采样 (temperature + top_k)
        slide_temp = temperature * 0.7  # slide 略低温度, 更稳定
        slide_topk = max(10, top_k // 2)

        if slide_temp > 0:
            sl = slide_logits / slide_temp
        else:
            sl = slide_logits.clone()
        if wifi_bias:
            sl = sl + _mask_like_logits(BIAS_WIFI_SLIDE_MASK, sl).view(1, 1, -1) * wifi_bias

        # Mask slide vocab entries that cannot legally attach to the predicted
        # Stage1 slide start position, e.g. slide1 + -1[8:1] -> 1-1[8:1].
        chart_np_for_slide = chart[0].detach().cpu().numpy()
        invalid_slide_ids_by_start: dict[str, list[int]] = {}
        for f in range(T):
            tok_str = ID_TO_TOKEN.get(int(chart_np_for_slide[f]))
            if not tok_str:
                continue
            slide_slot = 0
            for part in tok_str.split("+"):
                st = SimaiToken.from_string(part)
                if st is None or st.token_type != SimaiTokenType.SLIDE:
                    continue
                if slide_slot >= sl.shape[1]:
                    break
                if st.position not in invalid_slide_ids_by_start:
                    invalid_slide_ids_by_start[st.position] = [
                        pid for pid in _invalid_slide_vocab_ids_for_start(st.position)
                        if pid < sl.shape[-1]
                    ]
                invalid_ids = invalid_slide_ids_by_start[st.position]
                if invalid_ids:
                    sl[f, slide_slot, invalid_ids] = float("-inf")
                slide_slot += 1

        if slide_topk > 0 and slide_topk < sl.shape[-1]:
            topk_vals, _ = torch.topk(sl, slide_topk, dim=-1)
            min_topk = topk_vals[:, :, -1:]
            sl = torch.where(sl < min_topk, torch.full_like(sl, float("-inf")), sl)

        probs = F.softmax(sl, dim=-1)  # (T, S, V)
        flat_probs = probs.reshape(-1, sl.shape[-1])  # (T*S, V)
        slide_paths = torch.multinomial(flat_probs, 1).reshape(T, S).cpu().numpy()

    # ── Stage 4/5: Break / Ex ──
    note_mask = (chart > 0).bool()
    if "Stage 4" in skip_stages:
        break_pred = np.zeros((T, 1), dtype=bool)
    else:
        progress(0.70, desc="Stage 4: 预测 Break...")
        m4 = _load_model(4)
        break_logits = m4.forward(chart, audio, beat, diff_t, lvl_t, tags_t)["logits"]
        break_pred = _as_slot_array(
            _biased_binary_predict(break_logits, break_bias)[0].cpu().numpy(),
            T,
        ).astype(bool)

    if "Stage 5" in skip_stages:
        ex_pred = np.zeros_like(break_pred, dtype=bool)
    else:
        progress(0.78, desc="Stage 5: 预测 Ex...")
        m5 = _load_model(5)
        ex_pred = _as_slot_array(
            m5.predict(chart, audio, beat, diff_t, lvl_t, tags_t)[0].cpu().numpy(),
            T,
        ).astype(bool)

    # ── 构建 simai ──
    progress(0.85, desc="构建 simai 谱面...")
    chart_np = chart[0].cpu().numpy()
    measures: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))

    note_count = 0
    hold_count = 0
    slide_count = 0
    tap_count = 0
    break_count = 0
    ex_count = 0

    for f in range(T):
        tid = int(chart_np[f])
        if tid <= 0:
            continue
        tok_str = ID_TO_TOKEN.get(tid)
        if tok_str is None:
            continue

        m, bi = _output_grid_index(f, fr, measure_dur, subdiv)

        hold_slot = 0
        slide_slot = 0
        frame_objects: list[tuple[SimaiToken, int]] = []
        for obj_slot, part in enumerate(tok_str.split("+")):
            st = SimaiToken.from_string(part)
            if st is None:
                continue

            # 注入 hold 持续时间
            if st.token_type == SimaiTokenType.HOLD:
                if hold_slot < hold_durs.shape[1] and hold_durs[f, hold_slot] > 0:
                    st.params["dur"] = _duration_bin_to_str(int(hold_durs[f, hold_slot]))
                elif "dur" not in st.params or not st.params["dur"]:
                    st.params["dur"] = "4:1"
                hold_slot += 1
                hold_count += 1

            # 注入 slide 路径 + 持续时间
            if st.token_type == SimaiTokenType.SLIDE:
                pid = int(slide_paths[f, slide_slot]) if slide_slot < slide_paths.shape[1] else 0
                if pid > 1:  # <PAD>=0, <EOS>=1
                    seg = SLIDE_VOCAB_INV.get(pid, "")
                    if seg and seg not in ("<PAD>", "<EOS>"):
                        path, timing = _slide_vocab_token_to_params(seg)
                        if _validate_slide_path(st.position, path):
                            st.params["path"] = path
                            if timing:
                                st.params["dur"] = timing
                if "path" not in st.params or not st.params["path"]:
                    st.params["path"] = _default_slide_path(st.position)
                if "dur" not in st.params or not st.params["dur"]:
                    path_key = st.params.get("path", "")
                    if path_key and path_key in PATH_BEST_TIMING:
                        st.params["dur"] = PATH_BEST_TIMING[path_key]
                    elif hold_slot < hold_durs.shape[1] and hold_durs[f, hold_slot] > 0:
                        st.params["dur"] = _duration_bin_to_str(int(hold_durs[f, hold_slot]))
                    else:
                        st.params["dur"] = "4:1"
                slide_slot += 1
                slide_count += 1

            # 注入 break/ex
            if obj_slot < break_pred.shape[1] and break_pred[f, obj_slot]:
                st.params["break"] = ""
            if obj_slot < ex_pred.shape[1] and ex_pred[f, obj_slot]:
                st.params["ex"] = ""

            frame_objects.append((st, obj_slot))

        frame_notes = []
        for st, _ in frame_objects:
            if st.token_type == SimaiTokenType.TAP:
                tap_count += len(st.position)
            if st.has_break:
                break_count += 1
            if st.has_ex:
                ex_count += 1
            frame_notes.append(note_to_simai(st))
            note_count += len(st.position) if st.token_type == SimaiTokenType.TAP else 1

        if frame_notes:
            measures[m][bi].append("/".join(frame_notes))

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
        body = ",".join(parts) + ","
        if m == 0:
            lines.append(f"({bpm:.1f}){{{subdiv}}}{body}")
        else:
            lines.append(f"{{{subdiv}}}{body}")
    lines.append("E")

    simai_text = "\n".join(lines)

    # 统计信息
    info = (
        f"✅ 生成完成！\n\n"
        f"📊 统计信息:\n"
        f"  - 总音符数: {note_count}\n"
        f"  - Tap: {tap_count} | Hold: {hold_count} | Slide: {slide_count}\n"
        f"  - Break: {break_count} | Ex: {ex_count}\n"
        f"  - 三押及以上 Tap token 过滤: {'开启' if filter_multi_tap else '关闭'}"
        f" (vocab {len(_MULTI_TAP_IDS)} 个, 本次过滤 {filtered_multi_tap_tokens} 个 token/"
        f"{filtered_multi_tap_count} 个 tap)\n"
        f"  - WiFi bias: {wifi_bias:.2f} (vocab {len(_WIFI_SLIDE_IDS)} 个)\n"
        f"  - 跳过: {', '.join(sorted(skip_stages)) if skip_stages else '无'}\n"
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
        支持 Master 13 难度，使用多阶段 Transformer 模型。
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
                        value=cfg.chart.default_difficulty if hasattr(cfg.chart, 'default_difficulty') else "Master",
                        label="难度",
                    )
                    level = gr.Slider(
                        minimum=1.0, maximum=15.0,
                        value=cfg.chart.default_level if hasattr(cfg.chart, 'default_level') else 13.0,
                        step=0.1,
                        label="等级",
                    )

                designer = gr.Dropdown(
                    choices=DESIGNER_TAGS,
                    value="AI",
                    label="谱面作者 (Designer)",
                    allow_custom_value=True,
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
                        minimum=0.1, maximum=2.0,
                        value=cfg.generation.temperature,
                        step=0.05,
                        label="Temperature (温度)",
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=200,
                        value=cfg.generation.top_k,
                        step=1,
                        label="Top-K 采样",
                    )

                bpm_override = gr.Number(
                    value=-1, label="BPM 覆盖 (-1=自动检测)",
                    precision=1,
                )

                gr.Markdown("### 🎯 物块偏置 (Bias)")
                gr.Markdown("*正值=更多, 负值=更少, 0=不偏置*")

                density = gr.Slider(
                    minimum=-5.0, maximum=5.0, value=0.0, step=0.01,
                    label="📊 整体密度 (Density)",
                )
                with gr.Row():
                    tap_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.01,
                        label="👆 Tap",
                    )
                    hold_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.01,
                        label="✋ Hold",
                    )
                with gr.Row():
                    slide_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.01,
                        label="⭐ Slide",
                    )
                    wifi_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.01,
                        label="📶 WiFi",
                    )
                with gr.Row():
                    touch_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.01,
                        label="🖐 Touch",
                    )
                    touchhold_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.01,
                        label="✋ TouchHold",
                    )
                    break_bias = gr.Slider(
                        minimum=-5.0, maximum=5.0, value=0.0, step=0.01,
                        label="💥 Break",
                    )

                filter_multi_tap = gr.Checkbox(
                    value=True,
                    label="过滤三押及以上 Tap",
                )

                skip_stages = gr.CheckboxGroup(
                    choices=["Stage 2", "Stage 3", "Stage 4", "Stage 5"],
                    value=["Stage 5"],
                    label="跳过 Stage",
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
                        dens, tb, hb, sb, wb, ttb, thb, bb, fmt, skips):
            if audio_file is None:
                return "", "❌ 请先上传 MP3 文件！", gr.DownloadButton(visible=False)

            mp3_path = audio_file if isinstance(audio_file, str) else audio_file.name
            try:
                simai_text, info = generate_chart(
                    mp3_path, diff, lvl, des, cols, temp, topk, bpm_ov,
                    dens, tb, hb, sb, wb, ttb, thb, bb, fmt, skips,
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
                    density, tap_bias, hold_bias, slide_bias, wifi_bias, touch_bias,
                    touchhold_bias, break_bias, filter_multi_tap, skip_stages],
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
        | Tap/Hold/Slide/Touch/Break | 正值 → 该类型更多；负值 → 更少 | 0 |
        | 过滤三押及以上 Tap | 移除同一时刻 3 个或更多 Tap 的无理多押 | 开启 |
        | 跳过 Stage | 可跳过 Hold/Slide/Break/Ex 后处理；默认跳过 Stage 5 | Stage 5 |
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # 端口优先级: CLI --port > 配置文件 server.port > 默认 7860
    _port = _cli_port
    if _port is None and hasattr(cfg, 'server'):
        _port = cfg.server.port
    if _port is None:
        _port = 7860
    demo.launch(
        server_name=cfg.server.host if hasattr(cfg, 'server') else "0.0.0.0",
        server_port=_port,
        share=cfg.server.share if hasattr(cfg, 'server') else False,
        inbrowser=cfg.server.inbrowser if hasattr(cfg, 'server') else True,
        allowed_paths=[str(OUTPUT_DIR.resolve()), str(Path(DATA_DIR).resolve())],
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
