"""
Slide Star Tokenizer — Stage 2 星星（Slide）详细路径生成

Token 格式 (目标序列):
  [BOS, dur_num, dur_den, CONN_0, POS_1, CONN_1, POS_2, ..., POS_end, EOS]

输入上下文:
  - start_pos: 星星起始位置 (button 1-8)
  - global_audio: 全曲音频摘要
  - local_audio: 当前时间点音频
  - onset: 音频节拍特征
  - stage1_seq: Stage 1 生成的序列（含 slide_start 标记）

位置词汇表 (41 个):
  0-7:   按钮 1-8
  8-40:  触控区 0-32 (E1-E8=0-7, B1-B8=8-15, C=16, A1-D8 outer=17-32)

连接类型 (14 个):
  0: straight (-)    1: arc_CW (>)     2: arc_CCW (<)
  3: arc_auto (^)    4: center (v)     5: inner_CW (p)
  6: inner_CCW (q)   7: zigzag_S (s)   8: zigzag_Z (z)
  9: wifi (w)       10: V_shape (V)
 11: large_CW (pp)  12: large_CCW (qq)  13: branch (*)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
# Token ID 分配
# ═══════════════════════════════════════════════════════════════════════

# 特殊 token (0-2)
SLD_STAR_BOS = 0
SLD_STAR_EOS = 1
SLD_STAR_PAD = 2

# 位置 token (3-43): 41 个位置
SLD_POS_BASE = 3
SLD_POS_COUNT = 41  # 8 buttons + 33 touch zones

# 连接类型 token (44-57): 14 种
SLD_CONN_BASE = SLD_POS_BASE + SLD_POS_COUNT
SLD_CONN_COUNT = 14

# 时长 token (58-77): 复用 DUR_NUM(8) + DUR_DEN(12)
SLD_DUR_NUM_BASE = SLD_CONN_BASE + SLD_CONN_COUNT
SLD_DUR_NUM_COUNT = 8   # [1,2,3,4,6,8,12,16]
SLD_DUR_DEN_BASE = SLD_DUR_NUM_BASE + SLD_DUR_NUM_COUNT
SLD_DUR_DEN_COUNT = 12  # [1,2,3,4,6,8,12,16,24,32,48,64]

SLD_STAR_VOCAB_SIZE = SLD_DUR_DEN_BASE + SLD_DUR_DEN_COUNT


# ═══════════════════════════════════════════════════════════════════════
# 位置编码
# ═══════════════════════════════════════════════════════════════════════

def pos_to_id(is_button: bool, index: int) -> int:
    """将位置转为 token ID。button: index=1-8, touch: index=0-32"""
    if is_button:
        if not (1 <= index <= 8):
            raise ValueError(f"Button position out of range: {index}")
        return SLD_POS_BASE + (index - 1)
    else:
        if not (0 <= index <= 32):
            raise ValueError(f"Touch zone out of range: {index}")
        return SLD_POS_BASE + 8 + index


def id_to_pos(token_id: int) -> tuple[bool, int]:
    """将 token ID 转为 (is_button, index)。"""
    idx = token_id - SLD_POS_BASE
    if idx < 8:
        return (True, idx + 1)      # button 1-8
    else:
        return (False, idx - 8)     # touch zone 0-32


# ═══════════════════════════════════════════════════════════════════════
# 连接类型编码
# ═══════════════════════════════════════════════════════════════════════

CONN_NAMES = [
    "straight", "arc_CW", "arc_CCW", "arc_auto",
    "center", "inner_CW", "inner_CCW",
    "zigzag_S", "zigzag_Z", "wifi", "V_shape",
    "large_CW", "large_CCW", "branch",
]

CONN_CHAR_TO_ID: dict[str, int] = {
    "-": 0,  ">": 1,  "<": 2,  "^": 3,
    "v": 4,  "p": 5,  "q": 6,
    "s": 7,  "z": 8,  "w": 9,  "V": 10,
    "pp": 11, "qq": 12, "*": 13,
}

CONN_ID_TO_CHAR: dict[int, str] = {v: k for k, v in CONN_CHAR_TO_ID.items()}


def conn_to_id(conn_char: str) -> int:
    """连接字符 → token ID。"""
    cid = CONN_CHAR_TO_ID.get(conn_char)
    if cid is None:
        raise ValueError(f"Unknown connector: {conn_char}")
    return SLD_CONN_BASE + cid


def id_to_conn(token_id: int) -> str:
    """token ID → 连接字符。"""
    cid = token_id - SLD_CONN_BASE
    return CONN_ID_TO_CHAR.get(cid, "?") if 0 <= cid < SLD_CONN_COUNT else "?"


# ═══════════════════════════════════════════════════════════════════════
# 时长编码
# ═══════════════════════════════════════════════════════════════════════

DUR_NUM_VALUES = [1, 2, 3, 4, 6, 8, 12, 16]
DUR_DEN_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

DUR_NUM_TO_ID: dict[int, int] = {v: SLD_DUR_NUM_BASE + i for i, v in enumerate(DUR_NUM_VALUES)}
DUR_DEN_TO_ID: dict[int, int] = {v: SLD_DUR_DEN_BASE + i for i, v in enumerate(DUR_DEN_VALUES)}
ID_TO_DUR_NUM: dict[int, int] = {v: k for k, v in DUR_NUM_TO_ID.items()}
ID_TO_DUR_DEN: dict[int, int] = {v: k for k, v in DUR_DEN_TO_ID.items()}


def _nearest(values: list[int], value: int) -> int:
    return min(values, key=lambda x: abs(x - value))


def snap_duration(num: int, den: int) -> tuple[int, int]:
    """将时长 snap 到最近的合法值。"""
    n = _nearest(DUR_NUM_VALUES, max(1, num))
    d = _nearest(DUR_DEN_VALUES, max(1, den))
    return n, d


# ═══════════════════════════════════════════════════════════════════════
# Slide Star Token 序列编解码
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SlideStarPath:
    """一颗星星的完整路径。"""
    start_pos: int           # 起始按钮 1-8 (INPUT, not in target)
    duration: tuple[int, int]  # (num, den) in beats
    waypoints: list[int]     # 中间+结束位置 (按钮 1-8 或触控区 0-32)
    connectors: list[str]    # 连接类型，len = len(waypoints)


def encode_slide_star(path: SlideStarPath) -> list[int]:
    """
    将星星路径编码为目标 token 序列。

    格式: [BOS, dur_num, dur_den, CONN_0, POS_1, CONN_1, POS_2, ..., POS_n, EOS]
    """
    num, den = snap_duration(*path.duration)
    tokens = [SLD_STAR_BOS, DUR_NUM_TO_ID[num], DUR_DEN_TO_ID[den]]

    for i, wp in enumerate(path.waypoints):
        # 判断是按钮还是触控区
        if 1 <= wp <= 8:
            tokens.append(pos_to_id(True, wp))
        elif 0 <= wp <= 32:
            tokens.append(pos_to_id(False, wp))
        else:
            raise ValueError(f"Invalid waypoint: {wp}")
        # 下一个连接类型（最后一个 waypoint 后不加 connector）
        if i < len(path.connectors):
            tokens.append(conn_to_id(path.connectors[i]))

    tokens.append(SLD_STAR_EOS)
    return tokens


def decode_slide_star(tokens: list[int]) -> SlideStarPath | None:
    """
    从 token 序列解码星星路径。

    Returns None if invalid sequence.
    """
    if not tokens or tokens[0] != SLD_STAR_BOS:
        return None

    i = 1
    if i + 1 >= len(tokens):
        return None

    num_id, den_id = tokens[i], tokens[i + 1]
    i += 2
    dur_num = ID_TO_DUR_NUM.get(num_id, 1)
    dur_den = ID_TO_DUR_DEN.get(den_id, 1)

    waypoints: list[int] = []
    connectors: list[str] = []

    while i < len(tokens) and tokens[i] != SLD_STAR_EOS:
        tid = tokens[i]
        if SLD_POS_BASE <= tid < SLD_POS_BASE + SLD_POS_COUNT:
            is_btn, idx = id_to_pos(tid)
            waypoints.append(idx if is_btn else idx)
        elif SLD_CONN_BASE <= tid < SLD_CONN_BASE + SLD_CONN_COUNT:
            connectors.append(id_to_conn(tid))
        i += 1

    # start_pos 不在 token 序列中，设为第一个 waypoint 的推测值
    return SlideStarPath(
        start_pos=waypoints[0] if waypoints and waypoints[0] <= 8 else 1,
        duration=(dur_num, dur_den),
        waypoints=waypoints,
        connectors=connectors[:len(waypoints)],
    )


# ═══════════════════════════════════════════════════════════════════════
# 从 maidata 原始 note 构建 SlideStarPath
# ═══════════════════════════════════════════════════════════════════════

def from_mai_note(note) -> SlideStarPath | None:
    """
    从解析后的 _Note 对象构建 SlideStarPath。

    处理格式:
      7-2[4:1]      → start=7, waypoints=[2], conn=["-"], dur=(4,1)
      6<1[4:1]      → start=6, waypoints=[1], conn=["<"], dur=(4,1)
      3-7-4[8:1]    → start=3, waypoints=[7,4], conn=["-","-"], dur=(8,1)
    """
    if not note.is_slide:
        return None

    path = note.slide_path or note.positions
    types = note.slide_types or []
    if len(path) < 2:
        return None

    start = path[0]
    if not (1 <= start <= 8):
        return None

    waypoints = []
    connectors = []
    for idx in range(1, len(path)):
        wp = path[idx]
        if 1 <= wp <= 8:
            waypoints.append(wp)
        else:
            # 非按钮位置，尝试作为触控区
            waypoints.append(max(0, min(32, wp)))
        if idx - 1 < len(types):
            connectors.append(types[idx - 1])
        else:
            connectors.append("-")

    dur = note.hold_duration if note.hold_duration else (1, 1)

    return SlideStarPath(
        start_pos=start,
        duration=dur,
        waypoints=waypoints,
        connectors=connectors,
    )


# ═══════════════════════════════════════════════════════════════════════
# Debug / Info
# ═══════════════════════════════════════════════════════════════════════

def print_vocab_info() -> None:
    print(f"Slide Star Vocab")
    print(f"  Vocab size: {SLD_STAR_VOCAB_SIZE}")
    print(f"  Positions: {SLD_POS_COUNT} ({SLD_POS_BASE}-{SLD_POS_BASE + SLD_POS_COUNT - 1})")
    print(f"  Connectors: {SLD_CONN_COUNT} ({SLD_CONN_BASE}-{SLD_CONN_BASE + SLD_CONN_COUNT - 1})")
    print(f"  Dur nums: {SLD_DUR_NUM_COUNT}")
    print(f"  Dur dens: {SLD_DUR_DEN_COUNT}")
    print(f"  Specials: BOS={SLD_STAR_BOS} EOS={SLD_STAR_EOS} PAD={SLD_STAR_PAD}")


if __name__ == "__main__":
    print_vocab_info()
    print()

    # 测试
    p = SlideStarPath(
        start_pos=7,
        duration=(4, 1),
        waypoints=[2],
        connectors=["-"],
    )
    tokens = encode_slide_star(p)
    print(f"7-2[4:1] → {tokens}")
    decoded = decode_slide_star(tokens)
    print(f"  decoded: {decoded}")

    p2 = SlideStarPath(
        start_pos=3,
        duration=(8, 1),
        waypoints=[7, 4],
        connectors=["-", "-"],
    )
    tokens2 = encode_slide_star(p2)
    print(f"3-7-4[8:1] → {tokens2}")
    print(f"  decoded: {decode_slide_star(tokens2)}")
