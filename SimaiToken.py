"""
Simai Tokenizer - 将 simai 谱面文本 token 化为统一的 token 序列

Token 格式:
  tap<positions>{params}   - Tap/双押 (positions 是连续数字串)
  hold<position>{params}   - Hold 长按
  slide<position>{params}  - Slide 滑条
  touch<position>{params}  - Touch 触摸
  rest                    - 休止符
  bpm<value>              - BPM 变更
  measure<N>              - 小节标记

Params (花括号内逗号分隔):
  break    - 绝赞
  ex       - Ex-note
  firework - 烟花特效
  dur:...  - 持续时间
  path:... - Slide 路径
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

class SimaiTokenType(Enum):
    """Token 类型枚举"""
    TAP = "tap"
    HOLD = "hold"
    SLIDE = "slide"
    TOUCH = "touch"
    REST = "rest"
    BPM = "bpm"
    MEASURE = "measure"


@dataclass
class SimaiToken:
    """一个 token，表示一个音符物块或控制标记"""

    token_type: SimaiTokenType
    position: str = ""           # 位置 (e.g., "1", "12", "B1", "C")
    params: dict[str, str] = field(default_factory=dict)
    measure: int = 0             # 历史命名: 实际表示第几个 {N} BPM 拍组
    beat: float = 0.0            # 当前 BPM 拍内位置 (归一化 0~1)
    subdiv: int = 4              # 当前 BPM 拍被分为几份 ({N} 中的 N)
    raw_text: str = ""

    def to_string(self) -> str:
        """序列化为字符串 (不含 subdiv/measure/beat, 仅类型+位置+参数)"""
        base = f"{self.token_type.value}{self.position}"
        if self.params:
            parts = []
            for k, v in self.params.items():
                parts.append(f"{k}:{v}" if v else k)
            base += "{" + ",".join(parts) + "}"
        return base

    def to_full_string(self) -> str:
        """完整字符串 含 subdiv 和时序信息"""
        base = self.to_string()
        return f"[subdiv={self.subdiv}, M{self.measure}:{self.beat:.3f}] {base}"

    @classmethod
    def from_string(cls, s: str) -> Optional["SimaiToken"]:
        """从 to_string() 输出反序列化"""
        s = s.strip()
        if not s:
            return None
        match = re.match(r'^([a-z]+)([A-Z]?\d*)(?:\{(.+)\})?$', s)
        if not match:
            return None
        try:
            ttype = SimaiTokenType(match.group(1))
        except ValueError:
            return None
        position = match.group(2)
        if ttype == SimaiTokenType.TOUCH and re.fullmatch(r"[ABDE]", position):
            return None
        if ttype == SimaiTokenType.HOLD and re.fullmatch(r"[ABDE]", position):
            return None
        params = {}
        if match.group(3):
            for part in match.group(3).split(","):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    params[k.strip()] = v.strip()
                else:
                    params[part] = ""
        return cls(token_type=ttype, position=position, params=params)

    @property
    def is_note(self) -> bool:
        return self.token_type in (
            SimaiTokenType.TAP, SimaiTokenType.HOLD,
            SimaiTokenType.SLIDE, SimaiTokenType.TOUCH,
        )

    @property
    def has_break(self) -> bool:
        return "break" in self.params

    @property
    def has_ex(self) -> bool:
        return "ex" in self.params

    @property
    def has_firework(self) -> bool:
        return "firework" in self.params

    @property
    def duration(self) -> Optional[str]:
        return self.params.get("dur")

    @property
    def slide_path(self) -> Optional[str]:
        return self.params.get("path")

    def __repr__(self) -> str:
        return f"Token({self.to_string()})"


# ============================================================
# SimaiTokenizer
# ============================================================

class SimaiTokenizer:
    """将 simai 谱面文本 token 化"""

    HOLD_PATTERN = re.compile(r'^(\d+)([bx]*?)h\[(.+?)\]$')
    TOUCH_HOLD_PATTERN = re.compile(r'^([A-E])(\d*)h\[(.+?)\]$')
    BREAK_EX_TAP_PATTERN = re.compile(r'^(\d+)([bx]+)$')
    SIMPLE_TAP_PATTERN = re.compile(r'^(\d+)$')
    TOUCH_PATTERN = re.compile(r'^(C\d*|[ABDE]\d+)$')
    TOUCH_FIREWORK_PATTERN = re.compile(r'^(C\d*|[ABDE]\d+)f$')

    def __init__(self, bpm: float = 120.0):
        self.bpm = bpm
        self.current_measure = 0
        self.current_subdiv = 4

    def tokenize(self, raw_notes: str) -> list[SimaiToken]:
        """将原始谱面文本 token 化"""
        tokens: list[SimaiToken] = []
        self.current_measure = 0
        self.current_subdiv = 4

        if not raw_notes.strip():
            return tokens

        for line in raw_notes.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # 小节标记 (BPM){N}
            mm = re.match(r'(?:\((\d+\.?\d*)\))?\{(\d+)\}(.*)', line)
            if mm:
                new_bpm = mm.group(1)
                if new_bpm:
                    self.bpm = float(new_bpm)
                    tokens.append(SimaiToken(
                        SimaiTokenType.BPM, str(self.bpm),
                        measure=self.current_measure, beat=0.0,
                        subdiv=self.current_subdiv,
                        raw_text=f"({new_bpm})",
                    ))
                self.current_subdiv = int(mm.group(2))
                tokens.append(SimaiToken(
                    SimaiTokenType.MEASURE, str(self.current_subdiv),
                    measure=self.current_measure, beat=0.0,
                    subdiv=self.current_subdiv,
                    raw_text=f"{{{self.current_subdiv}}}",
                ))
                rest = mm.group(3).strip()
                if rest:
                    tokens.extend(self._parse_beat_group(rest))
                self.current_measure += 1
                continue

            tokens.extend(self._parse_beat_group(line))
            self.current_measure += 1

        return tokens

    def _parse_beat_group(self, line: str) -> list[SimaiToken]:
        """解析一行内的所有拍"""
        tokens: list[SimaiToken] = []
        for beat_idx, beat_str in enumerate(line.split(",")):
            beat_str = beat_str.strip()
            if not beat_str:
                continue
            bp = beat_idx / self.current_subdiv

            # 内联 BPM: (120)1h[4:1]
            ibm = re.match(r'^\((\d+\.?\d*)\)(.+)$', beat_str)
            if ibm:
                tokens.append(SimaiToken(
                    SimaiTokenType.BPM, str(float(ibm.group(1))),
                    measure=self.current_measure, beat=bp,
                    subdiv=self.current_subdiv,
                    raw_text=f"({ibm.group(1)})",
                ))
                beat_str = ibm.group(2)

            # 假 Each ` → 当 / 处理
            if "`" in beat_str and "[" not in beat_str:
                beat_str = beat_str.replace("`", "/")

            # Each / 分割
            if "/" in beat_str and "/" not in re.sub(r'\[.*?\]', '', beat_str):
                # 有 / 不在 [] 内
                pass  # 下面统一处理

            # 判断是否为 each
            parts = self._split_each(beat_str)
            if len(parts) > 1:
                sub_tokens = []
                all_tap = True
                all_touch = True
                for p in parts:
                    t = self._parse_one(p, bp)
                    if t:
                        sub_tokens.append(t)
                        if t.token_type != SimaiTokenType.TAP:
                            all_tap = False
                        if t.token_type != SimaiTokenType.TOUCH:
                            all_touch = False

                if all_tap and len(sub_tokens) >= 2:
                    # 合并为 tapNNN
                    positions = sorted(sub_tokens, key=lambda x: int(x.position))
                    pos_str = "".join(t.position for t in positions)
                    params = {}
                    if any(t.has_break for t in sub_tokens):
                        params["break"] = ""
                    if any(t.has_ex for t in sub_tokens):
                        params["ex"] = ""
                    tokens.append(SimaiToken(
                        SimaiTokenType.TAP, pos_str, params,
                        measure=self.current_measure, beat=bp,
                        subdiv=self.current_subdiv,
                        raw_text=beat_str,
                    ))
                elif all_touch and len(sub_tokens) >= 2:
                    # 合并为 touch<area><num><area><num>...
                    sorted_touch = sorted(sub_tokens, key=lambda t: (t.position[0], int(t.position[1:]) if t.position[1:] else 0))
                    pos_str = "".join(t.position for t in sorted_touch)
                    params = {}
                    if any(t.has_firework for t in sub_tokens):
                        params["firework"] = ""
                    tokens.append(SimaiToken(
                        SimaiTokenType.TOUCH, pos_str, params,
                        measure=self.current_measure, beat=bp,
                        subdiv=self.current_subdiv,
                        raw_text=beat_str,
                    ))
                else:
                    tokens.extend(sub_tokens)
            else:
                t = self._parse_one(beat_str, bp)
                if t:
                    tokens.append(t)
        return tokens

    def _split_each(self, s: str) -> list[str]:
        """按 / 分割 each (忽略 [] 内的 /)"""
        parts = []
        depth = 0
        cur = []
        for ch in s:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            if ch == "/" and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            parts.append("".join(cur).strip())
        return [p for p in parts if p]

    def _parse_one(self, beat_str: str, bp: float) -> Optional[SimaiToken]:
        """解析单个拍"""
        beat_str = beat_str.strip()
        if not beat_str:
            return None
        # Simai uses standalone "E" as an end marker. It is not a touch note.
        if beat_str == "E":
            return None

        mk = lambda tt, pos, **kw: SimaiToken(
            tt, pos, {k: v for k, v in kw.items() if v is not None},
            measure=self.current_measure, beat=bp,
            subdiv=self.current_subdiv, raw_text=beat_str,
        )

        # Touch Hold: Ch[4:1]
        m = self.TOUCH_HOLD_PATTERN.match(beat_str)
        if m:
            pos = f"{m.group(1)}{m.group(2)}" if m.group(2) else m.group(1)
            return mk(SimaiTokenType.HOLD, pos, dur=m.group(3))

        # Firework Touch Hold: Cfh[4:1]
        m = re.match(r'^([A-E])(\d*)fh\[(.+?)\]$', beat_str)
        if m:
            pos = f"{m.group(1)}{m.group(2)}" if m.group(2) else m.group(1)
            return mk(SimaiTokenType.HOLD, pos, dur=m.group(3), firework="")

        # Firework Touch: C1f
        m = self.TOUCH_FIREWORK_PATTERN.match(beat_str)
        if m:
            return mk(SimaiTokenType.TOUCH, m.group(1), firework="")

        # Simple Touch: B1, C, E4. A/B/D/E require an explicit zone number;
        # bare E is the chart end marker above.
        m = self.TOUCH_PATTERN.match(beat_str)
        if m:
            return mk(SimaiTokenType.TOUCH, m.group(1))

        # Hold: 1h[4:1], 1bh[4:1]
        m = self.HOLD_PATTERN.match(beat_str)
        if m:
            flags = m.group(2) or ""
            kw = {"dur": m.group(3)}
            if "b" in flags:
                kw["break"] = ""
            if "x" in flags:
                kw["ex"] = ""
            return mk(SimaiTokenType.HOLD, m.group(1), **kw)

        # Slide: 1-4[8:1], 1>5-8[1:1], 4>8[8:1]*V28[8:1], 1-4b[8:1]
        st = self._try_slide(beat_str, bp)
        if st:
            return st

        # Break/Ex Tap: 1b, 1x, 1bx
        m = self.BREAK_EX_TAP_PATTERN.match(beat_str)
        if m:
            flags = m.group(2)
            kw = {}
            if "b" in flags:
                kw["break"] = ""
            if "x" in flags:
                kw["ex"] = ""
            return mk(SimaiTokenType.TAP, m.group(1), **kw)

        # Simple Tap: 1
        m = self.SIMPLE_TAP_PATTERN.match(beat_str)
        if m:
            return mk(SimaiTokenType.TAP, m.group(1))

        return None

    def _try_slide(self, beat_str: str, bp: float) -> Optional[SimaiToken]:
        """尝试解析 Slide"""
        # 去掉最后的 timing，得到路径
        tm = re.search(r'\[([^\]]*)\]$', beat_str)
        total_timing = tm.group(1) if tm else None
        path_str = re.sub(r'\[[^\]]*\]$', '', beat_str) if tm else beat_str

        # 提取 start 和 flags
        sm = re.match(r'^(\d+)([bx]*)', path_str)
        if not sm:
            return None
        start_pos = sm.group(1)
        flags = sm.group(2) or ""
        rest = path_str[sm.end():]

        if not rest:
            return None

        # 检查路径末尾是否有 b/x (如 1-4b[8:1] → b 在终点 4 后面)
        trail_match = re.search(r'([bx]+)$', rest)
        if trail_match:
            flags += trail_match.group(1)
            rest = rest[:trail_match.start()]

        # 检查是否以 connector 开头
        if not re.match(r'^(pp|qq|PP|QQ|[><^vVpqszw-])', rest):
            return None

        # 去掉中间 timing，保留纯路径
        clean_path = re.sub(r'\[[^\]]*\]', '', rest)

        params = {"path": clean_path}
        if total_timing:
            params["dur"] = total_timing
        if "b" in flags:
            params["break"] = ""
        if "x" in flags:
            params["ex"] = ""

        return SimaiToken(
            SimaiTokenType.SLIDE, start_pos, params,
            measure=self.current_measure, beat=bp,
            subdiv=self.current_subdiv,
            raw_text=beat_str,
        )


# ============================================================
# 便捷函数
# ============================================================

def tokenize_notes(raw_notes: str, bpm: float = 120.0) -> list[SimaiToken]:
    """将原始谱面文本 token 化"""
    return SimaiTokenizer(bpm=bpm).tokenize(raw_notes)


def tokens_to_string(tokens: list[SimaiToken]) -> str:
    """将 token 列表转为可读字符串"""
    return " ".join(t.to_string() for t in tokens)


def _token_to_simai_note(t: SimaiToken) -> str:
    """单个音符 token → simai 字符串"""
    flags = ""
    if t.has_break:
        flags += "b"
    if t.has_ex:
        flags += "x"

    if t.token_type == SimaiTokenType.TAP:
        # 合并的 tap → 拆回 each: "12" → "1/2"
        if len(t.position) > 1:
            parts = [c for c in t.position]  # 每位数字一个 tap
            return "/".join(p + flags for p in parts)
        return t.position + flags

    if t.token_type == SimaiTokenType.HOLD:
        dur = t.params.get("dur", "")
        pos = t.position
        if not pos or re.fullmatch(r"[ABDE]", pos):
            return ""
        if pos[0] in "ABCDE":
            if t.has_firework:
                return f"{pos}fh[{dur}]"
            return f"{pos}h[{dur}]"
        return f"{pos}{flags}h[{dur}]"

    if t.token_type == SimaiTokenType.SLIDE:
        path = t.params.get("path", "")
        dur = t.params.get("dur", "")
        timing = f"[{dur}]" if dur else ""
        return f"{t.position}{flags}{path}{timing}"

    if t.token_type == SimaiTokenType.TOUCH:
        # 合并的 touch → 拆回 each: "B3B4" → "B3/B4"
        parts = [
            p for p in re.findall(r'[A-E]\d*', t.position)
            if p.startswith("C") or re.fullmatch(r"[ABDE]\d+", p)
        ]
        if not parts:
            return ""
        if len(parts) > 1:
            fw = "f" if t.has_firework else ""
            return "/".join(p + fw for p in parts)
        if t.has_firework:
            return f"{parts[0]}f"
        return parts[0]

    return ""


def tokens_to_simai(
    tokens: list[SimaiToken],
    title: str = "",
    artist: str = "",
    whole_bpm: float = 120.0,
) -> str:
    """将 token 列表导出为 simai (maidata.txt) 格式"""
    lines = []

    if title:
        lines.append(f"&title={title}")
    if artist:
        lines.append(f"&artist={artist}")
    lines.append(f"&wholebpm={whole_bpm}")

    # 按小节分组, 同时记录每小节的 subdiv
    current_bpm = whole_bpm
    current_subdiv = 4
    measure_subdivs: dict[int, int] = {}       # measure -> subdiv
    measure_beats: dict[int, dict[int, list[SimaiToken]]] = {}

    for t in tokens:
        if t.token_type == SimaiTokenType.BPM:
            current_bpm = float(t.position)
        elif t.token_type == SimaiTokenType.MEASURE:
            current_subdiv = int(t.position)
            measure_subdivs[t.measure] = current_subdiv
        elif t.is_note:
            m = t.measure
            sd = measure_subdivs.get(m, current_subdiv)
            bi = round(t.beat * sd)
            bi = max(0, min(bi, sd - 1))
            if m not in measure_beats:
                measure_beats[m] = {}
            if bi not in measure_beats[m]:
                measure_beats[m][bi] = []
            measure_beats[m][bi].append(t)

    if not measure_beats:
        return "\n".join(lines)

    max_measure = max(measure_beats.keys())

    for m in range(max_measure + 1):
        sd = measure_subdivs.get(m, current_subdiv)
        beats = measure_beats.get(m, {})
        beat_parts = []
        for bi in range(sd):
            if bi in beats:
                toks = beats[bi]
                parts = [_token_to_simai_note(t) for t in toks]
                beat_parts.append("/".join(parts))
            else:
                beat_parts.append("")
        line = ",".join(beat_parts)
        lines.append(f"{{{sd}}}{line}")

    return "\n".join(lines)


def tokens_summary(tokens: list[SimaiToken]) -> dict:
    """统计各类音符数量"""
    counts = {"tap": 0, "hold": 0, "slide": 0, "touch": 0,
              "bpm": 0, "measure": 0, "break": 0, "ex": 0, "firework": 0}
    for t in tokens:
        tv = t.token_type.value
        if tv in counts:
            counts[tv] += 1
        if t.has_break:
            counts["break"] += 1
        if t.has_ex:
            counts["ex"] += 1
        if t.has_firework:
            counts["firework"] += 1
    counts["total_notes"] = counts["tap"] + counts["hold"] + counts["slide"] + counts["touch"]
    return counts


# ============================================================
# Subdiv 缩放 — 将 token 序列统一到目标 subdiv
# ============================================================

def rescale_subdiv(
    tokens: list[SimaiToken],
    target_subdiv: int,
) -> list[SimaiToken]:
    """将 token 序列缩放到统一的 subdiv

    缩放规则:
      - factor = target / original
      - 拍位: new_beat_idx = round(old_beat_idx * factor)
      - 多个 token 落到同一拍 → 合并; 拍间空隙 → 插入 rest
      - 持续时间 (dur) 中的数值同步缩放
      - 非整数倍时四舍五入，不保证完美往返

    Args:
        tokens: 原始 token 列表
        target_subdiv: 目标统一 subdiv (e.g., 4)

    Returns:
        缩放后的 token 列表
    """
    if not tokens:
        return []

    import copy

    result: list[SimaiToken] = []
    measure_start = 0
    new_measure = 0

    for i in range(len(tokens) + 1):
        # 检测小节边界: 遇到 BPM token 或 MEASURE token 或到末尾
        is_boundary = (i == len(tokens))
        if not is_boundary:
            t = tokens[i]
            is_boundary = (t.token_type in (SimaiTokenType.MEASURE, SimaiTokenType.BPM) and i > measure_start)

        if is_boundary:
            # 处理当前小节
            measure_tokens = tokens[measure_start:i]

            # 找出本小节的 subdiv (从 MEASURE token 或上一个 MEASURE)
            orig_subdiv = 4
            bpm_token = None
            note_tokens = []
            for mt in measure_tokens:
                if mt.token_type == SimaiTokenType.MEASURE:
                    orig_subdiv = int(mt.position)
                elif mt.token_type == SimaiTokenType.BPM:
                    bpm_token = mt
                else:
                    note_tokens.append(mt)

            if orig_subdiv == target_subdiv:
                # 无需缩放，直接保留
                for mt in measure_tokens:
                    mt = copy.copy(mt)
                    mt.measure = new_measure
                    if mt.token_type == SimaiTokenType.MEASURE:
                        mt.position = str(target_subdiv)
                        mt.subdiv = target_subdiv
                    result.append(mt)
                new_measure += 1
                measure_start = i
                continue

            factor = target_subdiv / orig_subdiv

            # 缩放拍位 + 合并/展开
            scaled_notes = _scale_beat_tokens(
                note_tokens, factor, target_subdiv, new_measure
            )

            # 重建小节
            if bpm_token:
                bpm_token = copy.copy(bpm_token)
                bpm_token.measure = new_measure
                bpm_token.subdiv = target_subdiv
                result.append(bpm_token)

            measure_tok = SimaiToken(
                SimaiTokenType.MEASURE, str(target_subdiv),
                measure=new_measure, beat=0.0, subdiv=target_subdiv,
                raw_text=f"{{{target_subdiv}}}",
            )
            result.append(measure_tok)
            result.extend(scaled_notes)

            new_measure += 1
            measure_start = i

    return result


def _scale_beat_tokens(
    note_tokens: list[SimaiToken],
    factor: float,
    target_subdiv: int,
    measure: int,
) -> list[SimaiToken]:
    """缩放一小节内的音符 token"""
    if not note_tokens:
        return []

    import math

    # 计算每个 token 的新拍位 idx
    # old beat (0~1 normalized) → old beat_idx = beat * old_subdiv
    # new beat_idx = round(old beat_idx * factor)
    # new beat = new_beat_idx / target_subdiv

    new_beats: dict[int, list[SimaiToken]] = {}  # beat_idx → tokens

    for t in note_tokens:
        old_beat_idx = round(t.beat * (target_subdiv / factor))  # = beat * old_subdiv
        # Actually, let me compute properly:
        # t.beat is normalized 0~1. old_beat_idx = t.beat * old_subdiv
        # new_beat_idx = round(old_beat_idx * factor)
        old_subdiv = target_subdiv / factor
        old_beat_idx = t.beat * old_subdiv
        new_beat_idx = round(old_beat_idx * factor)
        # Clamp to valid range
        new_beat_idx = max(0, min(new_beat_idx, target_subdiv - 1))

        if new_beat_idx not in new_beats:
            new_beats[new_beat_idx] = []
        new_beats[new_beat_idx].append(t)

    # 按拍位输出，空拍补 rest
    result = []
    for bi in range(target_subdiv):
        if bi in new_beats:
            merged = _merge_tokens_at_beat(new_beats[bi], factor, measure, bi, target_subdiv)
            result.extend(merged)
        # 空拍不输出 (保持紧凑), 除非需要占位
        # 注: 空拍跳过，让序列更紧凑; 如需保留时长信息，可在上层处理

    return result


def _merge_tokens_at_beat(
    tokens: list[SimaiToken],
    factor: float,
    measure: int,
    beat_idx: int,
    subdiv: int,
) -> list[SimaiToken]:
    """将同一拍位上的 token 重新分配时序

    不合并位置字符串，保持每个 token 独立以支持完美往返。
    仅在同类型 tap 且都无 break/ex 时可选合并。
    """
    import copy

    if len(tokens) == 1:
        t = copy.copy(tokens[0])
        t.measure = measure
        t.beat = beat_idx / subdiv
        t.subdiv = subdiv
        _scale_duration(t, factor)
        return [t]

    # 多个 token 同拍: 各自独立输出 (保留往返信息)
    result = []
    for t in tokens:
        t = copy.copy(t)
        t.measure = measure
        t.beat = beat_idx / subdiv
        t.subdiv = subdiv
        _scale_duration(t, factor)
        result.append(t)

    return result


def _scale_duration(token: SimaiToken, factor: float):
    """缩放 token 的持续时间参数"""
    dur = token.params.get("dur")
    if dur is None:
        return

    # dur 格式: X:Y (如 4:1), ##seconds (如 ##2.5), BPM#X:Y (如 190#16:5)
    # 只缩放 X:Y 中的 X (分子), 不缩放 ##seconds (绝对时间)

    if "##" in dur:
        # 绝对秒数, 不缩放
        return

    # 处理 X:Y 或 BPM#X:Y
    parts = dur.split("#")
    if len(parts) == 2:
        # BPM#X:Y → 只缩放 X:Y 部分
        bpm_part = parts[0]
        xy_part = parts[1]
        if ":" in xy_part:
            x, y = xy_part.split(":", 1)
            try:
                new_x = round(float(x) * factor)
                token.params["dur"] = f"{bpm_part}#{new_x}:{y}"
            except ValueError:
                pass
    elif ":" in dur:
        x, y = dur.split(":", 1)
        try:
            new_x = round(float(x) * factor)
            token.params["dur"] = f"{int(new_x)}:{y}"
        except ValueError:
            pass


# ============================================================
# Token 扁平化 — 消除参数, 每个变体独立 token
# ============================================================

def flatten_tokens(tokens: list[SimaiToken]) -> list[SimaiToken]:
    """将 token 序列扁平化：参数烘焙到 token 类型中，消除花括号参数

    转换规则:
      tap1           → tap1
      tap1{break}    → tap1b
      tap1{ex}       → tap1x
      tap1{break,ex} → tap1bx
      tap12          → tap12       (双押保持不变)
      hold1{dur:4:1} → hold1       (去掉 dur)
      hold1{dur:4:1,break} → hold1b
      slide1{path:-4,dur:8:1} → slide1  (去掉 path/dur)
      slide1{path:-4,dur:8:1,break} → slide1b
      touchB1{firework} → touchB1f
      holdC{dur:4:1,firework} → holdCf

    控制 token (bpm/measure) 保持原样。

    Returns:
        新的 token 列表 (副本), 每个 token.params == {}
    """
    import copy

    FLAG_ORDER = ["break", "ex", "firework"]
    FLAG_SUFFIX = {"break": "b", "ex": "x", "firework": "f"}

    result: list[SimaiToken] = []

    for t in tokens:
        # 控制 token 不变
        if t.token_type in (SimaiTokenType.BPM, SimaiTokenType.MEASURE):
            result.append(copy.copy(t))
            continue

        # 非音符 token 直接保留
        if not t.is_note:
            result.append(copy.copy(t))
            continue

        # 构建后缀: break→b, ex→x, firework→f
        suffix = ""
        for flag in FLAG_ORDER:
            if flag in t.params:
                suffix += FLAG_SUFFIX[flag]

        new_pos = t.position + suffix if suffix else t.position

        new_t = SimaiToken(
            token_type=t.token_type,
            position=new_pos,
            params={},  # 无参数
            measure=t.measure,
            beat=t.beat,
            subdiv=t.subdiv,
            raw_text=t.raw_text,
        )
        result.append(new_t)

    return result


def flatten_token_to_str(token: SimaiToken) -> str:
    """单个 token 扁平化为纯字符串 (无花括号)"""
    if not token.is_note:
        return token.to_string()

    FLAG_SUFFIX = {"break": "b", "ex": "x", "firework": "f"}
    suffix = ""
    for flag in ["break", "ex", "firework"]:
        if flag in token.params:
            suffix += FLAG_SUFFIX[flag]

    return f"{token.token_type.value}{token.position}{suffix}"


def split_params(
    tokens: list[SimaiToken],
) -> tuple[list[SimaiToken], np.ndarray, np.ndarray, np.ndarray]:
    """将 token 序列拆分为: 结构token + break/ex/firework 掩码

    - 结构 token: 去掉 break/ex/firework 参数，保留 hold/slide 的 dur/path
      例: hold1{dur:4:1,break} → hold1{dur:4:1}
          tap1{break,ex} → tap1
    - 返回三个 (N,) bool 数组, 标记每个 token 位置是否有 break/ex/firework

    Args:
        tokens: 原始 token 列表

    Returns:
        (struct_tokens, break_mask, ex_mask, firework_mask)
    """
    import copy
    import numpy as np

    FLAG_KEYS = ("break", "ex", "firework")
    note_indices = [i for i, t in enumerate(tokens) if t.is_note]
    n = len(note_indices)

    break_mask = np.zeros(n, dtype=bool)
    ex_mask = np.zeros(n, dtype=bool)
    firework_mask = np.zeros(n, dtype=bool)

    result: list[SimaiToken] = []

    for t in tokens:
        if not t.is_note:
            result.append(copy.copy(t))
            continue

        # 记录 flags
        idx = len([x for x in result if x.is_note])
        if t.has_break:
            break_mask[idx] = True
        if t.has_ex:
            ex_mask[idx] = True
        if t.has_firework:
            firework_mask[idx] = True

        # 剥离 flags 参数
        new_params = {
            k: v for k, v in t.params.items()
            if k not in FLAG_KEYS
        }

        new_t = SimaiToken(
            token_type=t.token_type,
            position=t.position,
            params=new_params,
            measure=t.measure,
            beat=t.beat,
            subdiv=t.subdiv,
            raw_text=t.raw_text,
        )
        result.append(new_t)

    return result, break_mask, ex_mask, firework_mask

