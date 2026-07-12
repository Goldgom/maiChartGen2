"""
Simai Chart Parser - 解析 maimai 谱面文件 (maidata.txt)

谱面格式说明:
  头部: &key=value 格式的元信息
  谱面: &inote_N= 后跟音符数据

使用 SimaiTokenizer 进行 token 化解析。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from SimaiToken import (
    SimaiToken, SimaiTokenType, SimaiTokenizer,
    tokenize_notes, tokens_summary, rescale_subdiv,
)


@dataclass
class SimaiChart:
    """单个难度的谱面"""
    level: float = 0.0
    designer: str = ""
    difficulty: int = 0
    raw_notes: str = ""
    tokens: list[SimaiToken] = field(default_factory=list)
    bpm: float = 120.0

    DIFFICULTY_NAMES: dict[int, str] = field(default_factory=lambda: {
        1: "Easy", 2: "Basic", 3: "Advanced",
        4: "Expert", 5: "Master", 6: "Re:Master", 7: "UTAGE",
    }, repr=False, init=False)

    @property
    def difficulty_name(self) -> str:
        return self.DIFFICULTY_NAMES.get(self.difficulty, f"Unknown({self.difficulty})")

    @property
    def total_notes(self) -> int:
        return sum(1 for t in self.tokens if t.is_note)

    @property
    def tap_count(self) -> int:
        return sum(1 for t in self.tokens if t.token_type == SimaiTokenType.TAP)

    @property
    def hold_count(self) -> int:
        return sum(1 for t in self.tokens if t.token_type == SimaiTokenType.HOLD)

    @property
    def slide_count(self) -> int:
        return sum(1 for t in self.tokens if t.token_type == SimaiTokenType.SLIDE)

    @property
    def touch_count(self) -> int:
        return sum(1 for t in self.tokens if t.token_type == SimaiTokenType.TOUCH)

    @property
    def break_count(self) -> int:
        return sum(1 for t in self.tokens if t.has_break)

    @property
    def ex_count(self) -> int:
        return sum(1 for t in self.tokens if t.has_ex)

    @property
    def firework_count(self) -> int:
        return sum(1 for t in self.tokens if t.has_firework)

    def tokenize(self, bpm: float = 120.0) -> None:
        """使用 SimaiTokenizer 解析 raw_notes"""
        self.bpm = bpm
        self.tokens = tokenize_notes(self.raw_notes, bpm)

    def rescale(self, target_subdiv: int = 4) -> "SimaiChart":
        """返回 subdiv 统一缩放后的新 SimaiChart"""
        import copy
        new_chart = copy.copy(self)
        new_chart.tokens = rescale_subdiv(self.tokens, target_subdiv)
        return new_chart

    def summary(self) -> str:
        return (f"[{self.difficulty_name}] Lv.{self.level} "
                f"TAP:{self.tap_count} HLD:{self.hold_count} "
                f"SLD:{self.slide_count} TCH:{self.touch_count} "
                f"BRK:{self.break_count} EX:{self.ex_count} "
                f"TOTAL:{self.total_notes}")

    def __repr__(self) -> str:
        return (f"SimaiChart(diff={self.difficulty}({self.difficulty_name}), "
                f"level={self.level}, notes={self.total_notes})")


@dataclass
class SimaiData:
    """一首歌的完整谱面数据，包含所有难度"""
    # 元信息
    title: str = ""
    artist: str = ""
    artist_id: str = ""
    whole_bpm: float = 120.0
    genre: str = ""
    genre_id: str = ""
    version: str = ""
    cabinet: str = ""
    short_id: str = ""
    description: str = ""

    # 谱面数据 (key=difficulty_number 1-6)
    charts: dict[int, SimaiChart] = field(default_factory=dict)

    # 文件路径
    source_path: Optional[Path] = None

    DIFFICULTY_NAMES: dict[int, str] = field(default_factory=lambda: {
        1: "Easy",
        2: "Basic",
        3: "Advanced",
        4: "Expert",
        5: "Master",
        6: "Re:Master",
        7: "UTAGE",
    }, repr=False, init=False)

    @classmethod
    def load(cls, maidata_path: str | Path) -> "SimaiData":
        """从 maidata.txt 文件加载谱面数据"""
        path = Path(maidata_path)
        if not path.exists():
            raise FileNotFoundError(f"谱面文件不存在: {path}")

        raw_text = path.read_text(encoding="utf-8")
        return cls.parse(raw_text, source_path=path)

    @classmethod
    def parse(cls, raw_text: str, source_path: Optional[Path] = None,
              target_subdiv: int = 0) -> "SimaiData":
        """解析 maidata.txt 文本内容

        Args:
            raw_text: maidata.txt 文本
            source_path: 文件路径
            target_subdiv: 统一目标 subdiv (0=不缩放, 如 4)
        """
        data = cls(source_path=source_path)

        # 收集所有行
        lines = raw_text.strip().split("\n")

        # 第一遍: 解析头部元信息
        for line in lines:
            line = line.strip()
            if line.startswith("&inote_"):
                break  # 到谱面部分了
            if not line.startswith("&"):
                continue

            key, _, value = line[1:].partition("=")
            key = key.strip()
            value = value.strip()

            if key == "title":
                data.title = value
            elif key == "artist":
                data.artist = value
            elif key == "artistid":
                data.artist_id = value
            elif key == "wholebpm":
                try:
                    data.whole_bpm = float(value)
                except ValueError:
                    pass
            elif key == "genre":
                data.genre = value
            elif key == "genreid":
                data.genre_id = value
            elif key == "version":
                data.version = value
            elif key == "cabinet":
                data.cabinet = value
            elif key == "shortid":
                data.short_id = value
            elif key == "des":
                data.description = value

            # 处理 lv_N, des_N
            lv_match = re.match(r'lv_(\d+)', key)
            if lv_match:
                diff = int(lv_match.group(1))
                if diff not in data.charts:
                    data.charts[diff] = SimaiChart(difficulty=diff, bpm=data.whole_bpm)
                try:
                    data.charts[diff].level = float(value) if value else 0.0
                except ValueError:
                    data.charts[diff].level = 0.0

            des_match = re.match(r'des_(\d+)', key)
            if des_match:
                diff = int(des_match.group(1))
                if diff not in data.charts:
                    data.charts[diff] = SimaiChart(difficulty=diff, bpm=data.whole_bpm)
                data.charts[diff].designer = value

        # 第二遍: 解析谱面数据
        current_diff = None
        note_buffer: list[str] = []

        for line in lines:
            line_stripped = line.strip()

            inote_match = re.match(r'&inote_(\d+)=(.*)', line_stripped)
            if inote_match:
                # 保存上一个难度的数据
                if current_diff is not None and note_buffer:
                    data._finalize_chart(current_diff, note_buffer)
                    note_buffer = []

                current_diff = int(inote_match.group(1))
                rest = inote_match.group(2).strip()
                if rest:
                    note_buffer.append(rest)
                continue

            # 收集音符数据 (在当前 inote section中)
            if current_diff is not None:
                if line_stripped.startswith("&"):
                    # 下一个 section 开始
                    data._finalize_chart(current_diff, note_buffer)
                    note_buffer = []
                    current_diff = None
                else:
                    note_buffer.append(line.rstrip())

        # 处理最后一个难度
        if current_diff is not None and note_buffer:
            data._finalize_chart(current_diff, note_buffer)

        # 解析每个谱面的音符 (使用 tokenizer)
        for chart in data.charts.values():
            chart.tokenize(bpm=data.whole_bpm)

        # 统一 subdiv 缩放
        if target_subdiv > 0:
            for diff in list(data.charts.keys()):
                data.charts[diff] = data.charts[diff].rescale(target_subdiv)

        return data

    def _finalize_chart(self, diff: int, note_buffer: list[str]) -> None:
        """将收集到的音符行保存到对应难度的图表中"""
        if diff not in self.charts:
            self.charts[diff] = SimaiChart(difficulty=diff, bpm=self.whole_bpm)
        self.charts[diff].raw_notes = "\n".join(note_buffer)

    @property
    def available_difficulties(self) -> list[int]:
        """返回有数据的难度列表"""
        return sorted([d for d in self.charts.keys() if self.charts[d].total_notes > 0])

    def get_chart(self, difficulty: int) -> Optional[SimaiChart]:
        """获取指定难度的谱面"""
        return self.charts.get(difficulty)

    def summary(self) -> str:
        """返回谱面摘要"""
        lines = [f"Title: {self.title}", f"Artist: {self.artist}",
                 f"BPM: {self.whole_bpm}", f"Genre: {self.genre}",
                 f"Difficulties: {len(self.available_difficulties)}"]
        for diff in self.available_difficulties:
            chart = self.charts[diff]
            lines.append(
                f"  [{chart.difficulty_name}] Lv.{chart.level} "
                f"- TAP:{chart.tap_count} HLD:{chart.hold_count} "
                f"SLD:{chart.slide_count} TCH:{chart.touch_count} "
                f"BRK:{chart.break_count} EX:{chart.ex_count} "
                f"TOTAL:{chart.total_notes}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SimaiData(title={self.title!r}, charts={len(self.charts)})"


# ============ 便捷函数 ============

def load_simai(maidata_path: str | Path) -> SimaiData:
    """便捷函数: 从 maidata.txt 加载谱面"""
    return SimaiData.load(maidata_path)


def parse_simai(raw_text: str) -> SimaiData:
    """便捷函数: 从文本解析谱面"""
    return SimaiData.parse(raw_text)


# ============ Slide 路径+时序 提取 ============

def _split_slide_path(path: str) -> list[str]:
    """分解 slide 路径字符串为 segment 列表

    例: "-4" → ["-4"], ">5-8" → [">5", "-8"], ">8*V28" → [">8", "*V28"]
    """
    segments = re.findall(
        r'\*?(?:pp|qq|[-><^vVpqszw])\d+',
        path
    )
    return segments if segments else [path]


def extract_slide_tokens_from_chart(chart: SimaiChart) -> list[str]:
    """从一个难度的谱面中提取所有 slide path+timing token 字符串

    每个 slide 的路径被分解为 segment, 每个 segment 与 timing 组合:
      "-4[8:1]", ">5[2:1]", "*V28[4:3]", ...

    返回: token 字符串列表 (按谱面中出现顺序)
    """
    tokens: list[str] = []
    for t in chart.tokens:
        if t.token_type != SimaiTokenType.SLIDE:
            continue
        path = t.params.get("path", "")
        timing = t.params.get("dur", "")
        if not path:
            continue
        for seg in _split_slide_path(path):
            token_str = f"{seg}[{timing}]" if timing else seg
            tokens.append(token_str)
    return tokens


def extract_slide_tokens_from_data(data: SimaiData,
                                    difficulty: int | None = None) -> list[str]:
    """从 SimaiData 中提取 slide path+timing token 字符串

    Args:
        data: 已解析的谱面数据
        difficulty: 指定难度 (1-7), None 表示所有难度

    Returns:
        token 字符串列表
    """
    tokens: list[str] = []
    charts = [data.charts[difficulty]] if difficulty is not None else data.charts.values()
    for chart in charts:
        tokens.extend(extract_slide_tokens_from_chart(chart))
    return tokens


def extract_slide_tokens_from_file(maidata_path: str | Path,
                                    difficulty: int | None = None) -> list[str]:
    """从 maidata.txt 文件中提取 slide path+timing token 字符串

    Args:
        maidata_path: maidata.txt 文件路径
        difficulty: 指定难度 (1-7), None 表示所有难度

    Returns:
        token 字符串列表
    """
    data = SimaiData.load(maidata_path)
    return extract_slide_tokens_from_data(data, difficulty)


# 难度名称 → 难度编号 映射
DIFFICULTY_NAME_TO_NUM: dict[str, int] = {
    "easy": 1, "basic": 2, "advanced": 3,
    "expert": 4, "master": 5, "re:master": 6, "remaster": 6, "utage": 7,
}
