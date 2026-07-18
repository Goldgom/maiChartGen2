"""
PreProcess — 第一阶段模型训练预处理管线

产出每个谱面一个 .npz 文件，包含:
  - audio_tokens:    (T, C) int32   EnCodec 音频 token
  - beat_signal:     (T, 2) float32 节拍/重拍信号 (0/1)
  - chart_tokens:    (T,)  int32    扁平化谱面 token ID (0=空)
  - chart_vocab:     dict           token字符串 → ID 映射
  - metadata:        dict           标题/BPM/难度等信息

时间对齐到统一帧率网格 (默认 75Hz = EnCodec 24kHz 帧率)。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================
# PreprocessResult
# ============================================================

@dataclass
class PreprocessResult:
    """单个谱面的预处理结果"""

    audio_tokens: np.ndarray       # (T, C) int32
    beat_signal: np.ndarray        # (T, 2) float32 [beat, downbeat]
    chart_tokens: np.ndarray       # (T,) int32, 0=no note
    chart_vocab: dict[str, int]    # token_str → id
    metadata: dict                 # title, artist, bpm, difficulty, etc.
    frame_rate: float = 75.0
    frame_objects: dict = None      # frame index string -> list of per-object labels
    # Stage-2 用: break/ex/firework 掩码
    break_mask: np.ndarray = None   # (T,) bool
    ex_mask: np.ndarray = None      # (T,) bool
    firework_mask: np.ndarray = None  # (T,) bool
    object_mask: np.ndarray = None
    # Stage 2-3 训练目标
    hold_dur_targets: np.ndarray = None   # (T,) int, 0=非hold, >0=dur bin ID
    slide_path_targets: np.ndarray = None  # (T,) int, 0=非slide, >0=path segment ID
    # 标签
    tag_ids: np.ndarray = None
    tag_vocab: dict[str, int] = None
    slide_vocab: dict[str, int] = None

    def __post_init__(self):
        t = self.chart_tokens.shape[0]
        if self.break_mask is None:
            self.break_mask = np.zeros(t, dtype=bool)
        if self.ex_mask is None:
            self.ex_mask = np.zeros(t, dtype=bool)
        if self.firework_mask is None:
            self.firework_mask = np.zeros(t, dtype=bool)
        if self.object_mask is None:
            self.object_mask = np.zeros(t, dtype=bool)
        if self.hold_dur_targets is None:
            self.hold_dur_targets = np.zeros(t, dtype=np.int32)
        if self.slide_path_targets is None:
            self.slide_path_targets = np.zeros(t, dtype=np.int32)
        if self.tag_ids is None:
            self.tag_ids = np.array([], dtype=np.int32)
        if self.tag_vocab is None:
            self.tag_vocab = {}
        if self.slide_vocab is None:
            self.slide_vocab = {}
        if self.frame_objects is None:
            self.frame_objects = {}

    @property
    def num_frames(self) -> int:
        return self.audio_tokens.shape[0]

    @property
    def num_audio_codebooks(self) -> int:
        return self.audio_tokens.shape[1]

    def summary(self) -> str:
        notes = int((self.chart_tokens > 0).sum())
        beats = int(self.beat_signal[:, 0].sum())
        return (
            f"PreprocessResult(frames={self.num_frames}, "
            f"audio=({self.num_frames},{self.num_audio_codebooks}), "
            f"beats={beats}, chart_notes={notes}, vocab={len(self.chart_vocab)})"
        )

    def save(self, path: str | Path):
        """保存为 .npz 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            audio_tokens=self.audio_tokens,
            beat_signal=self.beat_signal,
            chart_tokens=self.chart_tokens,
            break_mask=self.break_mask,
            ex_mask=self.ex_mask,
            firework_mask=self.firework_mask,
            object_mask=self.object_mask,
            hold_dur_targets=self.hold_dur_targets,
            slide_path_targets=self.slide_path_targets,
            tag_ids=self.tag_ids,
            frame_rate=np.array([self.frame_rate], dtype=np.float32),
        )
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "chart_vocab": self.chart_vocab,
                "tag_vocab": self.tag_vocab,
                "slide_vocab": getattr(self, "slide_vocab", {}),
                "frame_objects": self.frame_objects,
                "metadata": self.metadata,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessResult":
        """从 .npz + .json 加载"""
        path = Path(path)
        data = np.load(path)
        meta_path = path.with_suffix(".json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            audio_tokens=data["audio_tokens"],
            beat_signal=data["beat_signal"],
            chart_tokens=data["chart_tokens"],
            break_mask=data.get("break_mask", np.zeros(data["chart_tokens"].shape[0], dtype=bool)),
            ex_mask=data.get("ex_mask", np.zeros(data["chart_tokens"].shape[0], dtype=bool)),
            firework_mask=data.get("firework_mask", np.zeros(data["chart_tokens"].shape[0], dtype=bool)),
            object_mask=data.get("object_mask", np.zeros(data["chart_tokens"].shape[0], dtype=bool)),
            hold_dur_targets=data.get("hold_dur_targets", np.zeros(data["chart_tokens"].shape[0], dtype=np.int32)),
            slide_path_targets=data.get("slide_path_targets", np.zeros(data["chart_tokens"].shape[0], dtype=np.int32)),
            tag_ids=data.get("tag_ids", np.array([], dtype=np.int32)),
            tag_vocab=meta.get("tag_vocab", {}),
            slide_vocab=meta.get("slide_vocab", {}),
            frame_objects=meta.get("frame_objects", {}),
            chart_vocab=meta["chart_vocab"],
            metadata=meta["metadata"],
            frame_rate=float(data["frame_rate"][0]),
        )

    def __repr__(self) -> str:
        return self.summary()


# ============================================================
# Preprocessor
# ============================================================

class Preprocessor:
    """第一阶段预处理管线"""

    def __init__(self, config=None):
        """
        Args:
            config: Config 对象 (通过 load_config() 获取), 或 None 使用默认
        """
        if config is None:
            from Config import load_config
            config = load_config()

        self.cfg = config
        self.output_dir = Path(config.preprocess.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_dir = Path(getattr(config.paths, "vocab_dir", "vocab"))

        # 全局 vocab (跨所有谱面)
        self.global_vocab: dict[str, int] = {}
        self._next_vocab_id = 1  # 0 保留给 "无音符"

        # 全局标签 vocab
        self.global_tag_vocab: dict[str, int] = {}
        self._next_tag_id = 0
        self.global_slide_vocab: dict[str, int] = {"<PAD>": 0}
        self._next_slide_id = 1

    # ---- 主流程 ----

    def process_all(self, dataset_dir: Optional[str] = None) -> list[Path]:
        """处理数据集中的所有谱面，返回输出文件路径列表"""
        from SimaiPaser import SimaiData

        ds_dir = Path(dataset_dir or self.cfg.paths.datasets_dir)
        chart_dirs = sorted(
            [d for d in ds_dir.iterdir() if d.is_dir() and (d / "maidata.txt").exists()],
            key=lambda x: (int(x.name) if x.name.isdigit() else 99999, x.name),
        )

        max_charts = self.cfg.preprocess.max_charts
        if max_charts > 0:
            chart_dirs = chart_dirs[:max_charts]

        output_paths = []
        total = len(chart_dirs)
        t_start = time.time()

        from tqdm import tqdm
        pbar = tqdm(enumerate(chart_dirs), total=total, desc="预处理", unit="chart",
                     ncols=100, mininterval=1.0)

        for idx, chart_dir in pbar:
            sid = chart_dir.name

            # 检查是否已有缓存 (任意难度文件存在即跳过)
            if self.cfg.preprocess.skip_existing:
                existing_for_sid = list(self.output_dir.glob(f"{sid}_*.npz"))
                if existing_for_sid:
                    for ep in existing_for_sid:
                        meta_path = ep.with_suffix(".json")
                        if meta_path.exists():
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            for tok, tid in meta.get("chart_vocab", {}).items():
                                if tok not in self.global_vocab:
                                    self.global_vocab[tok] = self._next_vocab_id
                                    self._next_vocab_id = max(self._next_vocab_id, tid + 1)
                            for tok, tid in meta.get("slide_vocab", {}).items():
                                if tok not in self.global_slide_vocab:
                                    self.global_slide_vocab[tok] = tid
                                    self._next_slide_id = max(self._next_slide_id, tid + 1)
                        output_paths.append(ep)
                    pbar.set_postfix_str(f"{sid} cached x{len(existing_for_sid)}")
                    continue

            try:
                results = self.process_one(chart_dir)
                for result in results:
                    diff_name = result.metadata.get("difficulty_name", "Unknown")
                    # 文件名不能有冒号 (Windows 限制)
                    safe_diff = diff_name.replace(":", "")
                    out_path = self.output_dir / f"{sid}_{safe_diff}.npz"
                    result.save(out_path)
                    output_paths.append(out_path)
                elapsed = time.time() - t_start
                rate = (idx + 1) / max(elapsed, 0.1)
                eta = (total - idx - 1) / max(rate, 0.01)
                pbar.set_postfix_str(
                    f"{sid} x{len(results)}diffs "
                    + (f"ETA:{eta/60:.1f}m" if eta > 60 else f"ETA:{eta:.0f}s")
                )
            except Exception as e:
                pbar.write(f"[{sid}] ERROR: {e}")

        pbar.close()

        # 保存全局 vocab
        if self.global_vocab:
            vocab_path = self.output_dir / "vocab.json"
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(self.global_vocab, f, ensure_ascii=False, indent=2)
            print(f"\n全局 chart vocab: {len(self.global_vocab)} tokens → {vocab_path}")

        # 保存标签 vocab
        if self.global_tag_vocab:
            tag_path = self.output_dir / self.cfg.tags.tag_vocab_path.split("/")[-1]
            with open(tag_path, "w", encoding="utf-8") as f:
                json.dump(self.global_tag_vocab, f, ensure_ascii=False, indent=2)
            print(f"全局 tag vocab: {len(self.global_tag_vocab)} tags → {tag_path}")

        print(f"\n完成: {len(output_paths)} 文件 / {total} 谱面目录, 耗时 {time.time()-t_start:.0f}s")
        if self.global_slide_vocab:
            slide_path = self.output_dir / "slide_vocab.json"
            with open(slide_path, "w", encoding="utf-8") as f:
                json.dump(self.global_slide_vocab, f, ensure_ascii=False, indent=2)
            print(f"Global slide vocab: {len(self.global_slide_vocab)} paths -> {slide_path}")

        self._sync_vocab_dir()
        return output_paths

    def _sync_vocab_dir(self) -> None:
        self.vocab_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in (
            "vocab.json",
            "tag_vocab.json",
            "slide_vocab.json",
            "slide_vocab_with_timing.json",
            "slide_path_timing_map.json",
        ):
            src = self.output_dir / name
            if not src.exists():
                continue
            dst = self.vocab_dir / name
            if src.resolve() == dst.resolve():
                continue
            shutil.copy2(src, dst)
            copied.append(dst)
        if copied:
            print(f"Synced vocab files -> {self.vocab_dir} ({len(copied)} files)")

    def process_one(self, chart_dir: Path) -> list[PreprocessResult]:
        """处理单个谱面目录中的所有难度，返回结果列表"""
        from SimaiPaser import SimaiData
        from SimaiToken import flatten_tokens
        from AudioTokenizer import AudioTokenizer
        from BeatTokenizer import BeatTokenizer

        maidata_path = chart_dir / "maidata.txt"
        audio_path = chart_dir / "track.mp3"

        # 1. 加载谱面 (所有难度)
        raw = maidata_path.read_text(encoding="utf-8")
        data = SimaiData.parse(raw, target_subdiv=self.cfg.chart.target_subdiv)

        diffs = data.available_difficulties
        if not diffs:
            raise ValueError(f"无有效难度: {chart_dir}")

        # 2. 音频 token (共享)
        audio_tok = AudioTokenizer(
            num_codebooks=self.cfg.preprocess.audio_codebooks,
            device=self.cfg.audio.device,
            local_path=self.cfg.audio.premodel_path or None,
        )
        audio_data = audio_tok.encode_file(str(audio_path))

        # 3. 节拍检测 (共享)
        beat_tok = BeatTokenizer(
            method=self.cfg.preprocess.beat_method,
            target_bpm=data.whole_bpm,
            beat_this_ckpt="premodels/beatthis.ckpt",
        )
        beat_list = beat_tok.analyse(str(audio_path))

        # 4. 帧率
        frame_rate = self.cfg.preprocess.frame_rate
        if frame_rate <= 0:
            frame_rate = audio_data.frame_rate

        # 5. 逐个难度处理
        results = []
        for diff in sorted(diffs):
            chart = data.charts[diff]
            tag_ids, tag_vocab_local = self._extract_tags(data, chart, chart_dir)
            result = self._align_v2(audio_data, beat_list, chart, frame_rate, data, tag_ids, tag_vocab_local)
            results.append(result)

        return results

    # ---- 时间对齐 ----

    def _align(
        self,
        audio_data,
        beat_list,
        chart,
        frame_rate: float,
        simai_data,
        tag_ids: list[int],
        tag_vocab_local: dict[str, int],
    ) -> PreprocessResult:
        """将音频/节拍/谱面对齐到统一帧率网格"""
        from SimaiToken import flatten_tokens

        duration = audio_data.duration
        num_frames = max(1, round(duration * frame_rate))
        num_codebooks = audio_data.num_codebooks

        # --- 音频 token: 重采样到目标帧率 ---
        audio_tokens = audio_data.tokens  # (src_frames, C)
        src_fr = audio_data.frame_rate
        if abs(src_fr - frame_rate) < 0.1:
            audio_aligned = audio_tokens[:num_frames]
        else:
            # 线性插值重采样
            src_times = np.arange(audio_tokens.shape[0]) / src_fr
            tgt_times = np.arange(num_frames) / frame_rate
            audio_aligned = np.zeros((num_frames, num_codebooks), dtype=np.int32)
            for c in range(num_codebooks):
                audio_aligned[:, c] = np.round(
                    np.interp(tgt_times, src_times, audio_tokens[:, c].astype(np.float32))
                ).astype(np.int32)

        # --- 节拍信号: 映射到帧 ---
        beat_signal = np.zeros((num_frames, 2), dtype=np.float32)
        for b in beat_list.beats:
            frame_idx = round(b.time * frame_rate)
            if 0 <= frame_idx < num_frames:
                beat_signal[frame_idx, 0] = max(beat_signal[frame_idx, 0], 0.5)
                if b.is_downbeat:
                    beat_signal[frame_idx, 1] = 1.0

        # 二值化 (默认)
        if self.cfg.preprocess.beat_as_binary:
            beat_signal = (beat_signal > 0.3).astype(np.float32)

        # --- 谱面 token: 拆分参数(先) + 扁平化(后) + 映射到帧 ---
        from SimaiToken import flatten_tokens, split_params

        # 1. 先拆分 break/ex/firework 掩码 (在 flatten 之前，params 还在)
        struct_tokens, break_mask_full, ex_mask_full, fw_mask_full = split_params(chart.tokens)
        # 2. 再扁平化 (去掉 dur/path，得到纯 token 类型)
        flat_tokens = flatten_tokens(struct_tokens)
        note_tokens = [t for t in flat_tokens if t.is_note]

        chart_ids = np.zeros(num_frames, dtype=np.int32)
        brk_mask = np.zeros(num_frames, dtype=bool)
        ex_mask_arr = np.zeros(num_frames, dtype=bool)
        fw_mask_arr = np.zeros(num_frames, dtype=bool)

        for i, t in enumerate(note_tokens):
            beat_start_time = self._beat_group_to_time(t.measure, beat_list)
            # In SimaiTokenizer, {n} means the next BPM beat is split into n
            # slots. t.beat is already normalized inside that one BPM beat.
            note_time = beat_start_time + t.beat * self._beat_duration(beat_list)
            frame_idx = round(note_time * frame_rate)
            if 0 <= frame_idx < num_frames:
                tok_str = t.to_string()
                if tok_str not in self.global_vocab:
                    self.global_vocab[tok_str] = self._next_vocab_id
                    self._next_vocab_id += 1
                chart_ids[frame_idx] = self.global_vocab[tok_str]
                brk_mask[frame_idx] = break_mask_full[i]
                ex_mask_arr[frame_idx] = ex_mask_full[i]
                fw_mask_arr[frame_idx] = fw_mask_full[i]

        return PreprocessResult(
            audio_tokens=audio_aligned,
            beat_signal=beat_signal,
            chart_tokens=chart_ids,
            break_mask=brk_mask,
            ex_mask=ex_mask_arr,
            firework_mask=fw_mask_arr,
            tag_ids=np.array(tag_ids, dtype=np.int32),
            tag_vocab=tag_vocab_local,
            chart_vocab=dict(self.global_vocab),
            metadata={
                "title": simai_data.title,
                "artist": simai_data.artist,
                "bpm": simai_data.whole_bpm,
                "chart_id": simai_data.short_id,
                "difficulty": chart.difficulty,
                "difficulty_name": chart.difficulty_name,
                "level": chart.level,
                "duration": duration,
                "frame_rate": frame_rate,
                "num_frames": num_frames,
            },
            frame_rate=frame_rate,
        )

    @staticmethod
    def _beat_duration(beat_list) -> float:
        """Return one BPM beat duration in seconds."""
        if beat_list.bpm > 0:
            return 60.0 / beat_list.bpm
        return 0.5  # fallback: 120 BPM

    @staticmethod
    def _beat_group_to_time(beat_group: int, beat_list) -> float:
        """Estimate the start time of a Simai {n} beat group.

        SimaiTokenizer currently stores the line/group index in token.measure,
        but in simai syntax used here each {n} group represents one BPM beat,
        not a full measure. Prefer detected beat times and fall back to a
        uniform BPM grid.
        """
        if 0 <= beat_group < len(beat_list.beats):
            return beat_list.beats[beat_group].time
        return beat_group * Preprocessor._beat_duration(beat_list)

    def _align_v2(
        self,
        audio_data,
        beat_list,
        chart,
        frame_rate: float,
        simai_data,
        tag_ids: list[int],
        tag_vocab_local: dict[str, int],
    ) -> PreprocessResult:
        """Align chart data using combined frame tokens plus JSON object labels."""
        duration = audio_data.duration
        num_frames = max(1, round(duration * frame_rate))
        num_codebooks = audio_data.num_codebooks

        audio_tokens = audio_data.tokens
        src_fr = audio_data.frame_rate
        if abs(src_fr - frame_rate) < 0.1:
            audio_aligned = audio_tokens[:num_frames]
        else:
            src_times = np.arange(audio_tokens.shape[0]) / src_fr
            tgt_times = np.arange(num_frames) / frame_rate
            audio_aligned = np.zeros((num_frames, num_codebooks), dtype=np.int32)
            for c in range(num_codebooks):
                audio_aligned[:, c] = np.round(
                    np.interp(tgt_times, src_times, audio_tokens[:, c].astype(np.float32))
                ).astype(np.int32)

        beat_signal = np.zeros((num_frames, 2), dtype=np.float32)
        for b in beat_list.beats:
            frame_idx = round(b.time * frame_rate)
            if 0 <= frame_idx < num_frames:
                beat_signal[frame_idx, 0] = max(beat_signal[frame_idx, 0], 0.5)
                if b.is_downbeat:
                    beat_signal[frame_idx, 1] = 1.0
        if self.cfg.preprocess.beat_as_binary:
            beat_signal = (beat_signal > 0.3).astype(np.float32)

        chart_ids = np.zeros(num_frames, dtype=np.int32)
        max_object_slots = getattr(getattr(self.cfg, "stage_model", None), "max_object_slots", 16)
        brk_mask = np.zeros((num_frames, max_object_slots), dtype=bool)
        ex_mask_arr = np.zeros((num_frames, max_object_slots), dtype=bool)
        fw_mask_arr = np.zeros((num_frames, max_object_slots), dtype=bool)
        object_mask = np.zeros((num_frames, max_object_slots), dtype=bool)
        max_hold_slots = getattr(getattr(self.cfg, "stage_model", None), "max_hold_slots", 8)
        max_slide_slots = getattr(getattr(self.cfg, "stage_model", None), "max_slide_slots", 8)
        hold_dur_targets = np.zeros((num_frames, max_hold_slots), dtype=np.int32)
        slide_path_targets = np.zeros((num_frames, max_slide_slots), dtype=np.int32)
        frame_buckets: dict[int, list[tuple[tuple, str, dict]]] = {}

        for t in chart.tokens:
            if not t.is_note:
                continue
            beat_start_time = self._beat_group_to_time(t.measure, beat_list)
            note_time = beat_start_time + t.beat * self._beat_duration(beat_list)
            frame_idx = round(note_time * frame_rate)
            if 0 <= frame_idx < num_frames:
                obj = self._frame_object(t)
                frame_buckets.setdefault(frame_idx, []).append(
                    (self._object_sort_key(obj), self._structure_token(t), obj)
                )

        frame_objects: dict[str, list[dict]] = {}
        for frame_idx, items in frame_buckets.items():
            items.sort(key=lambda x: x[0])
            combo_token = "+".join(item[1] for item in items)
            if combo_token not in self.global_vocab:
                self.global_vocab[combo_token] = self._next_vocab_id
                self._next_vocab_id += 1
            chart_ids[frame_idx] = self.global_vocab[combo_token]

            objects = [item[2] for item in items]
            frame_objects[str(frame_idx)] = objects
            for slot, obj in enumerate(objects[:max_object_slots]):
                object_mask[frame_idx, slot] = True
                brk_mask[frame_idx, slot] = obj["break"]
                ex_mask_arr[frame_idx, slot] = obj["ex"]
                fw_mask_arr[frame_idx, slot] = obj["firework"]
            hold_objects = [obj for obj in objects if obj["type"] == "hold"]
            for slot, obj in enumerate(hold_objects[:max_hold_slots]):
                hold_dur_targets[frame_idx, slot] = self._encode_dur(
                    obj["dur"],
                    max_bins=getattr(getattr(self.cfg, "stage_model", None), "hold_dur_bins", 64),
                )
            slide_objects = [obj for obj in objects if obj["type"] == "slide"]
            for slot, obj in enumerate(slide_objects[:max_slide_slots]):
                slide_path_targets[frame_idx, slot] = self._encode_slide_path(obj["path"])

        return PreprocessResult(
            audio_tokens=audio_aligned,
            beat_signal=beat_signal,
            chart_tokens=chart_ids,
            break_mask=brk_mask,
            ex_mask=ex_mask_arr,
            firework_mask=fw_mask_arr,
            object_mask=object_mask,
            hold_dur_targets=hold_dur_targets,
            slide_path_targets=slide_path_targets,
            tag_ids=np.array(tag_ids, dtype=np.int32),
            tag_vocab=tag_vocab_local,
            slide_vocab=dict(self.global_slide_vocab),
            frame_objects=frame_objects,
            chart_vocab=dict(self.global_vocab),
            metadata={
                "title": simai_data.title,
                "artist": simai_data.artist,
                "bpm": simai_data.whole_bpm,
                "chart_id": simai_data.short_id,
                "difficulty": chart.difficulty,
                "difficulty_name": chart.difficulty_name,
                "level": chart.level,
                "duration": duration,
                "frame_rate": frame_rate,
                "num_frames": num_frames,
            },
            frame_rate=frame_rate,
        )

    @staticmethod
    def _structure_token(token) -> str:
        return f"{token.token_type.value}{token.position}"

    @staticmethod
    def _frame_object(token) -> dict:
        return {
            "type": token.token_type.value,
            "pos": token.position,
            "dur": token.params.get("dur", ""),
            "path": token.params.get("path", ""),
            "break": bool(token.has_break),
            "ex": bool(token.has_ex),
            "firework": bool(token.has_firework),
            "raw": token.raw_text,
        }

    @staticmethod
    def _encode_dur(dur_str: str, max_bins: int = 64) -> int:
        """Encode a simai X:Y duration into a 1-based duration bin."""
        if not dur_str:
            return 1
        if dur_str.startswith("##"):
            try:
                seconds = float(dur_str[2:])
                bin_id = min(int(np.log2(max(seconds, 0.0625)) + 5), max_bins - 1) + 1
                return max(1, bin_id)
            except ValueError:
                return 1
        if "#" in dur_str:
            dur_str = dur_str.split("#", 1)[1]
        if ":" not in dur_str:
            return 1
        try:
            x, y = dur_str.split(":", 1)
            val = float(x) / float(y)
            if val <= 0:
                return 1
            bin_id = min(int(np.log2(max(val, 0.0625)) + 5), max_bins - 1) + 1
            return max(1, bin_id)
        except (ValueError, ZeroDivisionError):
            return 1

    def _encode_slide_path(self, path: str) -> int:
        """Encode a complete slide path string with an independent vocab."""
        key = path or "<EMPTY>"
        if key not in self.global_slide_vocab:
            self.global_slide_vocab[key] = self._next_slide_id
            self._next_slide_id += 1
        return self.global_slide_vocab[key]

    @staticmethod
    def _object_sort_key(obj: dict) -> tuple:
        order = {"tap": 0, "hold": 1, "slide": 2, "touch": 3}
        return (order.get(obj["type"], 99), Preprocessor._position_sort_key(obj["pos"]))

    @staticmethod
    def _position_sort_key(pos: str) -> tuple:
        import re
        parts = re.findall(r'[A-E]?\d+', pos)
        if not parts and pos:
            parts = [pos]
        keys = []
        for part in parts:
            if part[0].isdigit():
                keys.append((0, int(part)))
            else:
                digits = part[1:]
                keys.append((1, part[0], int(digits) if digits else 0))
        return tuple(keys) if keys else ((9, pos),)

    def _extract_tags(self, simai_data, chart, chart_dir: Path) -> tuple[list[int], dict[str, int]]:
        """提取谱面标签, 返回 (tag_ids, tag_vocab)

        自动标签 (由 auto_tags 配置控制):
          - designer: 谱师名字
          - difficulty: Easy/Basic/Advanced/Expert/Master/Re:Master/UTAGE
          - dx_type: SD 或 DX
          - level: 等级 (如 Lv5.0)

        手动标签: datasets/*/tags.txt (空格分隔)
        """
        auto_tags = [t.strip() for t in self.cfg.tags.auto_tags.split(",") if t.strip()]
        tags = []

        if "designer" in auto_tags and chart.designer and chart.designer != "-":
            tags.append(f"designer:{chart.designer}")
        if "difficulty" in auto_tags:
            tags.append(f"difficulty:{chart.difficulty_name}")
        if "dx_type" in auto_tags:
            dx = "DX" if "[DX]" in simai_data.title else "SD"
            tags.append(dx)
        if "level" in auto_tags and chart.level > 0:
            tags.append(f"Lv{chart.level:.1f}")

        # 手动标签
        if self.cfg.tags.auto_tags_file:
            tags_path = chart_dir / "tags.txt"
            if tags_path.exists():
                raw = tags_path.read_text(encoding="utf-8").strip()
                if raw:
                    tags.extend(raw.split())

        # collections 标签
        if self.cfg.tags.use_collections:
            coll_tags = self._get_collection_tags(chart_dir.name)
            tags.extend(coll_tags)

        # 分配 ID
        tag_ids = []
        local_vocab = {}
        for tag in tags:
            if tag not in self.global_tag_vocab:
                self.global_tag_vocab[tag] = self._next_tag_id
                self._next_tag_id += 1
            tid = self.global_tag_vocab[tag]
            tag_ids.append(tid)
            local_vocab[tag] = tid

        return tag_ids, local_vocab

    def _get_collection_tags(self, chart_id: str) -> list[str]:
        """从 collections/*/manifest.json 读取谱面所属集合的标签"""
        import json

        coll_dir = Path(self.cfg.tags.collections_dir)
        if not coll_dir.exists():
            return []

        tags = []
        for manifest_path in sorted(coll_dir.rglob("manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text("utf-8"))
            except (json.JSONDecodeError, KeyError):
                continue

            level_ids = manifest.get("levelIds", [])
            if chart_id in level_ids:
                name = manifest.get("name", manifest_path.parent.name)
                # 标签: collection:集合名
                tag = f"collection:{name}"
                if tag not in tags:
                    tags.append(tag)

        return tags


# ============================================================
# 便捷函数
# ============================================================

def preprocess_all(config=None) -> list[Path]:
    """运行完整预处理管线"""
    preprocessor = Preprocessor(config)
    return preprocessor.process_all()


def load_preprocessed(chart_id: str, preprocess_dir: str = "preprocessed") -> PreprocessResult:
    """加载已预处理的谱面"""
    return PreprocessResult.load(Path(preprocess_dir) / f"{chart_id}.npz")
