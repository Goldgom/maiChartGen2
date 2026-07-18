#!/usr/bin/env python3
"""
batch_infer.py — 批量推理脚本

对指定文件夹下所有 mp3/mp4 文件批量生成 maimai 谱面。
每首歌在输出目录下创建独立子文件夹，包含:
  - track.mp3     (音频)
  - maidata.txt   (谱面, 可含多个难度)
  - pv.mp4        (视频, 可选)
  - bg.png        (封面, 可选)

推理链路与 webui.py 完全一致，统一使用 infer_core.py 共享引擎。

配置位于主配置文件的 batch_infer 段 (Config/default.yaml / Config/server_4090.yaml):
  - 输入/输出路径
  - 难度列表和等级
  - 生成参数 (温度、偏置等)
  - 输出选项 (音频转换、视频处理、封面提取)

用法:
  python batch_infer.py                          # 使用 Config/default.yaml
  python batch_infer.py --config server_4090     # 使用 Config/server_4090.yaml
  python batch_infer.py --input_dir /path/to/mp3  # 命令行覆盖输入目录
  python batch_infer.py --dry_run                 # 仅预览文件列表
"""

from __future__ import annotations

import argparse
import gc
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

from Config import load_config
from infer_core import create_engine, AudioContext


# ============================================================
# 配置加载 — 从主配置系统读取
# ============================================================

# ============================================================
# 辅助函数
# ============================================================

def _normalize_skip_stages(skip_stages: list | tuple | set | str | None) -> list[str]:
    """标准化 stage 跳过配置为 'Stage N' 格式。"""
    if skip_stages is None:
        return []
    if isinstance(skip_stages, str):
        raw_items = re.split(r"[,，]", skip_stages)
    else:
        raw_items = list(skip_stages)
    normalized = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        match = re.search(r"([1-5])", text)
        if match:
            label = f"Stage {match.group(1)}"
            if label not in normalized:
                normalized.append(label)
    return normalized


# ============================================================
# 日志
# ============================================================

class Logger:
    def __init__(self, log_file: str | None = None, verbose: bool = True):
        self.verbose = verbose
        self.log_file = None
        if log_file:
            try:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                self.log_file = open(log_file, "a", encoding="utf-8")
            except OSError as e:
                if self.verbose:
                    print(f"[WARN] 日志文件不可写，改为仅输出到控制台: {log_file} ({e})",
                          flush=True)

    def _write(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        if self.verbose:
            print(line, flush=True)
        if self.log_file:
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def info(self, msg: str):
        self._write(f"INFO  | {msg}")

    def warn(self, msg: str):
        self._write(f"WARN  | {msg}")

    def error(self, msg: str):
        self._write(f"ERROR | {msg}")

    def success(self, msg: str):
        self._write(f"OK    | {msg}")

    def close(self):
        if self.log_file:
            self.log_file.close()


# ============================================================
# 文件工具
# ============================================================

def find_ffmpeg() -> str:
    """查找 ffmpeg 可执行文件"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(p).exists():
            return p
    return "ffmpeg"


def sanitize_filename(name: str) -> str:
    """清理文件名, 移除非法字符"""
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip()


def extract_audio_from_video(video_path: Path, output_path: Path,
                              fmt: str = "mp3", bitrate: str = "192k",
                              logger: Logger | None = None) -> bool:
    """使用 ffmpeg 从视频提取音频"""
    ffmpeg = find_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame" if fmt == "mp3" else "aac",
        "-b:a", bitrate,
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            if logger:
                stderr_str = result.stderr.decode('utf-8', errors='replace')[:200] if result.stderr else '(none)'
                logger.warn(f"ffmpeg 音频提取失败: {stderr_str}")
            return False
        return output_path.exists()
    except Exception as e:
        if logger:
            logger.warn(f"ffmpeg 异常: {e}")
        return False


def extract_first_frame(video_path: Path, output_path: Path,
                         max_size: int = 512,
                         logger: Logger | None = None) -> bool:
    """使用 ffmpeg 提取视频第一帧为 PNG"""
    ffmpeg = find_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_suffix(".temp.png")
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vframes", "1", "-q:v", "2",
        str(temp_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            if logger:
                stderr_str = result.stderr.decode('utf-8', errors='replace')[:200] if result.stderr else '(none)'
                logger.warn(f"ffmpeg 提取帧失败: {stderr_str}")
            return False

        if max_size > 0:
            cmd2 = [
                ffmpeg, "-y", "-i", str(temp_path),
                "-vf", f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease",
                str(output_path),
            ]
            subprocess.run(cmd2, capture_output=True, timeout=30)
            temp_path.unlink(missing_ok=True)
        else:
            shutil.move(str(temp_path), str(output_path))

        return output_path.exists()
    except Exception as e:
        if logger:
            logger.warn(f"ffmpeg 提取帧异常: {e}")
        return False


def copy_or_convert_audio(src_path: Path, dst_path: Path,
                           fmt: str = "mp3", bitrate: str = "192k",
                           logger: Logger | None = None) -> bool:
    """复制或转换音频文件到目标路径"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    src_ext = src_path.suffix.lower()
    if src_ext == f".{fmt}":
        try:
            shutil.copy2(src_path, dst_path)
            return True
        except Exception as e:
            if logger:
                logger.warn(f"复制音频失败: {e}")
            return False

    return extract_audio_from_video(src_path, dst_path, fmt, bitrate, logger)


# ============================================================
# 多难度合并
# ============================================================

def merge_multi_difficulty(
    title: str,
    artist: str,
    bpm: float,
    diff_results: list[tuple[str, str, float, int]],
    engine,
) -> str:
    """将多个难度的推理结果合并为一个 maidata.txt"""
    lines = [
        f"&title={title}",
        f"&artist={artist}",
        f"&wholebpm={bpm:.1f}",
    ]

    for diff_name, simai_body, level, note_count in diff_results:
        diff_num = engine.DIFF_MAP.get(diff_name, 5)
        lines.append(f"&lv_{diff_num}={level:.1f}")
        lines.append(f"&des_{diff_num}={artist}")
        lines.append(f"&inote_{diff_num}=")
        lines.append(simai_body)

    return "\n".join(lines) + "\n"


# ============================================================
# 主流程
# ============================================================

def scan_input_files(input_dir: Path, audio_exts: set, video_exts: set,
                     logger: Logger) -> list[Path]:
    """扫描输入文件夹, 返回待处理的文件列表 (去重: mp4 优先)"""
    if not input_dir.exists():
        logger.error(f"输入文件夹不存在: {input_dir}")
        return []

    allowed_exts = audio_exts | video_exts
    all_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts
    ]
    all_files = sorted(set(all_files), key=lambda p: p.stem)

    # 去重: mp4 和 mp3 同名时优先 mp4
    seen_names: dict[str, Path] = {}
    for f in all_files:
        name = f.stem
        is_video = f.suffix.lower() in video_exts
        if name not in seen_names:
            seen_names[name] = f
        elif is_video:
            if seen_names[name].suffix.lower() not in video_exts:
                seen_names[name] = f

    result = sorted(seen_names.values(), key=lambda p: p.stem)
    logger.info(f"扫描到 {len(all_files)} 个文件, 去重后 {len(result)} 个待处理")
    for f in result:
        logger.info(f"  - {f.name}")
    return result


def _clear_gpu_cache(device: str):
    """释放 GPU 缓存。"""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_one_file(
    src_path: Path,
    engine,
    cfg,
    logger: Logger,
) -> bool:
    """处理单个输入文件: 使用共享引擎生成所有难度谱面。"""
    bi = cfg.batch_infer
    out_cfg = cfg.batch_infer   # output options 也在 batch_infer 段
    gen = cfg.generation

    # 输出子文件夹名
    subdir_name = bi.output_subdir_template.format(
        input_name=sanitize_filename(src_path.stem),
    )
    output_dir = Path(bi.output_dir) / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    maidata_path = output_dir / "maidata.txt"
    track_path = output_dir / f"track.{out_cfg.audio_format}"
    pv_path = output_dir / "pv.mp4"
    bg_path = output_dir / "bg.png"

    # 检查是否跳过
    if out_cfg.skip_existing and maidata_path.exists():
        logger.info(f"跳过 (已存在): {output_dir}")
        return True

    is_video = src_path.suffix.lower() in set(bi.video_extensions)

    title = src_path.stem
    designer = bi.designer
    logger.info(f"处理: {title} ({'视频' if is_video else '音频'})")

    # ── Step 1: 准备音频 ──
    actual_audio_path: Path = src_path
    if is_video:
        if out_cfg.copy_audio:
            logger.info("  提取音频...")
            if extract_audio_from_video(
                src_path, track_path,
                fmt=out_cfg.audio_format,
                bitrate=out_cfg.audio_bitrate,
                logger=logger,
            ):
                actual_audio_path = track_path
            else:
                # ffmpeg 不可用时回退: 直接用视频文件推理
                logger.warn(f"  ffmpeg 不可用, 直接使用视频文件推理")
                actual_audio_path = src_path
        else:
            actual_audio_path = src_path
    else:
        if out_cfg.copy_audio:
            logger.info("  复制音频...")
            if copy_or_convert_audio(
                src_path, track_path,
                fmt=out_cfg.audio_format,
                bitrate=out_cfg.audio_bitrate,
                logger=logger,
            ):
                actual_audio_path = track_path
            else:
                # 转换失败时回退: 直接用源文件推理
                logger.warn(f"  音频复制/转换失败, 直接使用源文件推理")
                actual_audio_path = src_path
        # else: actual_audio_path already = src_path

    # ── Step 2: 复制视频 / 提取封面 ──
    if is_video:
        if out_cfg.copy_video:
            logger.info("  复制视频...")
            try:
                shutil.copy2(src_path, pv_path)
            except Exception as e:
                logger.warn(f"  复制视频失败: {e}")

        if out_cfg.extract_bg:
            logger.info("  提取封面...")
            video_for_bg = pv_path if pv_path.exists() else src_path
            if not extract_first_frame(video_for_bg, bg_path,
                                       max_size=out_cfg.bg_max_size,
                                       logger=logger):
                logger.warn("  封面提取失败 (非致命)")

    # ── Step 3: 准备难度列表 ──
    raw_diffs = bi.difficulties
    normalized_diffs: list[dict] = []
    for item in raw_diffs:
        if isinstance(item, str):
            normalized_diffs.append({"name": item, "level": 13.0})
        elif isinstance(item, dict):
            if "name" not in item:
                raise ValueError(f"batch_infer.difficulties 项缺少 name: {item}")
            normalized_diffs.append(dict(item))
        else:
            raise ValueError(f"batch_infer.difficulties 不支持的项: {item!r}")

    # ── Step 4: 推理各难度 (使用共享引擎) ──
    diff_results: list[tuple[str, str, float, int]] = []
    overall_bpm = 120.0
    audio_context_cache: dict[tuple, AudioContext] = {}

    # 全局默认参数
    global_defaults = {
        "temperature": gen.temperature,
        "top_k": getattr(bi, "top_k", gen.top_k),
        "bpm_override": bi.bpm_override,
        "density": bi.density,
        "tap_bias": bi.tap_bias,
        "hold_bias": bi.hold_bias,
        "slide_bias": bi.slide_bias,
        "wifi_bias": bi.wifi_bias,
        "touch_bias": bi.touch_bias,
        "touchhold_bias": bi.touchhold_bias,
        "break_bias": bi.break_bias,
        "filter_multi_tap": bi.filter_multi_tap,
        "allow_touch": bi.allow_touch,
        "skip_stages": _normalize_skip_stages(bi.skip_stages),
        "collections": bi.collections,
    }

    _OVERRIDABLE_KEYS = [
        "temperature", "top_k", "bpm_override",
        "density", "tap_bias", "hold_bias", "slide_bias", "wifi_bias",
        "touch_bias", "touchhold_bias", "break_bias",
        "filter_multi_tap", "allow_touch", "beat_method", "skip_stages",
    ]

    for i, diff_info in enumerate(normalized_diffs):
        diff_name = diff_info["name"]
        if diff_name not in engine.DIFF_MAP:
            logger.warn(f"  未知难度 '{diff_name}', 跳过")
            continue

        level = diff_info.get("level", 13.0)

        # 合并参数: 全局默认 + 难度覆盖
        diff_gen = dict(global_defaults)
        for k in _OVERRIDABLE_KEYS:
            if k in diff_info:
                val = diff_info[k]
                diff_gen[k] = _normalize_skip_stages(val) if k == "skip_stages" else val

        overrides_str = ", ".join(
            f"{k}={v}" for k, v in diff_info.items()
            if k in _OVERRIDABLE_KEYS and k != "name" and k != "level"
        )
        if overrides_str:
            logger.info(f"  [{i+1}/{len(normalized_diffs)}] {diff_name} Lv.{level} ({overrides_str})")
        else:
            logger.info(f"  [{i+1}/{len(normalized_diffs)}] {diff_name} Lv.{level}")

        try:
            # 预计算音频上下文 (缓存复用)
            audio_ctx_key = (
                float(diff_gen.get("bpm_override", 0) or 0),
                diff_gen.get("beat_method", "librosa"),
            )
            audio_ctx = audio_context_cache.get(audio_ctx_key)
            if audio_ctx is None:
                logger.info(f"    编码音频 (CPU)...")
                audio_ctx = engine.prepare_audio(
                    str(actual_audio_path),
                    bpm_override=diff_gen.get("bpm_override", 0),
                )
                audio_context_cache[audio_ctx_key] = audio_ctx
                logger.info(f"    音频: {audio_ctx.duration:.1f}s, BPM={audio_ctx.bpm:.1f}")
            else:
                logger.info("    复用音频/节拍缓存 (CPU)")

            # 调用共享引擎推理
            result = engine.generate_chart_body(
                mp3_path=str(actual_audio_path),
                difficulty=diff_name,
                level=level,
                designer=designer,
                collections=diff_gen.get("collections", []),
                temperature=diff_gen.get("temperature", 0.8),
                top_k=diff_gen.get("top_k", 50),
                bpm_override=diff_gen.get("bpm_override", 0),
                density=diff_gen.get("density", 0.0),
                tap_bias=diff_gen.get("tap_bias", 0.0),
                hold_bias=diff_gen.get("hold_bias", 0.0),
                slide_bias=diff_gen.get("slide_bias", 0.0),
                wifi_bias=diff_gen.get("wifi_bias", 0.0),
                touch_bias=diff_gen.get("touch_bias", 0.0),
                touchhold_bias=diff_gen.get("touchhold_bias", 0.0),
                break_bias=diff_gen.get("break_bias", 0.0),
                filter_multi_tap=diff_gen.get("filter_multi_tap", True),
                skip_stages=diff_gen.get("skip_stages", []),
                audio_ctx=audio_ctx,
                allow_touch=diff_gen.get("allow_touch", True),
            )

            if result is None:
                logger.error(f"  {diff_name} 推理失败")
                continue

            simai_body, bpm, note_count = result
            overall_bpm = bpm
            diff_results.append((diff_name, simai_body, level, note_count))
            logger.success(f"  {diff_name}: {note_count} notes, BPM={bpm:.1f}")

        except Exception as e:
            logger.error(f"  {diff_name} 推理异常: {e}")
            traceback.print_exc()
            _clear_gpu_cache(engine.device)
            continue

    if not diff_results:
        logger.error(f"  所有难度推理均失败: {title}")
        audio_context_cache.clear()
        _clear_gpu_cache(engine.device)
        return False

    # ── Step 5: 合并写入 maidata.txt ──
    maidata_text = merge_multi_difficulty(
        title=title,
        artist=designer,
        bpm=overall_bpm,
        diff_results=diff_results,
        engine=engine,
    )
    maidata_path.write_text(maidata_text, encoding="utf-8")
    logger.success(f"  谱面已保存: {maidata_path}")
    logger.info(f"  输出目录: {output_dir}")

    audio_context_cache.clear()
    _clear_gpu_cache(engine.device)
    return True


def main():
    parser = argparse.ArgumentParser(description="maiChartGen3 批量推理 (共享引擎)")
    parser.add_argument("--config", type=str, default=None,
                        help="主配置文件名 (不含 .yaml, 默认使用 default)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="输入文件夹 (覆盖配置文件)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出根目录 (覆盖配置文件)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--designer", type=str, default=None,
                        help="谱面作者")
    parser.add_argument("--skip_existing", action="store_true", default=None,
                        help="跳过已存在的输出")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅列出待处理文件, 不实际推理")
    args = parser.parse_args()

    # ── 加载配置 ──
    cfg = load_config(args.config)
    print(f"[batch_infer] 已加载配置: {cfg.config_name}")

    # CLI 覆盖
    if args.input_dir:
        cfg.batch_infer.input_dir = args.input_dir
    if args.output_dir:
        cfg.batch_infer.output_dir = args.output_dir
    if args.device:
        cfg.audio.device = args.device
    if args.designer:
        cfg.batch_infer.designer = args.designer
    if args.skip_existing is not None:
        cfg.batch_infer.skip_existing = args.skip_existing

    # ── 日志 ──
    log_file = cfg.logging.log_dir + "/batch_infer.log"
    logger = Logger(log_file, verbose=True)

    logger.info("=" * 60)
    logger.info("maiChartGen3 批量推理启动 (共享引擎)")
    logger.info(f"配置文件: {cfg.config_name}")
    logger.info(f"设备: {cfg.audio.device}")
    logger.info("=" * 60)

    # ── 创建推理引擎 ──
    logger.info("初始化推理引擎...")
    engine = create_engine(cfg)
    logger.info(f"引擎就绪: device={engine.device}, vocab={len(engine.vocab)} tokens")

    # ── 扫描输入文件 ──
    bi = cfg.batch_infer
    input_dir = Path(bi.input_dir)
    audio_exts = set(bi.audio_extensions)
    video_exts = set(bi.video_extensions)
    files = scan_input_files(input_dir, audio_exts, video_exts, logger)

    if not files:
        logger.error("没有找到可处理的文件")
        logger.close()
        return

    if args.dry_run:
        logger.info("Dry-run 模式, 不执行推理")
        logger.close()
        return

    # ── 批量处理 ──
    total = len(files)
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for i, src_path in enumerate(files):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"[{i+1}/{total}] {src_path.name}")
        logger.info(f"{'─' * 50}")

        try:
            if process_one_file(src_path, engine, cfg, logger):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"未捕获的异常: {e}")
            traceback.print_exc()
            fail_count += 1
        finally:
            _clear_gpu_cache(engine.device)

        # 进度
        elapsed = time.time() - start_time
        done = i + 1
        eta = elapsed / done * (total - done) if done > 0 else 0
        logger.info(f"进度: {done}/{total} | 成功: {success_count} | 失败: {fail_count} | "
                    f"耗时: {elapsed:.0f}s | ETA: {eta:.0f}s")

    # ── 总结 ──
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("批量推理完成")
    logger.info(f"  总数: {total}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  输出目录: {bi.output_dir}")
    logger.info("=" * 60)

    logger.close()


if __name__ == "__main__":
    main()
