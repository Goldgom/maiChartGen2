"""
ConfigLoader - 读取 Config/*.yaml 配置文件，提供类型安全的配置访问

设计:
  1. 配置模板 (template.yaml) 定义所有配置项、值域、说明
  2. 默认配置 (default.yaml) 提供开箱即用的默认值
  3. 用户可创建自定义配置 (e.g. my_experiment.yaml) 覆盖部分值
  4. ConfigLoader 负责加载、合并、校验

用法:
  from Config import load_config
  cfg = load_config()                  # 加载 Config/default.yaml
  cfg = load_config("my_experiment")   # 加载 Config/my_experiment.yaml
  cfg = load_config("my_experiment", use_default=False)  # 不使用默认值
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml


# ============================================================
# 配置数据类 (与 template.yaml 一一对应)
# ============================================================

@dataclass
class PathsConfig:
    datasets_dir: str = "datasets"
    output_dir: str = "output"
    model_dir: str = "checkpoints"
    cache_dir: str = "cache"
    vocab_dir: str = "vocab"


@dataclass
class TrainLoopConfig:
    mode: str = "stage_epochs"
    start_stage: int = 1
    epochs_per_stage: int = 5
    val_samples_per_difficulty: int = 25
    val_check_interval: int = 200
    save_best_only: bool = True
    max_rounds: int = 0
    max_frames: int = 0
    refine_max_frames: int = 2048
    oom_retry_attempts: int = 3
    oom_retry_max_frames: int = 2048


@dataclass
class AudioConfig:
    model_id: str = "facebook/encodec_24khz"
    premodel_path: str = ""          # 本地模型路径，空字符串=从 HF 下载
    num_codebooks: int = 8
    device: str = "cpu"
    frame_rate: Optional[float] = None  # 只读


@dataclass
class ChartConfig:
    max_difficulty: int = 6
    include_empty_measures: bool = False
    add_eos_token: bool = True
    target_subdiv: int = 4


@dataclass
class BeatConfig:
    method: str = "librosa"
    bpm_min: float = 60.0
    bpm_max: float = 240.0
    tightness: float = 50.0
    time_signature: int = 4
    downbeat_weight: float = 1.5
    quantize_beats: bool = True
    use_chart_bpm: bool = True


@dataclass
class PreprocessConfig:
    output_dir: str = "preprocessed"
    audio_codebooks: int = 8
    beat_method: str = "librosa"
    beat_as_binary: bool = True
    frame_rate: float = 0.0
    max_charts: int = 0
    skip_existing: bool = True


@dataclass
class TagsConfig:
    auto_tags: str = "designer,difficulty,dx_type"
    auto_tags_file: bool = True
    tag_vocab_path: str = "vocab/tag_vocab.json"
    use_collections: bool = True
    collections_dir: str = "collections"


@dataclass
class DataConfig:
    train_split: float = 0.85
    val_split: float = 0.10
    max_audio_duration: float = 180.0
    min_audio_duration: float = 30.0
    shuffle_seed: int = 42
    num_workers: int = 2


@dataclass
class ModelConfig:
    model_type: str = "transformer"
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 6
    n_audio_codebooks: int = 8
    audio_vocab_size: int = 1024
    chart_vocab_size: int = 512
    max_seq_len: int = 8192
    dropout: float = 0.1
    use_cross_attention: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.0001
    min_learning_rate: float = 0.000001
    warmup_steps: int = 1000
    max_epochs: int = 100
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    save_every_epochs: int = 5
    early_stopping_patience: int = 20


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_every_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "maiChartGen3"


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    top_k: int = 50
    stage1_use_kv_cache: bool = True
    stage1_history_frames: int = 256
    empty_penalty_start: int = 32
    empty_penalty_per_frame: float = 0.08
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    max_new_tokens: int = 4096
    do_sample: bool = True
    num_beams: int = 1


@dataclass
class BatchInferConfig:
    """批量推理配置

    每个难度可以有独立的推理参数，未指定的参数继承全局默认值。

    difficulties 格式:
      - name: "Master"          # 难度名称 (必填)
        level: 13.0             # 等级 (必填)
        temperature: 1.0        # 可选覆盖: 采样温度
        density: 1.0            # 可选覆盖: 密度偏置
        tap_bias: 0.5           # 可选覆盖: Tap 偏置
        ...                     # 任何全局参数均可按难度覆盖
    """

    # ── 路径 ──
    input_dir: str = "samples"
    output_dir: str = "output/batch"

    # ── 文件类型 ──
    audio_extensions: list[str] = field(default_factory=lambda: [".mp3", ".wav", ".ogg", ".flac"])
    video_extensions: list[str] = field(default_factory=lambda: [".mp4", ".webm", ".mkv"])
    output_subdir_template: str = "{input_name}"

    # ── 难度列表 (每项为 dict: {name, level, 可选覆盖参数...}) ──
    # 兼容旧格式: 纯字符串列表自动转为 {"name": str, "level": ...}
    difficulties: list = field(default_factory=lambda: [
        {"name": "Master", "level": 13.0}
    ])

    # ── 标签 ──
    designer: str = "AI"
    collections: list[str] = field(default_factory=lambda: [
        "Original",
        "niconicoボーカロイド",
        "POPSアニメ",
        "翠楼屋",
        "DX Chart",
        "maimai DX CiRCLE",
    ])

    # ── 全局生成参数 (各难度未覆盖时使用) ──
    top_k: int = 50
    bpm_override: float = 0.0
    density: float = 0.0
    tap_bias: float = 0.0
    hold_bias: float = 0.0
    slide_bias: float = 0.0
    wifi_bias: float = 0.0
    touch_bias: float = 0.0
    touchhold_bias: float = 0.0
    break_bias: float = 0.0
    filter_multi_tap: bool = True
    allow_touch: bool = False
    memory_mode: str = "per_stage"
    beat_method: str = "librosa"
    skip_stages: list[str] = field(default_factory=lambda: ["Stage 5"])

    # ── 输出选项 ──
    copy_audio: bool = True
    audio_format: str = "mp3"
    audio_bitrate: str = "192k"
    copy_video: bool = True
    extract_bg: bool = True
    bg_max_size: int = 512
    skip_existing: bool = False


@dataclass
class StageModelConfig:
    model_type: str = "transformer"
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    hold_dur_bins: int = 64
    max_hold_slots: int = 8
    max_slide_slots: int = 8
    max_object_slots: int = 16
    slot_n_layer: int = 2
    global_tag_scale: float = 1.0
    dynamic_tag_scale: float = 1.0
    slide_vocab_size: int = 256


@dataclass
class Config:
    """总配置"""
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    chart: ChartConfig = field(default_factory=ChartConfig)
    beat: BeatConfig = field(default_factory=BeatConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    tags: TagsConfig = field(default_factory=TagsConfig)
    train_loop: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stage1_training: TrainingConfig = field(default_factory=TrainingConfig)
    stage2_training: TrainingConfig = field(default_factory=TrainingConfig)
    stage3_training: TrainingConfig = field(default_factory=TrainingConfig)
    stage4_training: TrainingConfig = field(default_factory=TrainingConfig)
    stage5_training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    batch_infer: BatchInferConfig = field(default_factory=BatchInferConfig)
    stage_model: StageModelConfig = field(default_factory=StageModelConfig)
    stage1_model: StageModelConfig = field(default_factory=StageModelConfig)
    stage2_model: StageModelConfig = field(default_factory=StageModelConfig)
    stage3_model: StageModelConfig = field(default_factory=StageModelConfig)
    stage4_model: StageModelConfig = field(default_factory=StageModelConfig)
    stage5_model: StageModelConfig = field(default_factory=StageModelConfig)

    # 元信息
    config_name: str = ""
    _raw: dict = field(default_factory=dict, repr=False)

    # ---- 辅助属性 ----

    @property
    def effective_batch_size(self) -> int:
        return self.training.batch_size * self.training.gradient_accumulation_steps

    @property
    def test_split(self) -> float:
        return round(1.0 - self.data.train_split - self.data.val_split, 4)

    @property
    def is_cuda(self) -> bool:
        return self.audio.device == "cuda"

    def validate(self) -> list[str]:
        """校验配置合法性，返回错误列表"""
        errors = []

        # 数据分割
        total = self.data.train_split + self.data.val_split
        if total > 1.0 or total <= 0:
            errors.append(f"train_split + val_split = {total}, 必须在 (0, 1] 之间")

        # 音频 codebook 一致性
        if self.audio.num_codebooks != self.model.n_audio_codebooks:
            errors.append(
                f"audio.num_codebooks({self.audio.num_codebooks}) != "
                f"model.n_audio_codebooks({self.model.n_audio_codebooks})"
            )

        # d_model 必须能被 n_head 整除
        if self.model.d_model % self.model.n_head != 0:
            errors.append(f"d_model({self.model.d_model}) 必须能被 n_head({self.model.n_head}) 整除")

        # 值域校验
        for idx in range(1, 6):
            sc = getattr(self, f"stage{idx}_model")
            if sc.model_type != "transformer":
                errors.append(f"stage{idx}_model.model_type({sc.model_type}) must be transformer")
            if sc.d_model % sc.n_head != 0:
                errors.append(f"stage{idx}_model.d_model({sc.d_model}) must be divisible by n_head({sc.n_head})")
        if not (0.0 < self.data.train_split <= 1.0):
            errors.append(f"train_split({self.data.train_split}) 必须在 (0, 1] 之间")
        if self.data.max_audio_duration <= self.data.min_audio_duration:
            errors.append("max_audio_duration 必须大于 min_audio_duration")
        if self.training.learning_rate <= 0:
            errors.append("learning_rate 必须 > 0")
        for idx in range(1, 6):
            tc = getattr(self, f"stage{idx}_training")
            if tc.batch_size <= 0:
                errors.append(f"stage{idx}_training.batch_size 必须 > 0")
            if tc.gradient_accumulation_steps <= 0:
                errors.append(f"stage{idx}_training.gradient_accumulation_steps 必须 > 0")
            if tc.learning_rate <= 0:
                errors.append(f"stage{idx}_training.learning_rate 必须 > 0")
            if tc.min_learning_rate < 0:
                errors.append(f"stage{idx}_training.min_learning_rate 必须 >= 0")
            if str(tc.scheduler).lower() not in ("cosine", "linear", "constant"):
                errors.append(f"stage{idx}_training.scheduler must be cosine, linear, or constant")
            if str(tc.optimizer).lower() not in ("adam", "adamw", "sgd"):
                errors.append(f"stage{idx}_training.optimizer must be adam, adamw, or sgd")
        if not (1 <= self.train_loop.start_stage <= 5):
            errors.append(f"train_loop.start_stage({self.train_loop.start_stage}) must be in [1, 5]")
        if self.train_loop.mode not in ("stage_epochs", "round_robin"):
            errors.append("train_loop.mode must be 'stage_epochs' or 'round_robin'")
        if self.generation.temperature <= 0:
            errors.append("temperature 必须 > 0")
        if self.generation.top_p < 0 or self.generation.top_p > 1.0:
            errors.append("top_p 必须在 [0, 1] 之间")
        if self.generation.empty_penalty_start < 0:
            errors.append("empty_penalty_start 必须 >= 0")
        if self.generation.empty_penalty_per_frame < 0:
            errors.append("empty_penalty_per_frame 必须 >= 0")

        return errors

    def summary(self) -> str:
        """多行摘要"""
        lines = [f"Config: {self.config_name or '(unnamed)'}"]
        for section_name in ["paths", "audio", "chart", "beat", "preprocess", "tags", "train_loop", "data", "model",
                              "training", "stage1_training", "stage2_training", "stage3_training",
                              "stage4_training", "stage5_training", "logging", "generation",
                              "batch_infer",
                              "stage_model", "stage1_model", "stage2_model", "stage3_model",
                              "stage4_model", "stage5_model"]:
            section = getattr(self, section_name)
            lines.append(f"  [{section_name}]")
            for f in fields(section):
                val = getattr(section, f.name)
                lines.append(f"    {f.name}: {val}")
        lines.append(f"  effective_batch_size: {self.effective_batch_size}")
        lines.append(f"  test_split: {self.test_split}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Config({self.config_name or 'unnamed'})"


# ============================================================
# ConfigLoader
# ============================================================

class ConfigLoader:
    """配置加载器

    加载流程:
      1. 从 template.yaml 读取默认值
      2. 用 default.yaml 覆盖
      3. 用用户指定的配置文件覆盖
    """

    CONFIG_DIR = Path(__file__).parent
    TEMPLATE_FILE = "template.yaml"
    DEFAULT_FILE = "default.yaml"

    @classmethod
    def load(
        cls,
        config_name: Optional[str] = None,
        use_default: bool = True,
    ) -> Config:
        """加载配置

        Args:
            config_name: 配置文件名 (不含 .yaml 后缀)
                         为 None 时仅加载默认配置
            use_default: 是否先用 default.yaml 填充默认值

        Returns:
            Config: 合并后的配置对象
        """
        # 1. 加载模板 (仅获取默认值，不参与最终合并)
        template_path = cls.CONFIG_DIR / cls.TEMPLATE_FILE
        template_raw = cls._read_yaml(template_path) if template_path.exists() else {}

        # 2. 从模板提取纯默认值
        defaults = cls._extract_values(template_raw)

        # 3. 用 default.yaml 覆盖
        if use_default:
            default_path = cls.CONFIG_DIR / cls.DEFAULT_FILE
            if default_path.exists():
                default_raw = cls._read_yaml(default_path)
                default_values = cls._extract_values(default_raw)
                defaults = cls._deep_merge(defaults, default_values)

        # 4. 用用户配置覆盖
        user_raw = {}
        if config_name:
            # 自动加 .yaml 后缀
            if not config_name.endswith(".yaml"):
                config_name = config_name + ".yaml"
            user_path = cls.CONFIG_DIR / config_name
            if not user_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {user_path}")
            user_raw = cls._read_yaml(user_path)
            user_values = cls._extract_values(user_raw)
            defaults = cls._deep_merge(defaults, user_values)

        # 5. 构建 Config 对象
        cfg = cls._build_config(defaults)
        cfg.config_name = config_name or "default"
        cfg._raw = {"template": template_raw, "user": user_raw}

        # 6. 校验
        errors = cfg.validate()
        if errors:
            raise ConfigValidationError(errors)

        return cfg

    @classmethod
    def save_default(cls) -> Path:
        """将模板中的默认值提取并保存为 default.yaml"""
        template_path = cls.CONFIG_DIR / cls.TEMPLATE_FILE
        if not template_path.exists():
            raise FileNotFoundError(f"模板文件不存在: {template_path}")

        raw = cls._read_yaml(template_path)
        defaults = cls._extract_values(raw)

        default_path = cls.CONFIG_DIR / cls.DEFAULT_FILE
        with open(default_path, "w", encoding="utf-8") as f:
            yaml.dump(defaults, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        print(f"默认配置已保存: {default_path}")
        return default_path

    # ---- 内部方法 ----

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        content = path.read_text(encoding="utf-8")
        return yaml.safe_load(content) or {}

    @staticmethod
    def _extract_values(raw: dict) -> dict:
        """从含 :value/:description/:range 的 YAML 中提取纯值字典"""
        result = {}
        for key, val in raw.items():
            if isinstance(val, dict):
                # 检查是否含 :value 标记
                if ":value" in val:
                    result[key] = val[":value"]
                else:
                    # 递归
                    result[key] = ConfigLoader._extract_values(val)
            else:
                result[key] = val
        return result

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """深度合并两个字典，override 覆盖 base"""
        result = base.copy()
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = ConfigLoader._deep_merge(result[key], val)
            elif val is not None:
                result[key] = val
        return result

    @staticmethod
    def _build_config(values: dict) -> Config:
        """从字典构建 Config 数据类，自动类型转换"""

        def _coerce(section_cls, raw: dict) -> dict:
            """根据 dataclass 字段类型强制转换值"""
            result = {}
            for f in fields(section_cls):
                if f.name in raw:
                    val = raw[f.name]
                    target = f.type
                    # 处理 Optional[X]
                    origin = getattr(target, "__origin__", None)
                    if origin is Union:
                        args = target.__args__
                        non_none = [a for a in args if a is not type(None)]
                        target = non_none[0] if non_none else str
                    # 类型转换
                    try:
                        if target is int:
                            val = int(val) if not isinstance(val, int) else val
                        elif target is float:
                            val = float(val) if not isinstance(val, float) else val
                        elif target is bool and not isinstance(val, bool):
                            val = str(val).lower() in ("true", "1", "yes")
                        elif target is str:
                            val = str(val)
                    except (ValueError, TypeError):
                        pass  # 保持原值，校验时会报错
                    result[f.name] = val
            return result

        base_stage_raw = values.get("stage_model", {})
        base_training_raw = values.get("training", {})

        def _stage_values(name: str) -> dict:
            return ConfigLoader._deep_merge(base_stage_raw, values.get(name, {}))

        def _stage_training_values(name: str) -> dict:
            return ConfigLoader._deep_merge(base_training_raw, values.get(name, {}))

        return Config(
            paths=PathsConfig(**_coerce(PathsConfig, values.get("paths", {}))),
            audio=AudioConfig(**_coerce(AudioConfig, values.get("audio", {}))),
            chart=ChartConfig(**_coerce(ChartConfig, values.get("chart", {}))),
            beat=BeatConfig(**_coerce(BeatConfig, values.get("beat", {}))),
            preprocess=PreprocessConfig(**_coerce(PreprocessConfig, values.get("preprocess", {}))),
            tags=TagsConfig(**_coerce(TagsConfig, values.get("tags", {}))),
            train_loop=TrainLoopConfig(**_coerce(TrainLoopConfig, values.get("train_loop", {}))),
            data=DataConfig(**_coerce(DataConfig, values.get("data", {}))),
            model=ModelConfig(**_coerce(ModelConfig, values.get("model", {}))),
            training=TrainingConfig(**_coerce(TrainingConfig, values.get("training", {}))),
            stage1_training=TrainingConfig(**_coerce(TrainingConfig, _stage_training_values("stage1_training"))),
            stage2_training=TrainingConfig(**_coerce(TrainingConfig, _stage_training_values("stage2_training"))),
            stage3_training=TrainingConfig(**_coerce(TrainingConfig, _stage_training_values("stage3_training"))),
            stage4_training=TrainingConfig(**_coerce(TrainingConfig, _stage_training_values("stage4_training"))),
            stage5_training=TrainingConfig(**_coerce(TrainingConfig, _stage_training_values("stage5_training"))),
            logging=LoggingConfig(**_coerce(LoggingConfig, values.get("logging", {}))),
            generation=GenerationConfig(**_coerce(GenerationConfig, values.get("generation", {}))),
            batch_infer=BatchInferConfig(**_coerce(BatchInferConfig, values.get("batch_infer", {}))),
            stage_model=StageModelConfig(**_coerce(StageModelConfig, base_stage_raw)),
            stage1_model=StageModelConfig(**_coerce(StageModelConfig, _stage_values("stage1_model"))),
            stage2_model=StageModelConfig(**_coerce(StageModelConfig, _stage_values("stage2_model"))),
            stage3_model=StageModelConfig(**_coerce(StageModelConfig, _stage_values("stage3_model"))),
            stage4_model=StageModelConfig(**_coerce(StageModelConfig, _stage_values("stage4_model"))),
            stage5_model=StageModelConfig(**_coerce(StageModelConfig, _stage_values("stage5_model"))),
        )


# ============================================================
# 异常
# ============================================================

class ConfigValidationError(Exception):
    """配置校验失败"""

    def __init__(self, errors: list[str]):
        msg = "配置校验失败:\n  - " + "\n  - ".join(errors)
        super().__init__(msg)
        self.errors = errors


# ============================================================
# 便捷函数
# ============================================================

def load_config(
    config_name: Optional[str] = None,
    use_default: bool = True,
) -> Config:
    """加载配置的便捷函数

    Args:
        config_name: 配置名 (不含 .yaml), None=仅加载默认
        use_default: 是否合并 default.yaml

    Returns:
        Config 对象
    """
    return ConfigLoader.load(config_name, use_default)


def create_default_config() -> Path:
    """从模板生成默认配置文件"""
    return ConfigLoader.save_default()
