# 训练器设计

## 目标
- 统一支持 `stage1 / touch / slide / break / spike`
- 配置驱动，避免写死超参
- 支持断点续训、AMP、梯度累积、早停、最佳模型保存
- 训练、验证、推理解耦

## 推荐结构
```text
train/
  trainer.py
  data.py
  optim.py
  metrics.py
  checkpoint.py
  recipes.py
  configs/
    stage1.yaml
    touch.yaml
    slide.yaml
    break.yaml
    spike.yaml
```

## 核心接口
```python
Trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    loss_fn,
    metrics_fn,
    cfg,
)
```

## 配置示例
```yaml
stage: stage1
seed: 42
device: cuda
precision: amp
compile: false

data:
  root: datasets
  cache_dir: cache
  num_workers: 8
  max_seq_len: 4096
  audio_subdiv: 64

train:
  batch_size: 8
  grad_accum_steps: 4
  epochs: 20
  log_every: 50
  eval_every: 1000
  save_every: 1000
  clip_grad_norm: 1.0

optim:
  name: adamw
  lr: 3e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]

sched:
  name: cosine
  warmup_steps: 2000
  min_lr: 1e-5

checkpoint:
  dir: runs/stage1
  keep_last: 3
  keep_best: true
  resume: auto
```

## 训练流程
1. 读配置
2. 固定随机种子
3. 构建数据集、缓存、dataloader
4. 构建模型、优化器、scheduler
5. 恢复 checkpoint
6. 循环训练：
   - forward
   - loss
   - backward
   - grad clip
   - optimizer step
   - scheduler step
7. 定期验证
8. 保存 `last / best / periodic`

## 可靠性要求
- checkpoint 必须保存：
  - `model_state`
  - `optimizer_state`
  - `scheduler_state`
  - `scaler_state`
  - `epoch / step`
  - `best_metric`
  - `rng_state`
  - `config`
  - `git commit hash`
- 遇到 `nan / inf`：
  - 跳过 batch
  - 记录日志
  - 必要时降低学习率或中止
- 验证集固定，不参与缓存更新

## 各阶段训练目标
### Stage 1
- 目标：`config token + inline DUR/num/den`
- loss：token CE
- metric：token acc、config acc、duration acc、perplexity

### Stage 2
- 目标：touch zone / state
- loss：zone/state CE
- metric：zone F1、exact match

### Stage 2.5
- 目标：slide connector/path
- loss：路径 CE
- metric：path exact match、valid path rate

### Stage 3
- 目标：每个 press 位的 break 二分类
- loss：BCE / CE
- metric：macro F1、AUC

### Stage 4
- 目标：每个 touch zone 的 firework 二分类
- loss：BCE / CE
- metric：macro F1、AUC

## 数据策略
- `stage1`：
  - `strip_break`
  - `strip_firework`
  - touch 压缩
  - slide 只保留 start/end/duration
- `touch`：
  - 用 stage1 token 和原始 touch 标签对齐
- `break/spike`：
  - 直接从原 chart 生成标签
- 音频特征建议预缓存：
  - spectral 特征
  - audio tokens
  - manifest

## 预处理与分批训练
为了进一步节省显存，建议把训练拆成“离线预处理”和“分阶段批训练”两层。

### 离线预处理
- 先把原始 `maidata.txt` / 音频转换成 stage 无关的中间缓存
- 缓存内容按需分层保存：
  - `audio_spectral.pt`
  - `audio_tokens.pt`
  - `stage1_tokens.pt`
  - `stage1_hidden.pt`（可选，若后续 stage 固定 stage1）
  - `touch_targets.pt`
  - `break_targets.pt`
  - `spike_targets.pt`
  - `slide_targets.pt`
- 每个缓存必须带 `manifest.json`
  - 数据版本
  - tokenizer 版本
  - 预处理参数
  - 音频采样率
  - subdivision

### 分阶段批训练
- 每个 stage 只加载自己需要的缓存
- `stage1` 只读：
  - `audio_spectral`
  - `audio_tokens`
  - `stage1_tokens`
- `touch / break / spike / slide` 只读：
  - 对应 stage1 输出或其缓存后的 hidden
  - 该 stage 的标签
- 训练器支持 `stage_recipe`
  - 决定当前 batch 需要哪些字段
  - 自动丢弃无关字段
  - 降低 batch 内存占用

### 进一步省显存的做法
- `stage1_hidden` 可在 stage1 训练后离线导出，后续 stage 直接读取
- 冻结 stage1 时，不再回传 stage1 梯度
- `audio_tokens` 和 spectral 特征都可用 fp16/bf16 存储缓存
- 长序列可按 chunk 预处理，再在训练时拼接或滑窗读取

### 推荐训练流程
1. 离线预处理全量数据
2. 训练 stage1
3. 导出 stage1 hidden cache
4. 训练 touch / slide / break / spike
5. 若需要，再做联合微调

## 推荐训练顺序
1. `stage1`
2. `touch`
3. `slide`
4. `break`
5. `spike`
6. 可选端到端微调

## 最终建议
- 一个训练运行只负责一个 stage
- stage 只切换：
  - dataset recipe
  - loss
  - metrics
  - freeze policy
  - postprocess
- 所有 stage 共用同一个 `Trainer`
