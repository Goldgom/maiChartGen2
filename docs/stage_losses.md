# Stage Loss 计算方式文档

> 生成时间：2026-07-04 | 基于当前代码状态

---

## 总体框架

所有 stage 的训练流程统一在 `train/trainer.py` 的 `RotatingMultiStageTrainer._train_turn()` 中进行：

1. **bfloat16 autocast** 下前向计算
2. `step_fn(model, batch, device)` → 返回 `(loss_tensor, stats_dict)`
3. `loss.backward()` → gradient accumulation → optimizer.step()
4. 异常处理：`loss` 非有限时 `continue`（跳过该 batch）

NaN 防护惯例：target 无效时返回 `logits.sum() * 0.0`（保持计算图连续性）。

---

## Stage 1: maiG（配置序列生成）

| 项目 | 内容 |
|------|------|
| **模型** | `MaiGenerator` (`models/stage1.py`) |
| **step_fn** | `stage1_step()` (`train/recipes.py:48`) |
| **任务** | 自回归预测下一时刻的 config token |
| **输入** | `onset`, `chroma`, `centroid`, `tokens[:, :-1]`, `bpm`, `level`, `genre`, `distances`, `audio_tokens` |
| **输出** | `hidden_states: [B, T, hidden_dim]` |

### Loss 公式

$$
\mathcal{L}_{\text{stage1}} = \text{CrossEntropy}\big(\text{lm\_head}(\text{hidden}),\ \text{tokens}[:, 1:],\ \text{ignore\_index}=\text{PAD}(0)\big)
$$

**实现细节**：
- `MaiGenerator.forward()` 将输入切为 `inp = tokens[:, :-1]`，目标为 `tgt = tokens[:, 1:]`
- 训练时用 `_chunked_lm_ce()` 分块计算：每 chunk 256 个 token，逐块 `lm_head → CE → sum`，避免 `[B×T, 161K]` 大 tensor OOM
- 最终 `total_loss / total_valid_tokens` 取平均
- 支持滑窗模式（`use_sliding_window=True`）：随机采样 `window_tokens` 长度窗口训练
- 位置编码：`BeatPositionEncoding`（bar/beat/sub + global） + `RelativeTimingEncoding`（4 类事件距离）

```python
# 核心代码路径
if self.training:
    loss = _chunked_lm_ce(self.lm_head, x, tgt, self.pad_token_id)  # pad=0 忽略
```

---

## Stage 2: Touch（触控区域精炼）

| 项目 | 内容 |
|------|------|
| **模型** | `TouchRefiner` (`models/touch_stage.py`) |
| **step_fn** | `touch_step()` (`train/recipes.py:62`) |
| **任务** | 给定 Stage 1 config tokens，逐 slot 预测触摸区域和状态 |
| **输入** | `config_tokens`, `stage1_hidden`, `audio_memory` (可选) |
| **输出** | `zone_logits: [B, T, 33 zones, 3 states]` |

### Loss 公式

$$
\mathcal{L}_{\text{touch}} = \text{CrossEntropy}\big(\text{zone\_logits}[\text{valid}][\text{mask}],\ \text{zone\_targets}[\text{valid}][\text{mask}] - 1\big)
$$

**实现细节**：
- 两步过滤：
  1. `valid = (config_tokens != PAD)` — 非填充位置
  2. `mask = (flat_targets > 0)` — 目标中 state>0 的位置（state=0 表示"无触摸"）
- `zone_targets` 取值 0/1/2（0=无触摸/1=Touch/2=Hold），参与 loss 的只减 1 变为 0/1
- 输出维度 `[33 zones × 3 states]`，loss 只在有触摸的 zone 上计算

```python
flat_logits = zone_logits[valid].reshape(-1, self.num_states)  # [N, 3]
flat_targets = zone_targets[valid].reshape(-1)                  # [N]
mask = flat_targets > 0                                         # 过滤 state=0
return F.cross_entropy(flat_logits[mask], flat_targets[mask] - 1)
```

---

## Stage 3: Slide（星星路径生成）

| 项目 | 内容 |
|------|------|
| **模型** | `SlidePathGenerator` (`models/slide_stage.py`) |
| **step_fn** | `slide_step()` (`train/recipes.py:75`) |
| **任务** | 自回归生成星星完整路径（BOS → dur → connector → position → ... → EOS） |
| **输入** | `target_path`, `start_pos`, `audio_memory`, `stage1_hidden`, `onset` |
| **输出** | `logits: [B, T_path-1, 78]`, `loss` |

### Loss 公式

$$
\mathcal{L}_{\text{slide}} = \text{CrossEntropy}\big(\text{logits},\ \text{tgt},\ \text{ignore\_index}=\text{SLD\_STAR\_PAD}(2)\big)
$$

**实现细节**：
- `SlidePathGenerator.forward()` 是**自包含**的：forward 内部计算 loss 并返回 `{"logits", "loss"}`
- Teacher forcing：`inp = target_path[:, :-1]`，`tgt = target_path[:, 1:]`
- Vocab 大小 78（3 特殊 + 41 位置 + 14 连接符 + 8 dur_num + 12 dur_den）
- Causal mask（上三角 `-inf`），与模型参数同 dtype
- Logits clamp `[-50, 50]` 防止溢出
- `stage1_hidden` 和 `onset` 均全局池化为 `[B, 1, D]` 作为条件
- NaN 保护：`audio_memory` 做 `nan_to_num`

```python
logits = self.head(x)
logits = torch.clamp(logits, min=-50.0, max=50.0)
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=SLD_STAR_PAD)
```

---

## Stage 4: Hold（按键长按持续时间）

| 项目 | 内容 |
|------|------|
| **模型** | `HoldDurationPredictor` (`models/hold_stage.py`) |
| **step_fn** | `hold_step()` (`train/recipes.py:126`) |
| **任务** | 在 hold_start slot 预测持续时间 (dur_num, dur_den) |
| **输入** | `tokens`, `stage1_hidden`, `audio_memory`, `onset` |
| **输出** | `num_logits: [B, T, 8]`, `den_logits: [B, T, 12]` |

### Loss 公式

$$
\mathcal{L}_{\text{hold}} = \text{CE}(\text{num\_logits}[\text{mask}],\ \text{num\_targets}[\text{mask}]) + \text{CE}(\text{den\_logits}[\text{mask}],\ \text{den\_targets}[\text{mask}])
$$

**实现细节**：
- 只在 `hold_mask == True` 的 slot 计算 loss（即 hold_start 位置）
- 双头独立 CrossEntropy 后相加
- 分子分类：8 类 → `{1, 2, 3, 4, 6, 8, 12, 16}`
- 分母分类：12 类 → `{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}`
- 模型输出所有位置的 logits，但 loss 只在 hold_start 计算
- 音频和节拍特征通过全局池化广播到序列

```python
valid = hold_mask.bool()
num_logits = outputs["num_logits"][valid]  # [N, 8]
den_logits = outputs["den_logits"][valid]  # [N, 12]
loss_num = F.cross_entropy(num_logits, num_targets[valid])
loss_den = F.cross_entropy(den_logits, den_targets[valid])
return loss_num + loss_den
```

---

## Stage 5: TouchHold（触控长按持续时间）

| 项目 | 内容 |
|------|------|
| **模型** | `TouchHoldDurationPredictor` (`models/hold_stage.py`) |
| **step_fn** | `touch_hold_step()` (`train/recipes.py:146`) |
| **任务** | 在 touch_hold_start slot 预测持续时间 |

### Loss 公式

$$
\mathcal{L}_{\text{touch\_hold}} = \text{CE}(\text{num\_logits}[\text{mask}],\ \text{num\_targets}[\text{mask}]) + \text{CE}(\text{den\_logits}[\text{mask}],\ \text{den\_targets}[\text{mask}])
$$

**实现细节**：
- `TouchHoldDurationPredictor` 直接继承 `HoldDurationPredictor`（`pass`），结构和 loss 完全一致
- 区别仅在于 mask 字段名：`touch_hold_mask` vs `hold_mask`
- 双头 CrossEntropy 求和，只在 mask slot 计算

---

## Stage 6: Star（星星路径精炼）

| 项目 | 内容 |
|------|------|
| **模型** | `SlideStarRefiner` (`models/slide_stage.py:227`) |
| **step_fn** | `star_step()` (`train/recipes.py:180`) |
| **任务** | 将 Stage 2 粗粒度 slide star 路径精炼为更详细排列 |
| **输入** | `coarse_path`, `stage1_hidden`, `audio_memory`, `onset`, `target_path` |
| **输出** | `logits: [B, T_coarse, 78]`, `loss` |

### Loss 公式

$$
\mathcal{L}_{\text{star}} = \text{CrossEntropy}\big(\text{logits},\ \text{target\_path},\ \text{ignore\_index}=\text{SLD\_STAR\_PAD}(2)\big)
$$

**实现细节**：
- **Bidirectional** Transformer（非 Causal），可看到完整 coarse_path
- 每个 position 预测对应的 target_path token
- 与 Slide 相同 vocab（78 tokens）、相同 ignore_index
- Logits clamp `[-50, 50]`
- 上下文通过全局池化广播：`stage1_hidden.mean()`, `audio_memory.mean()`, `onset.mean()` 扩展到 `[B, T_c, D]`

```python
logits = self.head(x)
logits = torch.clamp(logits, min=-50.0, max=50.0)
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_path.reshape(-1), ignore_index=SLD_STAR_PAD)
```

---

## Stage 7: Break（Break/Spike 音符分类）

| 项目 | 内容 |
|------|------|
| **模型** | `BreakClassifier` (`models/break_stage.py`) |
| **step_fn** | `break_step()` (`train/recipes.py:106`) |
| **任务** | 在每个 press slot 的 8 个按钮位置判断是 tap 还是 break |
| **输入** | `tokens`, `stage1_hidden` |
| **输出** | `logits: [B, T, 8 positions, 2 classes]` |

### Loss 公式

$$
\mathcal{L}_{\text{break}} = \text{CrossEntropy}\big(\text{logits}[\text{press\_mask}],\ \text{targets}[\text{press\_mask}] - 1\big)
$$

**实现细节**：
- 只在 `press_mask == True` 的 slot 计算 loss
- 每个 position（8 个按钮位）独立二分类
- `targets` 取值 1(tap) / 2(break)，减 1 映射为 0/1
- 8 个位置共享相同的 2 类 logits head

```python
valid = press_mask.bool()
flat_logits = logits[valid].reshape(-1, 2)
flat_targets = targets[valid].reshape(-1) - 1  # 1→0(tap), 2→1(break)
return F.cross_entropy(flat_logits, flat_targets)
```

---

## Stage 8: Spike（Firework/Spike 触控分类）

| 项目 | 内容 |
|------|------|
| **模型** | `SpikeClassifier` (`models/spike_stage.py`) |
| **step_fn** | `spike_step()` (`train/recipes.py:114`) |
| **任务** | 在每个 touch slot 的 33 个触控区域判断是否有 firework |
| **输入** | `tokens`, `stage1_hidden` |
| **输出** | `logits: [B, T, 33 zones, 2 classes]` |

### Loss 公式

$$
\mathcal{L}_{\text{spike}} = \text{CrossEntropy}\big(\text{logits}[\text{touch\_mask}],\ \text{targets}[\text{touch\_mask}]\big)
$$

**实现细节**：
- 只在 `touch_mask == True` 的 slot 计算 loss
- 每个 zone（33 个触控区域）独立二分类
- 与 Break 不同：targets 不偏移（直接是 0/1 标签）

```python
valid = touch_mask.bool()
flat_logits = logits[valid].reshape(-1, 2)
flat_targets = targets[valid].reshape(-1)
return F.cross_entropy(flat_logits, flat_targets)
```

---

## 附：TouchPattern（触控排列精炼）

| 项目 | 内容 |
|------|------|
| **模型** | `TouchPatternRefiner` (`models/touch_pattern_stage.py`) |
| **step_fn** | `touch_pattern_step()` (`train/recipes.py:165`) |
| **任务** | 多标签二分类：在 touch slot 预测 16 个触控区域是否激活 |

### Loss 公式

$$
\mathcal{L}_{\text{touch\_pattern}} = \text{BCEWithLogits}\big(\text{logits}[\text{mask}],\ \text{targets}[\text{mask}]\big)
$$

**实现细节**：
- **唯一使用 BCEWithLogits** 的 stage（其他均为 CrossEntropy）
- 每个 touch slot 输出 `[16]` logits，target 是 `[16]` 多热编码
- 只在 `touch_pattern_mask == True` 的位置计算

```python
pred = logits[valid].reshape(-1, TOUCH_PATTERN_NUM_ZONES)
tgt = pattern_targets[valid].reshape(-1, TOUCH_PATTERN_NUM_ZONES).float()
return F.binary_cross_entropy_with_logits(pred, tgt)
```

---

## Loss 类型汇总

| Loss 类型 | 使用的 Stage |
|-----------|-------------|
| `CrossEntropy` (多分类，161K vocab) | Stage 1 |
| `CrossEntropy` (3 分类 per zone) | Stage 2 (Touch) |
| `CrossEntropy` (78 vocab, autoregressive) | Stage 3 (Slide) |
| `CrossEntropy` (8-class + 12-class 双头) | Stage 4 (Hold), Stage 5 (TouchHold) |
| `CrossEntropy` (78 vocab, bidirectional) | Stage 6 (Star) |
| `CrossEntropy` (二分类 per position/zone) | Stage 7 (Break), Stage 8 (Spike) |
| `BCEWithLogits` (多标签 16 分类) | TouchPattern |

所有 stage 都通过 `ignore_index` 或 mask 过滤掉非目标位置。所有辅助 loss 均使用 `logits.sum() * 0.0` 作为零梯度占位。
