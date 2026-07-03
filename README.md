# MaiGenerator v2 — 设计文档

## 概述

MaiGenerator v2 是一个基于 Transformer 的 maimai 谱面生成系统。核心思路：将谱面编码为离散 token 序列，用自回归模型逐 token 预测，实现端到端谱面生成。

---

## 一、Token 体系

### 1.1 基础 Token 词表（87 个，ID 0~86）

```
 0-4:   特殊控制  [PAD] [BOS] [EOS] [SEP] [MASK]
 5-15:  拍号      div_1 ~ div_384
16:     休止符    [RST]
17:     时长标记  [DUR]  (后跟 dur_num + dur_den)
18-25:  TAP       tap_1 ~ tap_8
26-33:  BREAK     brk_1 ~ brk_8
34-41:  HOLD      hld_1 ~ hld_8
42-49:  SLIDE     sld_1 ~ sld_8
50-51:  Slide 控制 [SLD_BEG] [SLD_END]
52-53:  同时押控制 [SIM_BEG] [SIM_END]
54-86:  TOUCH     33 区: A1-A8(8) + B1-B8(8) + C(1) + D1-D8(8) + E1-E8(8)
                  C1-C8 在解析阶段归一化为 C（实机只有一个中心触控区）
87:     SIM_COUNT_2
88-107: 时长参数  dur_num_{1..16}, dur_den_{1..64}
108-121:Slide 类型 sld_type_-, >, <, ^, v, p, q, s, z, w, V, pp, qq, *
122:    FIREWORK  烟花标记
123:    FAKE_EACH 假双押
124:    EX_NOTE   EX 音符标记
```

### 1.2 Config Token 词表（164,208 个，ID 256~164,463）

单时间槽完整配置的紧凑编码。基于 **人类双手约束** 枚举所有合法配置。

#### 约束模型

```
每只手状态（互斥）:
  - 空闲
  - 按钮动作: press / hold_start / hold_ongoing / slide_start / slide_end
  - 触控动作: 一组触控区 (max 2 zones, 不可相邻)

双手分配（总和 ≤ 2）:
  rest            0 按钮 + 0 触控
  1 手按钮        1 按钮 + 0 触控
  1 手触控        0 按钮 + 1 触控
  2 手全按钮      2 按钮 + 0 触控
  1 按钮+1 触控   1 按钮 + 1 触控
  (2 按钮+触控 ❌ 需要 3 手)
```

#### 每位置状态（简化，不含时长/break）

```
按钮位置 (1-8):
  empty=1, press=1, hold_start=1, hold_ongoing=1, slide_start=1, slide_end=1
  → 每位置 5 种 active 状态

触控区 (33 区):
  empty=1, touch=1, touch_hold_start=1, touch_hold_ongoing=1
  → 每 zone 3 种 active 状态
  → 2 zone 时不可相邻（相邻图案留给 Stage 2 细化）
```

#### 配置空间

| 配置类型 | 组合数 |
|---------|--------|
| rest | 1 |
| 1 手按钮 | 40 |
| 1 手触控 (≤2 zones, 不相邻) | 3,987 |
| 2 手全按钮 | 700 |
| 1 按钮 + 1 触控 | 159,480 |
| **总计** | **164,208** |

> Config Token 范围: `256 ~ 164,463` (18 bits)
> 后续放开 3+ touch zones / 相邻触控时扩展

---

## 二、分阶段生成策略

模型采用 **级联生成 (Cascaded Generation)** ，分 4 个阶段逐步细化谱面：

```
Stage 1: 粗粒度结构
  ├─ 每时间槽生成 1 个 Config Token (65 万选 1)
  ├─ 决定: 按钮状态、触控状态 (max 2 zones)、slide 起止、hold 持续
  └─ 不关心: break/firework、具体时长、slide 路径细节

Stage 2: Touch 细化
  ├─ 展开 Stage 1 中标记的触控配置
  ├─ 追加更多 touch zone (放开 2 限制)
  ├─ 补充 touch hold 持续时间
  └─ 补充 touch slide (wifi) 路径

Stage 3: Break 判定
  ├─ 对 Stage 1/2 中每个 press/hold_start 判定是否为 break
  └─ 输出 brk 标记 token

Stage 4: Firework 判定
  └─ 对每个 touch 判定是否带烟花特效
```

### 为什么分阶段？

```
全量枚举: 按钮(6^8) × 触控(4^41) → 天文数字，不可能单 token
Stage 1 精简: 按钮(1057) × 触控≤2(13284) = 65 万，可单 token
Stage 2-4: 逐 token 细化，模型只需关注局部决策
```

### 时长 (Duration) 处理

- 时长 **不编码在 Config Token 中**
- hold_start / touch_hold_start 后紧跟 `[DUR] [num] [den]` 三个独立 token
- 模型在 Stage 1 只需决定 "这里开始一个 hold"，时长由后续 token 指定
- 这避免了 89 种时长 × 每位置的组合爆炸

---

## 三、Token 序列示例

```
输入: (173){4}1h[2:1],2/5,Ch[4:1],E

Stage 1 (粗结构):
  [BOS] div_4
  cfg_btn1_hld1         ← 1 手: 位置1 hold_start
  [DUR] 2 1             ← 时长
  cfg_btn2_pair_2_5     ← 2 手: 位置2 press + 位置5 press
  cfg_mix_btn0_tch_C    ← 1 手触控: C 区 touch_hold_start
  [DUR] 4 1
  [EOS]

Stage 2 (touch 细化，此处无额外 touch):
  (C 区 touch 已在 Stage 1 覆盖，无需追加)

Stage 3 (break 判定):
  [MASK] [MASK] ...      ← 对应每个 press/hold_start 位置的 break 标记
  (此处无 break)

Stage 4 (firework 判定):
  (此处无 firework)
```

---

## 四、关键设计决策

| 决策 | 理由 |
|------|------|
| 双手约束 | 人类只有两只手，大幅压缩配置空间 |
| Config Token 不含时长 | 避免 89× 膨胀，时长独立 token |
| Slide 中段不占手 | 滑动手已经在 slide 起点被占用 |
| Touch = 1 手 | 多指触控算一只手，zone 数量另计 |
| 分阶段生成 | 粗→细，每阶段决策空间可控 |
| max 2 touch (Stage 1) | 覆盖 95%+ 谱面，后续阶段可扩展 |

---

## 五、文件结构

```
Tokenizer/
  MaiChartTokenizer.py   谱面 tokenizer (simai ↔ token)
  MaiTrackTokenizer.py   音频 tokenizer (EnCodec)
  config_vocab.py        Config 词表 (164,208 个单时间槽配置)

utils/
  bpm_detector.py        BPM 自动检测

scripts/
  count_configs.py       配置空间计算 (无约束)
  count_configs_2hand.py 配置空间计算 (双手约束)
  count_final.py         最终配置空间 (2手+多指touch)
  print_tokenizer_test.py 谱面 tokenize 测试输出

tests/
  test_chart_tokenizer.py 57 项单元测试

model.py                 模型定义
```

---

## 六、开发路线

- [x] MaiChartTokenizer (simai ↔ base tokens)
- [x] 基础 token 词表 (0-132)
- [x] 配置空间建模 & 计数
- [x] BPM 自动检测
- [ ] Config Token 枚举 & 编码器 (256~652228)
- [ ] Stage 1 模型训练 (粗结构生成)
- [ ] Stage 2-4 模型训练 (细化)
- [ ] 音频条件注入 (MaiTrackTokenizer + cross-attention)
- [ ] 端到端谱面生成 pipeline
