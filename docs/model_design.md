# MaiGenerator v2 — 四阶段模型设计

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        maiGenerator                              │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Stage 1  │──>│ Stage 2  │──>│Stage 2.5 │──>│ Stage 3  │──>│ Stage 4  │    │
│  │ 粗结构   │   │ Touch细化│   │Slide形状 │   │ Break判定│   │ 烟花判定  │    │
│  │ maiG.py  │   │ touchG.py│   │slideG.py│   │ breakG.py│   │ spikeG.py│    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │              │              │              │              │            │
│       v              v              v              v              v            │
│   Config Token   Touch Zone     Slide Path     Break Labels   Firework        │
│   +DUR/SLIDE     展开序列       conn+mid        per-position    per-touch      │
│   161K vocab     33 zone ×3     auto-reg                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: 粗结构生成 (`maiG.py`)

```
                         ┌──────────────────────────┐
                         │   Audio Preprocessing     │
                         │   shared audio encoder    │
                         │                           │
                         │  onset_strength  [T]      │
                         │  chroma          [T, 12]  │
                         │  centroid        [T]      │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │ onset/chroma    │ onset/chroma     │  ...
                    │ centroid [t=0]  │ centroid [t=1]   │
                    └────────┬────────┴────────┬────────┘
                             │                 │
                             v                 v
┌──────────────────────────────────────────────────────────────────┐
│                  Stage 1 — maiG  (Transformer Decoder)            │
│                                                                   │
│  每步输入 (5 路融合):                                               │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ ① Token Embed      [D]   当前 config token 语义           │    │
│  │ ② Beat Pos Embed   [D]   bar/beat/sub 节拍感知位置        │    │
│  │ ③ Timing Embed     [D]   距上次 press/hold/slide/touch    │    │
│  │ ④ Audio Memory     [D]   dual-stream fused context       │    │
│  │ ⑤ Condition        [D]   BPM + Level + Genre (AdaLN)     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────────┐                        │
│  │          Decoder Layer               │                        │
│  │  ┌──────────────────────────────┐    │                        │
│  │  │ Self-Attention (Causal)      │    │  ← 历史 token 序列     │
│  │  │ 看到: [BOS, tok_0, ... t-1]  │    │                        │
│  │  ├──────────────────────────────┤    │                        │
│  │  │ Cross-Attention (Audio)      │    │  ← fused audio memory    │
│  │  │ Query: 当前 hidden            │    │                        │
│  │  │ K/V: 全曲音频特征             │    │                        │
│  │  ├──────────────────────────────┤    │                        │
│  │  │ FFN + AdaLN (Cond)           │    │  ← BPM/Level/Genre     │
│  │  └──────────────────────────────┘    │                        │
│  │             × N layers               │                        │
│  └──────────────────┬───────────────────┘                        │
│                     │                                            │
│                     v                                            │
│  ┌──────────────────────────────────────┐                        │
│  │          Output Head                  │                        │
│  │  ┌────────────────────────────┐       │                        │
│  │  │ Single LM Head             │       │  ← [1] Duration as     │
│  │  │ over 161K+ vocab           │       │    next-token: DUR/    │
│  │  │ (Config + DUR + num + den) │       │    num/den are inline  │
│  │  └────────────────────────────┘       │    tokens in vocab     │
│  └──────────────────────────────────────┘                        │
│                                                                   │
│  Loss: CE over all tokens (standard LM)   ← [6] no separate heads│
│  Cross-Attn: FULL audio at every step      ← [2] no truncation   │
│  Audio Encoder: dual-stream + fusion       ← shared across stages│
│  Cross-Attn: FULL fused audio memory       ← [2] no truncation   │
└──────────────────────────────────────────────────────────────────┘
```

### 输入嵌入详解

```
每时间槽的输入向量由 3 路嵌入求和 + 音频交叉注意力:

  ┌─────────────────────────────────────────────┐
  │  x = token_embed(tok_t)                     │  ← 当前token语义
  │    + beat_pos_embed(bar, beat, sub)         │  ← 音乐结构位置
  │    + timing_embed(dist_press, dist_hold,    │  ← 相对节奏间距
  │                   dist_slide, dist_touch)   │
  │                                             │
  │  cross_attn(audio_memory)                   │  ← dual-stream fused memory
  │  cond_adaln(bpm, level, genre)              │  ← 条件注入
  └─────────────────────────────────────────────┘

Beat Pos Embed (4 分量):
  bar_idx     → Embed  → 第几个小节 (0-511)
  beat_in_bar → Embed  → 小节内第几拍 (0-3)
  sub_beat    → Embed  → 拍内第几分 (0-63)
  global_pos  → Embed  → 绝对位置 (fallback)

Timing Embed (4 分量):
  dist_press  → Embed  → 距上次 tap/break 的 slot 数
  dist_hold   → Embed  → 距上次 hold_start 的 slot 数
  dist_slide  → Embed  → 距上次 slide_start 的 slot 数
  dist_touch  → Embed  → 距上次 touch 的 slot 数

Audio Stream A (Discrete):
  EnCodec tokens → token embed + lightweight Transformer
    → frame-level discrete memory

Audio Stream B (Spectral):
  onset_strength → Linear
  chroma         → MLP
  centroid       → Linear
    → beat-aligned spectral memory

Fusion:
  dual-stream memory + BPM prior → shared audio memory
  local window pooling → per-slot context
```

### Stage 1 推理过程

```
预处理:
  Audio → extract_features(track.mp3, BPM, subdiv=64)
       → onset[T], chroma[T,12], centroid[T]

推理循环:
  Step 0:  tok=[BOS]  dist=(0,0,0,0)  pos=(bar=0,beat=0,sub=0)
           + audio[t=0]  + cond(BPM=173, Level=12.4)
           → predict Config_0
  Step 1:  tok=[BOS, Config_0]  dist=(1,0,0,0)  pos=(bar=0,beat=0,sub=1)
           + audio[t=1]  + cond
           → predict Config_1
  ...
           if Config_t has hold_start → predict DUR, dur_num, dur_den
           if Config_t has slide_start → predict slide_start_pos, dur_num, dur_den, slide_end_pos
           (Stage 1 only commits start/end/duration; Stage 2.5 fills intermediate path)  [1]
```

---

## Stage 2: Touch 细化 (`touchG.py`)

```
┌──────────────────────────────────────────────────────────────┐
│                       Stage 2 — touchG                        │
│                                                              │
│  输入                                                         │
│  ├─ Stage 1 Config Token 序列   [B, T]                       │
│  ├─ Stage 1 Hidden States      [B, T, D]  (复用Stage1编码器)  │
│  ├─ Audio Memory               [B, T_a, D] (双流融合结果)      │
│  └─ Touch Expansion Map        (预计算映射表)                  │
│                                                              │
│  ┌─────────────────────────────────────┐                     │
│  │      Touch Expansion Decoder        │                     │
│  │  ┌─────────────────────────────┐    │                     │
│  │  │ Embed: Config Token + Pos   │    │                     │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ Self-Attention (Bidirectional)│  │  ← [5] Full sequence│
│  │  │ 看到: 全部 Config Token 序列  │    │    both directions │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ Cross-Attn (Stage1 Hidden)  │    │                     │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ FFN                         │    │                     │
│  │  └─────────────────────────────┘    │                     │
│  │            × N layers               │                     │
│  └──────────────────┬──────────────────┘                     │
│                     │                                        │
│                     v                                        │
│  ┌─────────────────────────────────────┐                     │
│  │         Touch Zone Head             │                     │
│  │  33-way + 3-state classifier        │                     │
│  │  per expansion step                 │                     │
│  └─────────────────────────────────────┘                     │
│                                                              │
│  输出 (仅对包含touch的Config Token)                            │
│  └─ Touch Zone 序列    展开后的触控区列表                      │
│                                                              │
│  示例:                                                        │
│  Config=[tch E1 touch]                                        │
│    → 展开: [E1]                    (单点,不改)                 │
│  Config=[tch E1 touch, tch B1 touch]  (非相邻,不改)           │
│    → 展开: [E1, B1]                                         │
│  Config=[tch E1 touch, tch E2 touch]                          │
│    → 展开: [E1, E2]                (Stage 1 只允许非相邻2点)  │
│                                                              │
│  Loss: CrossEntropy(zone_id) + CrossEntropy(zone_state)       │
└──────────────────────────────────────────────────────────────┘
```

---

## Stage 2.5: Slide 形状补全 (`slideG.py`)

```
┌──────────────────────────────────────────────────────────────┐
│                     Stage 2.5 — slideG                        │
│                                                              │
│  输入                                                         │
│  ├─ Start Pos       (1-8)    来自 Stage 1 slide_start_pos    │
│  ├─ End Pos         (1-8)    来自 Stage 1 slide_end_pos      │
│  ├─ Duration        (num,den) 来自 Stage 1 dur_num/dur_den   │
│  ├─ Audio Context   [T_win,D]   dual-stream fused local window │
│  └─ Stage1 Hidden   [T, D]    (可选 cross-attn)               │
│                                                              │
│  ┌─────────────────────────────────────┐                     │
│  │    Slide Path Decoder               │                     │
│  │  ┌─────────────────────────────┐    │                     │
│  │  │ Cond: start + end + dur     │    │                     │
│  │  │       → [B, 1, D]          │    │                     │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ Self-Attn (Causal, gen)     │    │                     │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ Cross-Attn (audio + cond)   │    │                     │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ FFN                         │    │                     │
│  │  └─────────────────────────────┘    │                     │
│  │            × N layers               │                     │
│  └──────────────────┬──────────────────┘                     │
│                     │                                        │
│                     v                                        │
│  ┌─────────────────────────────────────┐                     │
│  │     Output: [conn, mid, conn, ...]  │                     │
│  │     alternating connector + button  │                     │
│  │     stops when end button is emitted                      │                     │
│  └─────────────────────────────────────┘                     │
│                                                              │
│  示例:                                                        │
│  start=1, end=5, dur=[4:1]                                   │
│    → generate: [116("-"), 3(btn), 117(">"), 5(btn)]          │
│    → simai: "1-3>5[4:1]"                                     │
│                                                              │
│  Loss: CE over connector + button sequence                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Stage 3: Break 判定 (`breakG.py`)

```
┌──────────────────────────────────────────────────────────────┐
│                       Stage 3 — breakG                        │
│                                                              │
│  输入                                                         │
│  ├─ Stage 1+2 Token 序列       [B, T]                        │
│  ├─ Stage 1 Hidden States      [B, T, D]                     │
│  └─ 按钮位置掩码               [B, T, 8] (哪些位置有press)     │
│                                                              │
│  ┌─────────────────────────────────────┐                     │
│  │         Break Classifier            │                     │
│  │  ┌─────────────────────────────┐    │                     │
│  │  │ Stage1 Hidden + Pos Embed   │    │                     │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ Lightweight MLP / Attn      │    │                     │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ Per-position Binary Head    │    │                     │
│  │  │ (8 positions × 2 classes)   │    │                     │
│  │  └─────────────────────────────┘    │                     │
│  └─────────────────────────────────────┘                     │
│                                                              │
│  输出 (每个时间槽的每个按钮位置)                                │
│  └─ Break Label  [B, T, 8, 2]   is_break? yes/no             │
│                                                              │
│  示例:                                                        │
│  Config=[btn1_press, btn3_press]                              │
│    → Break: [pos1=NO, pos3=YES]   →  3b/1                    │
│  Config=[btn2_press]                                          │
│    → Break: [pos2=NO]              →  2 (普通tap)             │
│                                                              │
│  Loss: BinaryCrossEntropy per position per slot               │
│  (仅对有press的位置计算loss, 空位置mask掉)                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Stage 4: 烟花判定 (`spikeG.py`)

```
┌──────────────────────────────────────────────────────────────┐
│                      Stage 4 — spikeG                         │
│                                                              │
│  输入                                                         │
│  ├─ Stage 1+2 Token 序列       [B, T]                        │
│  ├─ Stage 1 Hidden States      [B, T, D]                     │
│  └─ Touch 位置掩码             [B, T, 33] (哪些zone有touch)    │
│                                                              │
│  ┌─────────────────────────────────────┐                     │
│  │        Firework Classifier          │                     │
│  │  ┌─────────────────────────────┐    │                     │
│  │  │ Stage1 Hidden + Zone Embed  │    │                     │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ Lightweight MLP             │    │                     │
│  │  ├─────────────────────────────┤    │                     │
│  │  │ Per-touch Binary Head       │    │                     │
│  │  │ (33 zones × 2 classes)      │    │                     │
│  │  └─────────────────────────────┘    │                     │
│  └─────────────────────────────────────┘                     │
│                                                              │
│  输出 (每个时间槽的每个触控区)                                  │
│  └─ Firework Label [B, T, 33, 2]  is_firework? yes/no        │
│                                                              │
│  示例:                                                        │
│  Touch=[B1, C]                                                │
│    → Firework: [B1=YES, C=NO]     →  B1f/C                   │
│                                                              │
│  Loss: BinaryCrossEntropy per touch zone per slot             │
│  (仅对有touch的zone计算loss)                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 训练流程

```
                     Raw Chart (simai)
                          │
                          v
                  ┌───────────────┐
                  │  预处理        │
                  │  strip break   │
                  │  strip firework│
                  │  compress touch│
                  └───────┬───────┘
                          │
          ┌───────────────┼───────────────┐
          v               v               v
    Stage1 Target    Stage3 Label    Stage4 Label
   (config tokens)  (break flags)   (firework flags)
          │
          v
    ┌──────────┐
    │ Stage 1  │  训练: Audio → Config + DUR Tokens
    │  maiG    │  [1] Single LM head, DUR/num/den inline
    └────┬─────┘
         │ hidden states 保存
         v
    ┌──────────┐
    │ Stage 2  │  训练: Config → Expanded Touch Zones
    │ touchG   │  [5] Bidirectional self-attention
    └────┬─────┘
         │
         v
    ┌──────────┐
    │Stage 2.5  │  训练: Start+End+Dur → Slide Path
    │ slideG   │  Loss: CE(conn + mid sequence)
    └────┬─────┘
         │
         v
    ┌──────────┐
    │ Stage 3  │  训练: Hidden → Break Labels
    │ breakG   │  Loss: BCE per position
    └────┬─────┘
         │
         v
    ┌──────────┐
    │ Stage 4  │  训练: Hidden → Firework Labels
    │ spikeG   │  Loss: BCE per touch zone
    └──────────┘
```

---

## 推理流程

```
    Audio → dual-stream encoder → Audio Memory [T_audio, D]
                │
                │  ┌─────────────────────────────────────┐
                │  │        Autoregressive Loop           │
                │  │                                     │
                │  │  step t:                            │
                │  │    input = [BOS, tok_0, ..., tok_{t-1}]  ← 历史生成谱面 │
                │  │    cross_attn(audio_memory)          ← 全曲双流融合音频 │
                │  │    BPM/Level/Genre conditioning      ← 条件注入         │
                │  │    Level = 连续值 (e.g. 7.1, 12.4)   ← 可内插难度       │
                │  │         │                           │
                │  │         v                           │
                │  │    ┌──────────────────────┐          │
                │  │    │  Stage 1 (maiG)      │          │
                │  │    │  → Config Token      │          │
                │  │    │  → DUR → num → den   │  [1]    │
                │  │    │    (if hold_start)   │  inline │
                │  │    └──────────┬───────────┘          │
                │  │               │                      │
                │  │    tok_t 加入历史, t += 1             │
                │  └───────────────┼──────────────────────┘
                │                  │
                v                  v
    ┌──────────────────────┐
    │  Stage 2 (touchG)    │  对 touch config 展开为具体 zone
    │  Touch Expansion      │  (单点→多点、相邻→分离等)
    └──────────┬───────────┘
               │
               v
    ┌──────────────────────┐
    │Stage 2.5 (slideG)   │  start + end + dur + audio
    │ Slide Path Generate  │  → [conn, mid, conn, mid, ...]
    └──────────┬───────────┘
               │
               v
    ┌──────────────────────┐
    │  Stage 3 (breakG)    │  对每个 press 判定是否为 break
    │  Break Classification │  (1 → 1b, 2→2b, 1/2→1b/2b)
    └──────────┬───────────┘
               │
               v
    ┌──────────────────────┐
    │  Stage 4 (spikeG)    │  对每个 touch 判定是否烟花
    │  Firework Classify    │  (B1 → B1f)
    └──────────┬───────────┘
               │
               v
    ┌──────────────────────┐
    │  Token → Simai 解码   │  完整谱面输出
    │  MaiChartTokenizer    │
    └──────────────────────┘
```

---

## 模型参数估算

| Stage | 模型 | 词表 | 输出头 | 参数量(估) |
|-------|------|------|--------|-----------|
| 1 | maiG | 161K | Config + Duration×2 | ~60M |
| 2 | touchG | 33+161K | Zone + State | ~20M |
| 3 | breakG | — | 8×2 binary | ~5M |
| 4 | spikeG | — | 33×2 binary | ~5M |
| **Total** | | | | **~90M** |

### Stage 1 输入嵌入维度

| 嵌入 | 维度 | 内容 |
|------|------|------|
| Token Embed | D | Config token 语义 |
| Beat Pos Embed | D | bar(¼) + beat(¼) + sub(¼) + global(¼) |
| Timing Embed | D | press(¼) + hold(¼) + slide(¼) + touch(¼) |
| **求和** | **D** | 3 路融合 |
| Cross-Attention | D | dual-stream fused audio memory |
| Audio Stream A | D | EnCodec token encoder |
| Audio Stream B | D | onset/chroma/centroid encoder |
| Fusion | D | audio memory + local context |
| AdaLN Condition | D | BPM + Level + Genre |

### 双流音频编码

| 特征 | 维度 | 来源 | 含义 |
|------|------|------|------|
| EnCodec tokens | [T_a] | MaiTrackTokenizer | 离散时频结构、瞬态、纹理 |
| onset_strength | [T_s] | librosa.onset | 每个 slot 的起音强度 |
| chroma | [T_s, 12] | librosa.chroma_cqt | 12 维音高色彩 |
| centroid | [T_s] | librosa.spectral_centroid | 频谱重心/音色明暗 |

### 双流融合策略

| 组件 | 输入 | 输出 |
|------|------|------|
| Stream A Encoder | EnCodec tokens | frame-level discrete memory |
| Stream B Encoder | onset/chroma/centroid | beat-aligned spectral memory |
| Fusion Block | A + B + BPM prior | shared audio memory |
| Local Pooling | shared memory | per-slot context |

### 条件注入

| 参数 | 类型 | 编码 |
|------|------|------|
| BPM | float (50~300) | Linear(1→D) → SiLU → Linear(D→D) |
| Level | **float (1.0~15.0)** | **Linear(1→D) → SiLU → Linear(D→D)** |
| Genre | int (0~127) | Embedding(128, D) |

> Level 使用连续值投影而非离散 Embedding，模型可以内插到训练时未见过的难度等级。

> 双流音频编码中，Stream A 负责全局时频与瞬态结构，Stream B 负责局部节奏与和声提示。

> 双流音频编码中，Stream A 负责全局时频与瞬态结构，Stream B 负责局部节奏与和声提示。

---

## Changelog

| Date | Change | Description |
|------|--------|-------------|
| 2024-07 | Slide system | Stage 1 predicts slide start/end positions plus duration; `slideG.py` (Stage 2.5) autoregressively generates intermediate connectors + midpoints from start/end/duration/audio. |
| 2024 | [1] Duration as next-token | Single LM head over full vocab; DUR/num/den are inline tokens instead of parallel heads. Vocab: 0-161511. |
| 2024 | [2] Full audio cross-attn | Cross-attention always sees the complete audio track; no truncation in `generate()`. |
| 2024 | [5] Stage 2 bidirectional | `touchG` self-attention uses **bidirectional** mask (no causal), since full Stage 1 sequence is available. |
| 2024 | [6] Duration loss masking | Standard LM CE loss over all tokens; no separate duration head loss. |
| 2024 | [7] Local audio context | `LocalAudioContext` conv window (W=8) around each slot provides per-position rhythmic context added to token embedding. |
