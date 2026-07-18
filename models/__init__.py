"""
models — 多阶段谱面生成 Transformer 网络

Stage 1: chart — 音频+条件 → 扁平谱面 token
Stage 2: hold  — 自回归补全 hold 长度
Stage 3: slide — 自回归补全 slide 路径 (独立 vocab)
Stage 4: break — 逐 token 预测 break (双向)
Stage 5: ex    — 逐 token 预测 ex (双向, 仅 DX)
"""

from models.stage1_chart import Stage1ChartModel
from models.stage2_hold import Stage2HoldModel
from models.stage3_slide import Stage3SlideModel
from models.stage4_break import Stage4BreakModel
from models.stage5_ex import Stage5ExModel
