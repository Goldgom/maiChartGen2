# maiG_v2 环境配置指南

## 环境概览

| 项目 | 版本 |
|------|------|
| Python | 3.10.13 |
| PyTorch | 2.12.0+cu130 |
| CUDA | 13.0 |
| cuDNN | 9.2.0 |
| 包管理器 | conda (miniforge3) + pip |

---

## 方式一：一键安装（推荐）

```bash
# 1. 创建 conda 环境
conda create -n maiG python=3.10 -y
conda activate maiG

# 2. 安装 PyTorch（CUDA 13.0）
pip install torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# 3. 安装其余依赖
pip install -r requirements.txt
```

---

## 方式二：完全复现当前环境

```bash
# 1. 创建 conda 环境
conda create -n maiG python=3.10 -y
conda activate maiG

# 2. 安装全部依赖（完整复现）
pip install -r requirements_full.txt
```

---

## 方式三：手动安装关键依赖

如果上述方式遇到兼容性问题，按以下顺序手动安装：

### 1. PyTorch 生态
```bash
pip install torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install triton==3.7.0
pip install torchsde==0.2.6
```

### 2. Transformers
```bash
pip install transformers==5.8.1 tokenizers==0.22.2
pip install sentencepiece==0.2.1 huggingface_hub==1.14.0
pip install safetensors==0.7.0
```

### 3. CUDA 工具链
```bash
pip install cupy-cuda12x==12.3.0
pip install nvidia-cuda-cupti==13.0.85 nvidia-cuda-nvrtc==13.0.88
pip install nvidia-cuda-runtime==13.0.96 nvidia-cudnn-cu13==9.20.0.48
```

### 4. 科学计算
```bash
pip install numpy==1.26.4 scipy==1.15.3 einops==0.8.2
```

### 5. 工具库
```bash
pip install PyYAML==6.0.1 tqdm rich==15.0.0
```

---

## 验证安装

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('cuDNN:', torch.backends.cudnn.version())

import transformers
print('Transformers:', transformers.__version__)

import triton
print('Triton:', triton.__version__)

print('ALL OK!')
"
```

期望输出：
```
PyTorch: 2.12.0+cu130
CUDA available: True
CUDA version: 13.0
cuDNN: 92000
Transformers: 5.8.1
Triton: 3.7.0
ALL OK!
```

---

## 现有文件说明

| 文件 | 用途 |
|------|------|
| `requirements.txt` | 核心依赖（精简版，推荐使用） |
| `requirements_full.txt` | 完整依赖列表（复现当前环境） |

---

## 注意事项

1. **CUDA 版本**：本环境使用 CUDA 13.0，确保目标机器安装了 NVIDIA 驱动 ≥ 545.23.06
2. **GCC**：部分包需要 GCC 编译器，Ubuntu 可运行 `sudo apt install build-essential -y`
3. **磁盘空间**：环境约需 15-20GB（含 PyTorch 和 CUDA 包）
4. **清华镜像**：如遇下载慢，可使用清华源加速：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
