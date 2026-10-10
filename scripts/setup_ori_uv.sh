#!/usr/bin/env bash
# Set up the official ORI Python/CUDA package stack without changing system Python.
set -Eeuo pipefail
trap 'echo "安装失败（第 ${LINENO} 行）。修复报错后可重新运行本脚本。" >&2' ERR

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/setup_ori_uv.sh [absolute-tools-directory]
Default: $HOME/.local/share/lingbot-vla-tools
Layout: TOOLS/uv/uv, TOOLS/python/, TOOLS/cache/; repository/.venv/
Requires: Linux x86_64, NVIDIA GPU/driver, CUDA toolkit 12.8 (nvcc),
          git, curl, g++, make, ffmpeg and FFmpeg shared libraries.
Optional env: CUDA_HOME, MAX_JOBS (default 4), UV_VERSION (installer version).
Does not install drivers, download model weights, or start training.
HELP
    exit 0
fi
[[ $# -le 1 ]] || { echo '只接受一个工具目录参数。' >&2; exit 2; }
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TOOLS_ROOT=${1:-"$HOME/.local/share/lingbot-vla-tools"}
[[ $TOOLS_ROOT = /* ]] || { echo '工具目录必须是绝对路径。' >&2; exit 2; }
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo '本脚本面向 Linux x86_64 NVIDIA 训练服务器。' >&2; exit 2; }
for executable in git curl g++ make ffmpeg nvidia-smi; do
    command -v "$executable" >/dev/null || { echo "缺少系统命令：$executable" >&2; exit 2; }
done
if [[ -z ${CUDA_HOME:-} ]]; then
    if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
        export CUDA_HOME=/usr/local/cuda-12.8
    elif command -v nvcc >/dev/null; then
        CUDA_HOME=$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v nvcc)")")")
        export CUDA_HOME
    else
        echo '需要 CUDA toolkit 12.8；请安装后设置 CUDA_HOME。nvidia-smi 显示 CUDA 版本不代表已安装 nvcc。' >&2
        exit 2
    fi
fi
[[ -x $CUDA_HOME/bin/nvcc ]] || { echo "找不到 $CUDA_HOME/bin/nvcc" >&2; exit 2; }
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12\.8' || { echo '请将 CUDA_HOME 指向 CUDA toolkit 12.8。' >&2; exit 2; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

mkdir -p -- "$TOOLS_ROOT/uv" "$TOOLS_ROOT/python" "$TOOLS_ROOT/cache"
TOOLS_ROOT=$(cd -- "$TOOLS_ROOT" && pwd)
UV_EXE="$TOOLS_ROOT/uv/uv"
export UV_PYTHON_INSTALL_DIR="$TOOLS_ROOT/python"
export UV_PYTHON_BIN_DIR="$TOOLS_ROOT/python/bin"
export UV_CACHE_DIR="$TOOLS_ROOT/cache"
export UV_LINK_MODE=copy
export MAX_JOBS=${MAX_JOBS:-4}
export PATH="$TOOLS_ROOT/uv:$CUDA_HOME/bin:$PATH"
# Ignore an activated unrelated environment; every uv install explicitly targets this venv.
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_EXE="$VENV_DIR/bin/python"
cd -- "$REPO_ROOT"
if [[ -e $VENV_DIR && ! -f $VENV_DIR/.ori-uv-environment ]]; then
    echo "$VENV_DIR 已存在且不是本脚本建立的环境；请先移走它或使用干净 checkout。" >&2
    exit 2
fi
if [[ ! -x $UV_EXE ]]; then
    installer=$(mktemp)
    if [[ -n ${UV_VERSION:-} ]]; then
        installer_url="https://astral.sh/uv/${UV_VERSION}/install.sh"
    else
        installer_url=https://astral.sh/uv/install.sh
    fi
    curl --fail --location --retry 3 "$installer_url" -o "$installer"
    UV_UNMANAGED_INSTALL="$TOOLS_ROOT/uv" sh "$installer"
    rm -- "$installer"
fi
"$UV_EXE" --version
"$UV_EXE" python install 3.12.3
if [[ ! -f $VENV_DIR/.ori-uv-environment ]]; then
    "$UV_EXE" venv --python 3.12.3 --managed-python "$VENV_DIR"
    touch "$VENV_DIR/.ori-uv-environment"
fi

# LeRobot 0.4.2 metadata conflicts with ORI's explicit dependency pins.
# uv overrides deliberately preserve ORI's versions instead of downgrading torch.
OVERRIDES="$VENV_DIR/ori-overrides.txt"
cat > "$OVERRIDES" <<'PINS'
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0+cu128
torchcodec==0.6.0
datasets==3.6.0
transformers==4.51.3
numpy==1.26.4
opencv-python==4.11.0.86
opencv-python-headless==4.11.0.86
PINS
uv_install() {
    "$UV_EXE" pip install --python "$PYTHON_EXE" --overrides "$OVERRIDES" --torch-backend cu128 "$@"
}
uv_install 'setuptools>=77,<81' wheel packaging ninja psutil einops
uv_install 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' 'torchaudio==2.8.0+cu128'
"$PYTHON_EXE" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA 不可用，请检查驱动和 GPU 可见性。'
x = torch.ones(1, device='cuda'); print('CUDA ready:', torch.cuda.get_device_name(), x.item())
PY

# Use the submodule commits recorded in this checkout, not changing upstream HEADs.
git submodule update --init --recursive
uv_install 'lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz' -e "$REPO_ROOT"
uv_install -e "$REPO_ROOT/lingbotvla/models/vla/vision_models/MoGe" 'gradio<6'
# Match official install.sh: do not pull lingbot-depth's older torch/xformers pins.
uv_install --no-deps -e "$REPO_ROOT/lingbotvla/models/vla/vision_models/lingbot-depth"
uv_install 'flash-attn==2.8.3' --no-build-isolation

# Verify environment and kernels, without loading model weights or training.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_EXE" - "$REPO_ROOT" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path
import torch
import torchvision
import torchaudio
from torchcodec.decoders import VideoDecoder
from flash_attn import flash_attn_func
from torch.nn.attention.flex_attention import flex_attention
import lingbotvla
from lingbotvla.data.vla_data.base_dataset import VLADataset
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
import tasks.vla.train_lingbotvla
root = Path(sys.argv[1]).resolve()
assert Path(lingbotvla.__file__).resolve().is_relative_to(root), lingbotvla.__file__
for name, expected in [('torch','2.8.0+cu128'), ('torchvision','0.23.0+cu128'),
                       ('torchaudio','2.8.0+cu128'), ('torchcodec','0.6.0'),
                       ('lerobot','0.4.2'), ('datasets','3.6.0'),
                       ('transformers','4.51.3'), ('numpy','1.26.4'), ('flash-attn','2.8.3')]:
    actual = metadata.version(name)
    assert actual == expected, (name, actual, expected)
    print(f'{name}=={actual}')
q = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
flash_attn_func(q, q, q).float().sum().backward()
assert torch.isfinite(q.grad).all()
f = torch.randn(1, 2, 16, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
torch.compile(flex_attention)(f, f, f).float().sum().backward()
assert torch.isfinite(f.grad).all()
print('PASS: ORI imports, TorchCodec import, FlashAttention/FlexAttention forward and backward.')
print('This does not validate full-model training capacity or the optional depth-model pipeline.')
PY
"$UV_EXE" pip freeze --python "$PYTHON_EXE" > "$VENV_DIR/installed-requirements.txt"
cat <<EOF
环境配置完成。
uv:     $UV_EXE
Python: $UV_PYTHON_INSTALL_DIR
venv:   $VENV_DIR
激活：  source "$VENV_DIR/bin/activate"
训练：  在 "$REPO_ROOT" 中使用 python -m torch.distributed.run ...
依赖快照：$VENV_DIR/installed-requirements.txt
说明：LeRobot/lingbot-depth 的原始元数据约束与官方 VLA 版本冲突；
      本脚本显式使用官方 VLA 版本，未修改第三方包元数据。
EOF
