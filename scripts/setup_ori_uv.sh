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
Optional env: CUDA_HOME, MAX_JOBS (default 4), UV_VERSION.
ORI_PYPI_INDEX: Python package mirror (default Tsinghua).
ORI_TORCH_INDEX: optional cu128 wheel index; unset uses official PyTorch backend.
ORI_UV_BOOTSTRAP: auto (default), pypi, or official. pypi needs python3 + pip.
UV_PYTHON_INSTALL_MIRROR: optional Python release mirror (must include 3.12.3).
UV_INSTALLER_URL: optional installer script URL for official bootstrap mode.
HTTPS_PROXY / HTTP_PROXY: proxy for remaining GitHub/network downloads.
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
ORI_PYPI_INDEX=${ORI_PYPI_INDEX:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}
ORI_TORCH_INDEX=${ORI_TORCH_INDEX:-}
ORI_UV_BOOTSTRAP=${ORI_UV_BOOTSTRAP:-auto}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-120}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-5}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
# Used by build subprocesses that invoke pip rather than uv.
export PIP_INDEX_URL="$ORI_PYPI_INDEX"
case "$ORI_UV_BOOTSTRAP" in
    auto|pypi|official) ;;
    *) echo 'ORI_UV_BOOTSTRAP must be auto, pypi or official' >&2; exit 2 ;;
esac
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
    use_pypi=false
    if [[ $ORI_UV_BOOTSTRAP != official ]] && command -v python3 >/dev/null && python3 -m pip --version >/dev/null 2>&1; then
        use_pypi=true
    elif [[ $ORI_UV_BOOTSTRAP == pypi ]]; then
        echo 'PyPI 安装 uv 需要 python3 和 pip；Docker root 可执行 apt-get install -y python3-pip。' >&2
        exit 2
    fi
    if [[ $use_pypi == true ]]; then
        uv_requirement=uv
        [[ -z ${UV_VERSION:-} ]] || uv_requirement="uv==$UV_VERSION"
        # --target does not install into or replace system Python packages.
        python3 -m pip install --index-url "$ORI_PYPI_INDEX" --only-binary=:all: \
            --no-deps --upgrade --target "$TOOLS_ROOT/uv/bootstrap" "$uv_requirement"
        cp -- "$TOOLS_ROOT/uv/bootstrap/bin/uv" "$UV_EXE"
        cp -- "$TOOLS_ROOT/uv/bootstrap/bin/uvx" "$TOOLS_ROOT/uv/uvx"
        chmod +x "$UV_EXE" "$TOOLS_ROOT/uv/uvx"
    else
        installer=$(mktemp)
        if [[ -n ${UV_INSTALLER_URL:-} ]]; then
            installer_url=$UV_INSTALLER_URL
        elif [[ -n ${UV_VERSION:-} ]]; then
            installer_url="https://astral.sh/uv/${UV_VERSION}/install.sh"
        else
            installer_url=https://astral.sh/uv/install.sh
        fi
        curl --fail --location --retry 3 "$installer_url" -o "$installer"
        UV_UNMANAGED_INSTALL="$TOOLS_ROOT/uv" sh "$installer"
        rm -- "$installer"
    fi
fi
"$UV_EXE" --version
# uv honors UV_PYTHON_INSTALL_MIRROR; no silent Python version change on mirror 404.
"$UV_EXE" python install 3.12.3
if [[ ! -f $VENV_DIR/.ori-uv-environment ]]; then
    "$UV_EXE" venv --python 3.12.3 --managed-python "$VENV_DIR"
    touch "$VENV_DIR/.ori-uv-environment"
fi

# Override only the known LeRobot/ORI recipe disagreements. In particular,
# NumPy must remain a constraint so incompatible downstream packages are rejected.
OVERRIDES="$VENV_DIR/ori-overrides.txt"
cat > "$OVERRIDES" <<'OVERRIDES_END'
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchcodec==0.6.0
datasets==3.6.0
OVERRIDES_END
CONSTRAINTS="$VENV_DIR/ori-constraints.txt"
cat > "$CONSTRAINTS" <<'PINS'
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0+cu128
torchcodec==0.6.0
datasets==3.6.0
transformers==4.51.3
numpy==1.26.4
# NumPy 1.x-compatible Arrow wheel; verified separately from the reference env.
pyarrow==20.0.0
# Versions verified in the existing training environment.
scipy==1.14.1
pandas==3.0.5
numpydantic==1.8.1
opencv-python==4.11.0.86
opencv-python-headless==4.11.0.86
setuptools>=77,<81
packaging>=24.2,<26
flash-attn==2.8.3
PINS
TORCH_SOURCE_ARGS=(--torch-backend cu128)
if [[ -n $ORI_TORCH_INDEX ]]; then
    # --torch-backend ignores custom indexes for torch packages, so omit it here.
    unset UV_TORCH_BACKEND
    TORCH_SOURCE_ARGS=(--index "$ORI_TORCH_INDEX")
fi
uv_install() {
    "$UV_EXE" pip install --python "$PYTHON_EXE" --overrides "$OVERRIDES" --constraints "$CONSTRAINTS" \
        --default-index "$ORI_PYPI_INDEX" "${TORCH_SOURCE_ARGS[@]}" "$@"
}
# Fetch numerical packages from the general mirror, not the CUDA wheel index.
"$UV_EXE" pip install --python "$PYTHON_EXE" --constraints "$CONSTRAINTS" \
    --default-index "$ORI_PYPI_INDEX" \
    numpy scipy pyarrow pandas numpydantic setuptools wheel packaging ninja psutil einops
"$PYTHON_EXE" - <<'PY_NUMERIC'
import numpy as np
import scipy
import pyarrow as pa
import pandas as pd
from scipy.optimize import linear_sum_assignment
assert linear_sum_assignment([[1, 5], [6, 2]])[1].tolist() == [0, 1]
np.testing.assert_array_equal(pa.array(np.arange(3)).to_numpy(), np.arange(3))
print('Numerical imports ready:', np.__version__, scipy.__version__, pa.__version__, pd.__version__)
PY_NUMERIC
uv_install 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' 'torchaudio==2.8.0+cu128'
"$PYTHON_EXE" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA 不可用，请检查驱动和 GPU 可见性。'
x = torch.ones(1, device='cuda'); print('CUDA ready:', torch.cuda.get_device_name(), x.item())
PY

# Use the submodule commits recorded in this checkout, not changing upstream HEADs.
git submodule update --init --recursive
# Same release, fetched from PyPI/mirror instead of the GitHub tag archive.
# Resolve training and MoGe dependencies together to preserve shared constraints.
uv_install 'lerobot==0.4.2' -e "$REPO_ROOT" \
    -e "$REPO_ROOT/lingbotvla/models/vla/vision_models/MoGe" 'gradio<6'
"$PYTHON_EXE" - <<'PY_DATA'
import datasets
from transformers import PreTrainedModel
from lingbotvla.data.vla_data.base_dataset import VLADataset
print('Training data imports ready:', datasets.__version__)
PY_DATA
# Match official install.sh: do not pull lingbot-depth's older torch/xformers pins.
uv_install --no-deps -e "$REPO_ROOT/lingbotvla/models/vla/vision_models/lingbot-depth"
uv_install 'flash-attn==2.8.3' --no-build-isolation

# Verify environment and kernels, without loading model weights or training.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_EXE" - "$REPO_ROOT" "$CONSTRAINTS" <<'PY'
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
# Read the installation constraints directly; no second version list to drift.
from packaging.requirements import Requirement
for line in Path(sys.argv[2]).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    requirement = Requirement(line)
    actual = metadata.version(requirement.name)
    assert requirement.specifier.contains(actual), (requirement.name, actual, str(requirement.specifier))
    print(f'{requirement.name}=={actual}')
assert metadata.version('lerobot') == '0.4.2'
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
