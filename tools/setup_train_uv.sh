#!/usr/bin/env bash
# Install the V2 training runtime into this checkout's .venv, using direct network access.
set -Eeuo pipefail
trap 'echo "Installation failed at line ${LINENO}; fix the error and rerun this script." >&2' ERR

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
UV_EXE=${UV_EXE:-/root/.local/bin/uv}
VENV_DIR="$REPO_ROOT/.venv"
PYTHON_EXE="$VENV_DIR/bin/python"
[[ -x "$PYTHON_EXE" ]] || { echo "Create .venv with Python 3.12 before running this script." >&2; exit 1; }
"$PYTHON_EXE" -c 'import sys; assert sys.version_info[:2] == (3, 12)'

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONPATH PYTHONHOME
export NO_PROXY='*' no_proxy='*' PYTHONNOUSERSITE=1
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0=''
export GIT_CONFIG_KEY_1=https.proxy GIT_CONFIG_VALUE_1=''
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$VENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export UV_CACHE_DIR="$VENV_DIR/uv-cache"
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=5
export UV_CONCURRENT_DOWNLOADS=4 MAX_JOBS=${MAX_JOBS:-4}
export UV_DEFAULT_INDEX=${V2_PYPI_INDEX:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}
export PIP_INDEX_URL="$UV_DEFAULT_INDEX"

# MoGe's standalone trainer uses MLflow; the VLA teacher runtime does not.
# plyfile 1.1.2 supports NumPy 1.26, whereas 1.1.4 requires NumPy 2.
sed -e '/^mlflow==/d' -e 's/^plyfile==1.1.4$/plyfile==1.1.2/' \
    requirements-depth.txt > "$VENV_DIR/v2-depth-runtime.txt"
cat > "$VENV_DIR/v2-overrides.txt" <<'EOF'
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0+cu128
torchcodec==0.6.0
datasets==3.6.0
accelerate==1.7.0
EOF
cat > "$VENV_DIR/v2-extra-constraints.txt" <<'EOF'
setuptools>=77,<81
opencv-python==4.11.0.86
gradio<6
flash-attn==2.8.3
numpydantic==1.8.1
EOF
# LeRobot's visualization SDK requires NumPy 2; VLA's dataset reader does not use it.
printf '%s\n' rerun-sdk > "$VENV_DIR/v2-excludes.txt"

v2_install() {
    "$UV_EXE" pip install --python "$PYTHON_EXE" --torch-backend cu128 \
        --overrides "$VENV_DIR/v2-overrides.txt" \
        --excludes "$VENV_DIR/v2-excludes.txt" \
        --constraints requirements.txt \
        --constraints "$VENV_DIR/v2-depth-runtime.txt" \
        --constraints "$VENV_DIR/v2-extra-constraints.txt" "$@"
}

v2_install 'setuptools>=77,<81' wheel pip packaging ninja
# Optional local wheels copied from a matching trusted download cache.
# Install them before resolving dependencies to avoid downloading identical large packages.
for wheel in "$VENV_DIR"/local-wheels/torch-*.whl "$VENV_DIR"/local-wheels/triton-*.whl; do
    [[ -f "$wheel" ]] || continue
    "$UV_EXE" pip install --python "$PYTHON_EXE" --no-deps "$wheel"
done
v2_install -r requirements.txt -r "$VENV_DIR/v2-depth-runtime.txt" \
    'lerobot==0.4.2' 'numpydantic==1.8.1' 'gradio<6' \
    -e . -e ./lingbotvla/models/vla/vision_models/MoGe
# Use the repository's Torch stack, retaining the depth model's SDPA fallback.
v2_install --no-deps -e ./lingbotvla/models/vla/vision_models/lingbot-depth

"$PYTHON_EXE" -c 'import setuptools, torch; print("Build prerequisites:", setuptools.__version__, torch.__version__, torch.version.cuda)'
if [[ -z ${FLASH_ATTN_WHEEL:-} && -f "$VENV_DIR/local-wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl" ]]; then
    FLASH_ATTN_WHEEL="$VENV_DIR/local-wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
fi
if [[ -n ${FLASH_ATTN_WHEEL:-} ]]; then
    v2_install --no-deps "$FLASH_ATTN_WHEEL"
else
    v2_install 'flash-attn==2.8.3' --no-build-isolation
fi
"$UV_EXE" pip freeze --python "$PYTHON_EXE" > "$VENV_DIR/installed-requirements.txt"
if "$UV_EXE" pip check --python "$PYTHON_EXE" > "$VENV_DIR/dependency-check.log" 2>&1; then
    echo "Dependency metadata checks passed."
else
    echo "Dependency metadata differences recorded in .venv/dependency-check.log (LeRobot/MDM upstream pins differ from V2)."
fi
cat > "$VENV_DIR/INSTALL-NOTES.txt" <<'EOF'
V2 training runtime, Python 3.12, installed only in this checkout's .venv.
Network downloads use direct connections, with HTTP/SOCKS proxy variables cleared.
Large matching wheels may be copied from the old download cache; environments remain separate.

Core recipe: requirements.txt, CUDA 12.8 Torch wheels.
Runtime changes relative to requirements-depth.txt:
  - Omit MLflow: only used by MoGe's standalone training script, conflicts with PyArrow 21.
  - plyfile 1.1.2: supports the pinned NumPy 1.26.4.
  - numpydantic 1.8.1: NumPy 1.x compatible version used by the old environment.
  - Exclude LeRobot's rerun-sdk visualization dependency (requires NumPy 2).
  - LeRobot Torch/datasets/accelerate requirements overridden to the V2 training recipe.
  - MDM installed without its old Torch 2.6/xformers pins; teacher uses available SDPA path.

See installed-requirements.txt for the installed snapshot and dependency-check.log for
the upstream metadata mismatches. Package metadata is preserved, not rewritten.

Activate: source .venv/bin/activate
Check: python tools/check_train_env.py
The check reports GPU-only checks as skipped on a CPU node. Run it again on a GPU node
to verify training/deployment imports and FlashAttention/FlexAttention kernels.
EOF
"$PYTHON_EXE" tools/check_train_env.py
echo "Installed V2 environment: $VENV_DIR"
