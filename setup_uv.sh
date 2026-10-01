#!/usr/bin/env bash
# Run with bash; all managed tools and environments stay beside this file.
set -Eeuo pipefail
trap 'printf "ERROR: setup failed at line %s. Fix the error above and rerun.\n" "$LINENO" >&2' ERR

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PYTHON_VERSION=3.10.12
UV_VERSION=0.10.9
PYTHON_ONLY=0
FLASH_ATTN=1
for arg in "$@"; do
  case "$arg" in
    --python-only) PYTHON_ONLY=1 ;;
    --skip-flash-attn) FLASH_ATTN=0 ;;
    -h|--help)
      cat <<'HELP'
Usage: bash setup_uv.sh [--python-only] [--skip-flash-attn]

Installs uv 0.10.9 into .tools/uv/, CPython 3.10.12 into .python/,
and the training environment into .venv/ beside this script.
Target: Linux x86_64 (glibc), NVIDIA GPU, CUDA 12.8 PyTorch wheels.
Requires curl, tar, git and FFmpeg shared libraries for full setup.
Does not install drivers, modify shell profiles, or replace an existing
environment with a different Python version/location.

--python-only       Install only uv, Python and the virtual environment.
--skip-flash-attn   Omit FlashAttention; use eager or sdpa attention.

After setup: source .venv/activate-local.sh
HELP
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'This profile supports Linux x86_64 only.'
for tool in curl tar; do
  command -v "$tool" >/dev/null || die "Install the system tool '$tool' first."
done

export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_DOWNLOADS=manual
UV="$ROOT/.tools/uv/uv"
ENV_PYTHON="$ROOT/.venv/bin/python"

# Check before any installation; never silently reuse the current 3.12 venv.
if [[ -e "$ROOT/.venv" || -L "$ROOT/.venv" ]]; then
  [[ -x "$ENV_PYTHON" ]] || die '.venv exists but has no working Python; move it aside before rerunning.'
  "$ENV_PYTHON" - "$PYTHON_VERSION" "$UV_PYTHON_INSTALL_DIR" <<'PY' || exit 1
import pathlib, sys
expected = sys.argv[1]
base = pathlib.Path(sys.base_prefix).resolve()
local_root = pathlib.Path(sys.argv[2]).resolve()
if sys.version.split()[0] != expected or local_root not in base.parents:
    sys.exit(f"Existing .venv uses Python {sys.version.split()[0]} at {base}. "
             f"Expected {expected} under {local_root}. "
             "The existing environment was not changed. Move it aside explicitly first.")
PY
fi

if (( ! PYTHON_ONLY )); then
  command -v git >/dev/null || die 'Install git first.'
  command -v ffmpeg >/dev/null || die 'Install FFmpeg and its shared libraries first (Ubuntu: sudo apt-get install ffmpeg).'
fi

if [[ ! -x "$UV" ]] || [[ $("$UV" --version) != "uv $UV_VERSION"* ]]; then
  mkdir -p "$ROOT/.tools/uv"
  installer=$(mktemp)
  trap 'rm -f -- "$installer"' EXIT
  curl --fail --location --show-error --retry 3 \
    "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$installer"
  UV_UNMANAGED_INSTALL="$ROOT/.tools/uv" sh "$installer"
  rm -f -- "$installer"
  trap - EXIT
fi

"$UV" python install "$PYTHON_VERSION" --no-bin
BASE_PYTHON="$UV_PYTHON_INSTALL_DIR/cpython-$PYTHON_VERSION-linux-x86_64-gnu/bin/python3.10"
[[ -x "$BASE_PYTHON" ]] || die "Expected managed Python missing: $BASE_PYTHON"
if [[ ! -e "$ROOT/.venv" ]]; then
  "$UV" venv --python "$BASE_PYTHON" "$ROOT/.venv"
fi
"$ENV_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3, 10, 12); print(sys.version)'

# A child shell cannot activate its parent shell. Supply an explicit source file.
{
  printf 'export UV_PYTHON_INSTALL_DIR=%q\n' "$UV_PYTHON_INSTALL_DIR"
  printf 'export UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR"
  printf 'export PATH=%q:"$PATH"\n' "$ROOT/.tools/uv"
  printf 'source %q\n' "$ROOT/.venv/bin/activate"
} > "$ROOT/.venv/activate-local.sh"

if (( PYTHON_ONLY )); then
  printf '\nPython environment ready. Run: source %q\n' "$ROOT/.venv/activate-local.sh"
  exit 0
fi

cd "$ROOT"
# Preserve the committed submodule revisions, not the upstream branch tips.
git submodule update --init --recursive
"$UV" pip install --python "$ENV_PYTHON" \
  -r "$ROOT/requirements-server.lock"

# Keep the tested VLA versions instead of resolving conflicting upstream pins.
# This is a training profile, not LeRobot's robot-control/GUI/all-extras profile.
"$UV" pip install --python "$ENV_PYTHON" --no-deps \
  'https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz' \
  -e "$ROOT" \
  -e "$ROOT/lingbotvla/models/vla/vision_models/lingbot-depth" \
  -e "$ROOT/lingbotvla/models/vla/vision_models/MoGe"

if (( FLASH_ATTN )); then
  # Match Python 3.10, Torch 2.8, CUDA 12 and the Torch C++11 ABI exactly.
  ABI=$("$ENV_PYTHON" -c 'import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())')
  FLASH_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"
  "$UV" pip install --python "$ENV_PYTHON" --no-deps "$FLASH_WHEEL"
fi

"$ENV_PYTHON" - "$FLASH_ATTN" <<'PY'
import importlib, runpy, sys
import torch
for name in ('torchvision', 'torchaudio', 'torchcodec', 'lerobot',
             'lingbotvla.data.vla_data.base_dataset',
             'lingbotvla.models.vla.pi0.modeling_lingbot_vla',
             'moge.model.v2', 'mdm.model.v2'):
    importlib.import_module(name)
    print(f'Import OK: {name}')
if sys.argv[1] == '1':
    importlib.import_module('flash_attn')
    print('Import OK: flash_attn')
# Import the actual training entry without invoking main() or allocating a model.
runpy.run_path('tasks/vla/train_lingbotvla.py', run_name='setup_import_check')
print(f'Torch {torch.__version__}, CUDA runtime {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('GPU unavailable to this process; verify the driver/GPU allocation before training.')
PY
"$UV" pip freeze --python "$ENV_PYTHON" > "$ROOT/.venv/installed-requirements.txt"
printf '\nTraining dependencies installed. Run: source %q\n' "$ROOT/.venv/activate-local.sh"
printf 'Before training, configure model/data/norm_stats paths for this server.\n'
