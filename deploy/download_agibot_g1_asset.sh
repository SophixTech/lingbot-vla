#!/usr/bin/env bash
# Download only the public official AgiBot G1 Omnipicker asset subset.
# Hugging Face keeps a resumable cache, so rerunning this command is safe.
set -euo pipefail

readonly hf="/home/bjtc/Sophix/lingbot-vla/.venv/bin/hf"
readonly destination="/home/bjtc/Sophix/third_party/geniesim_assets"

exec "$hf" download agibot-world/GenieSimAssets \
    --repo-type dataset \
    --include \
        'robot/G1_omnipicker/**' \
        'robot/curobo_robot/assets/robot/G1/G1_omnipicker.urdf' \
        'pyproject.toml' \
        'README.md' \
    --local-dir "$destination"
