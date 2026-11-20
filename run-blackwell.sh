#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")" && pwd)
# Keep Merizo at its validated precision. run_method.py applies the override
# only in the separate Chainsaw process, before importing torch.
exec env -u NVIDIA_TF32_OVERRIDE TED_CHAINSAW_DISABLE_TF32=1 \
    python "$repo/run_fast.py" "$@"
