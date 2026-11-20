#!/usr/bin/env bash
set -e
repo=$(cd -- "$(dirname -- "$0")/../../../.." && pwd)
cd /home/zhr/CVPR2027/BiooSeek-motif/data/TED_fast
source /home/zhr/miniforge3/etc/profile.d/conda.sh
conda activate tes-env
set -u -o pipefail
d=benchmark/model-opt
python "$repo/run_fast.py" -i benchmark/full-input -o "$d/full-ted-fast" --gpus 5 6 7 --variant ted-fast \
    --structures-per-gpu 1 --overlap --unidoc-backend fast --unidoc-batch-size 32 --unidoc-workers 4 > "$d/full-ted-fast.log" 2>&1
python "$d/validate.py" benchmark/full-reference benchmark/full-stream "$d/full-ted-fast" "$d/full-validation.json"
