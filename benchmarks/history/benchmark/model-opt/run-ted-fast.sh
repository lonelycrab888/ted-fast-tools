#!/usr/bin/env bash
set -e
repo=$(cd -- "$(dirname -- "$0")/../../../.." && pwd)
cd /home/zhr/CVPR2027/BiooSeek-motif/data/TED_fast
source /home/zhr/miniforge3/etc/profile.d/conda.sh
conda activate tes-env
for rep in 1 2
do
    d=benchmark/model-opt
    out="$d/ted-fast-$rep"
    python "$repo/run_fast.py" -i benchmark/holdout -o "$out" --gpus 4 --variant ted-fast --overlap \
       --structures-per-gpu 1 --unidoc-workers 4 --unidoc-batch-size 32 --unidoc-backend fast > "$out.log" 2>&1
    python "$d/validate.py" benchmark/holdout-reference "$d/ted-v2-1" "$out" "$d/holdout-gru-$rep.json"
done
