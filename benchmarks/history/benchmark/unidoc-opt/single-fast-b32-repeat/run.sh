#!/usr/bin/env bash
set -e
source /home/zhr/miniforge3/etc/profile.d/conda.sh
conda activate tes-env
set -u -o pipefail
cd /home/zhr/CVPR2027/BiooSeek-motif/data/TED_fast
out=benchmark/unidoc-opt/single-fast-b32-repeat
count=$(find -L benchmark/holdout -maxdepth 1 -type f -name '*.pdb' | wc -l)
test "$count" -eq 128
nvidia-smi -i 4 --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv > "$out/gpu-before.csv"
for rep in 1 2
do
    date -Is
    echo "Starting pass $rep"
    python run_fast.py -i benchmark/holdout -o "$out/run-$rep" \
        --variant v2 --gpus 4 --structures-per-gpu 1 --overlap \
        --unidoc-workers 4 --unidoc-batch-size 32 --unidoc-backend fast > "$out/run-$rep.log" 2>&1
    python -c 'import json, sys; data=json.load(open(sys.argv[1])); print("wall_seconds", data["wall_seconds"], "inputs", data["inputs"], flush=True)' "$out/run-$rep/timing.json"
done
python benchmark/compare.py benchmark/holdout-reference "$out/run-1" "$out/run-2" -o "$out/vs-reference.json"
python benchmark/compare.py "$out/run-1" "$out/run-2" -o "$out/between-runs.json"
date -Is
