#!/usr/bin/env bash
set -e
cd /home/zhr/CVPR2027/BiooSeek-motif/data/TED_fast
source /home/zhr/miniforge3/etc/profile.d/conda.sh
conda activate tes-env
set -u -o pipefail
d=benchmark/model-opt
nvidia-smi -i 4 --query-gpu=index,uuid,name,utilization.gpu,memory.used --format=csv > "$d/gpu4-before.csv"
python "$d/measure_method.py" chainsaw benchmark/holdout "$d/chainsaw-stride-v4-1" --variant v4
for pair in v2-1 v4-1 v4-2 v2-2
do
    variant=$(echo "$pair" | cut -d- -f1)
    result="$d/ted-$pair"
    python run_fast.py -i benchmark/holdout -o "$result" --gpus 4 --variant "$variant" \
        --structures-per-gpu 1 --overlap --unidoc-backend fast --unidoc-batch-size 32 --unidoc-workers 4 > "$result.log" 2>&1
    python benchmark/compare.py benchmark/holdout-reference "$result" -o "$result-comparison.json"
done
python "$d/measure_method.py" chainsaw benchmark/holdout "$d/chainsaw-stride-v4-2" --variant v4
