#!/usr/bin/env bash
set -eo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
root=/home/zhr/tools/ted-tools/ted_consensus_1.0
source /home/zhr/miniforge3/etc/profile.d/conda.sh
conda activate tes-env
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)"
out="${2:-$base/reference}"
mkdir -p "$(dirname "$out")"
out="$(realpath -m "$out")"
mkdir "$out"
inputs="$out/inputs"
mkdir "$inputs"
cp -p "$base"/frozen/*.pdb "$inputs/"
ulimit -c 0
TIMEFORMAT=%R
cd "$root"
for method in merizo chainsaw unidoc; do
    options=()
    if [[ "$method" == unidoc ]]; then options=(-c "$out/chopping_merizo.txt"); fi
    { time bash scripts/segment.sh -i "$inputs" -o "$out" -m "$method" "${options[@]}" > "$out/$method.log" 2>&1; } 2> "$out/$method.time"
    echo "$method $(cat "$out/$method.time")"
done
{ time (
    cd "$out"
    python "$root/scripts/get_consensus.py" -c chopping_merizo.txt chopping_chainsaw.txt chopping_unidoc.txt -o raw.tsv
    python "$root/scripts/filter_domains_consensus.py" raw.tsv -o consensus.tsv
) > "$out/consensus.log" 2>&1; } 2> "$out/consensus.time"
echo "consensus $(cat "$out/consensus.time")"
