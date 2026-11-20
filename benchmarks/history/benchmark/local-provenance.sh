#!/usr/bin/env bash
set -euo pipefail
root=/home/zhr/tools/ted-tools/ted_consensus_1.0
envpy=/home/zhr/miniforge3/envs/tes-env/bin/python
base="$(cd "$(dirname "$0")/.." && pwd)"
cd "$base"
{
  hostname
  date -Is
  "$envpy" --version
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  g++ --version | sed -n '1p'
  "$envpy" -c 'import numpy,scipy,torch; print("numpy",numpy.__version__,"scipy",scipy.__version__,"torch",torch.__version__,"CUDA",torch.version.cuda); print("defaults torch TF32",torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)'
} > benchmark/environment.txt
sha256sum chainsaw_assign.cpp ted_kernels.py run_method.py run_fast.py run_unidoc.py \
  "$root"/programs/chainsaw/saved_models/secondary_structure_epoch17/version_2/{config.json,weights.pt} \
  "$root"/programs/chainsaw/src/domain_assignment/sparse_lowrank.py \
  "$root"/programs/chainsaw/stride/stride \
  "$root"/programs/merizo/weights/*.pt \
  "$root"/programs/unidoc/bin/{UniDoc_structure,stride} \
  "$root"/programs/unidoc/Run_UniDoc_from_scratch_structure_afdb.py \
  "$root"/scripts/{get_consensus.py,domain_consensus.py,filter_domains.py,filter_domains_consensus.py} \
  > benchmark/provenance.sha256
