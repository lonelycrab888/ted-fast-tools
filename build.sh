#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
target=${1:-all}
case "$target" in all|chainsaw|merizo) ;; *) echo "Usage: bash build.sh [all|chainsaw|merizo]" >&2; exit 2;; esac
build_dir=$(mktemp -d ./.ted-build.XXXXXXXX)
trap 'rm -rf -- "$build_dir"' EXIT
if [[ "$target" != merizo ]]; then
    "${CXX:-g++}" -std=c++17 -O3 -fPIC -shared modules/chainsaw/chainsaw_assign.cpp -o "$build_dir/chainsaw_assign.so"
    mv "$build_dir/chainsaw_assign.so" modules/chainsaw/
fi
if [[ "$target" != chainsaw ]]; then
    nvcc_path=${NVCC:-$(command -v nvcc || true)}
    nvcc_path=${nvcc_path:-/usr/local/cuda/bin/nvcc}
    if ! "$nvcc_path" --list-gpu-arch | grep -qx compute_80; then
        echo 'Set NVCC to a CUDA compiler supporting compute_80 (tested with CUDA Toolkit 12.6).' >&2
        exit 1
    fi
    arch=(-gencode=arch=compute_80,code=compute_80)
    if "$nvcc_path" --list-gpu-code | grep -qx sm_89; then arch+=(-gencode=arch=compute_89,code=sm_89); fi
    if "$nvcc_path" --list-gpu-code | grep -qx sm_120; then arch+=(-gencode=arch=compute_120,code=sm_120); fi
    "$nvcc_path" -O3 --fmad=false -Xcompiler=-fPIC -shared "${arch[@]}" modules/merizo/merizo_points.cu -o "$build_dir/merizo_points.so"
    mv "$build_dir/merizo_points.so" modules/merizo/
fi
