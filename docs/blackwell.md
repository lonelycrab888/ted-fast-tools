# Blackwell 部署与验证

RTX 5090 的计算能力为 12.0（sm_120）。已验证环境为 Linux x86_64、Python 3.11.13、PyTorch 2.7.1+cu128、nvcc 12.8.93 和驱动 580.178.04。在这张卡上，4090 版的 PyTorch 2.0.1+cu118 可以枚举设备，但启动 Merizo 时会报 `no kernel image is available for execution on the device`。[PyTorch Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)

## 安装

在仓库根目录执行。环境名是 `ted-fast-blackwell`。Conda 文件指定 Python 版本，pip 清单固定 Linux x86_64 的 Python 依赖，包括 PyTorch 和它所需的 CUDA 运行库。示例中的 GPU 编号按部署位置替换。`NVCC` 可指定 CUDA Toolkit 12.8 的 nvcc 路径。下方的 `--download` 会获取完整 TED 所需的八个运行文件。[资产来源](SOURCES.md)

```bash
conda env create -f environment-blackwell.yml
conda activate ted-fast-blackwell
python -m pip install -r env/requirements-blackwell.txt
python -m pip check

python scripts/assets.py --download
bash build.sh
python scripts/smoke_test.py --platform blackwell --gpus 0 1 --output runs/smoke-blackwell
```

已有本地模型和程序时，`python scripts/assets.py` 可校验八个资产。离线部署可在联网的相同平台、相同 Python 版本中先运行 `python -m pip download -r env/requirements-blackwell.txt -d wheelhouse`。在新环境内使用 `python -m pip install --no-index --find-links=/path/to/wheelhouse -r env/requirements-blackwell-linux.txt` 安装。模型资产与 Python wheel 分别迁移。[资产恢复说明](INSTALL.md)

## 完整流程

```bash
./run-blackwell.sh -i examples/pdb -o runs/demo-blackwell \
  --gpus 0 1 --overlap --unidoc-batch-size 8 --unidoc-workers 4 --export-domains
```

启动脚本使用当前激活环境的 Python。输入和输出路径相对于调用者当前目录。脚本将 `TED_CHAINSAW_DISABLE_TF32=1` 传给工作进程，并清除继承的全局 `NVIDIA_TF32_OVERRIDE`。只有 Chainsaw 的独立预测进程在导入 torch 前设置 TF32 覆盖，Merizo 保留默认精度。`timing.json` 的 `chainsaw_disable_tf32` 记录该选择。在先前的 48 条对照中，全局关闭 TF32 改变了 3 个结构的 high/medium 输出。

单独运行 Chainsaw 时用下列命令获得相同的精度设置。Merizo 和 UniDoc 沿用各自的独立入口。

```bash
env -u NVIDIA_TF32_OVERRIDE TED_CHAINSAW_DISABLE_TF32=1 \
  python modules/chainsaw/run.py -i examples/pdb -o runs/chainsaw-blackwell --gpu 0
```

## 加速路径与正确性

编译器支持 sm_120 时，`build.sh` 生成对应机器码。Merizo 在已验证的 PyTorch 2.7.1+cu128、sm_120 和受支持的输入形状上启用 IPA 原生核和 GRU 图复用。17 种长度（1–1,400）的两个 IPA 运算在设备上逐元素通过比较，长度 1,400 的 GRU 用 6 个不同输入核对通过。此前的 48 结构运行记录了 84 次图回放。

5090 与 4090 的参考比较分为 48 条配置验证和 24 条独立复核。共计 72/72 条 high、medium 边界和 233/233 个导出 PDB 完全一致。完整共识表为 70/72 行一致。两条 low 边界差异对应 `AF-P00395-F1-model_v4` 和 `AF-P04114-F7-model_v4`。部分 Merizo、Chainsaw 分数也有差异。[跨卡验证记录](../benchmarks/evidence/blackwell-compatibility.json)

`--platform blackwell` 的示例回归覆盖三个独立方法、整链 UniDoc 和仓库启动脚本。使用 `--output` 保留完整运行目录和 `cross-gpu.json`。若要比较其他已完成的导出，参考目录需要提供三个方法表、共识表及 `domains.sha256`。工具会记录所有逐行差异，同时检查结构身份、保留域边界、域文件集合和每个 PDB 的 SHA-256。

```bash
python scripts/compare_cross_gpu.py examples/expected runs/demo-blackwell runs/comparison.json
```
