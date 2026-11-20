# 安装与迁移

请按目标 GPU 选择环境。RTX 5090 使用独立的 Conda 文件、cu128 依赖清单和仓库根目录的 `run-blackwell.sh`。模型资产清单在两种平台上通用。[5090 完整安装与回归](blackwell.md)

## RTX 4090 D 环境

已验证平台为 Linux x86_64、RTX 4090 D（24 GB）、NVIDIA 驱动 565.77、Python 3.10.22 和 PyTorch 2.0.1+cu118。该版本的 PyTorch 自带 CUDA 11.8 运行库。系统的 CUDA Toolkit 12.6.85 用于编译 Merizo 的原生核，Chainsaw 的 C++ 库使用 GCC 12.2。原生输入程序 STRIDE 和 UniDoc 使用清单指定的 Linux 二进制。版本记录见 [环境快照](../benchmarks/evidence/environment.txt)。

## 从 Git 副本配置环境

在仓库根目录运行。已有同名环境时，可以在 `conda env create` 后加 `-n ted-fast-test`，并激活新名字。创建步骤分为 Conda 解释器和锁定的 Python 依赖两步。

```bash
conda env create -f environment.yml
conda activate tes-env
python -m pip install -r env/requirements-online.txt
python -m pip check
```

如需复现 Linux x86_64 测试时的 Conda 包构建，可用 `conda create -n tes-env --file env/conda-linux-64.lock` 替代上面的 Conda 创建步骤，随后仍需激活环境并安装同一份 pip 依赖。

在线配置从 PyPI 和 PyTorch 的 CUDA 11.8 wheel 目录取包。离线迁移时，可在联网的同平台 CPython 3.10 环境执行 `python -m pip download -r env/requirements-online.txt -d wheelhouse`。把 wheelhouse 复制到目标机后，在新建的 Conda 环境改用以下安装命令。

```bash
python -m pip install --no-index --find-links=/path/to/wheelhouse -r env/requirements-linux.txt
```

## 恢复模型权重与原生程序

新克隆在仓库根目录运行以下命令，可从官方来源装齐五份权重和三个 Linux x86_64 原生程序。脚本仅使用 Python 标准库，已有文件校验通过就会跳过。新文件在 SHA-256 匹配后写入目标位置，程序会获得执行权限。可按需使用 `--download-weights` 仅下载权重，或 `--download-native` 仅下载程序。[权重来源](SOURCES.md#模型权重原始下载) · [原生程序来源](SOURCES.md#原生程序下载)

```bash
python scripts/assets.py --download
```

UniDoc 2023 的官方归档在一次安装中只下载一次，脚本校验归档后读取两个指定程序。离线部署可用本地资产包恢复全部八个文件。4090 上可直接使用以下路径。

```bash
python scripts/assets.py --bundle /home/zhr/CVPR2027/BiooSeek-motif/data/ted_fast_tools/assets/ted-fast-linux-x86.64.tar
python scripts/assets.py
```

也可以从另一份已安装 TED 的 `ted_consensus_1.0` 目录导入。导入和恢复命令都按仓库内的清单逐文件校验。三份 Merizo 权重、两份 Chainsaw 权重、两个不同的 STRIDE 和一个 UniDoc 程序各有独立的 SHA-256。[资产清单](../assets/manifest.json)

```bash
python scripts/assets.py --from-ted /home/zhr/tools/ted-tools/ted_consensus_1.0
```

只部署一个模块时，以上命令可以加 `--module merizo`、`--module chainsaw` 或 `--module unidoc`。已有完整资产的开发副本可用 `python scripts/assets.py --pack /path/to/new-assets.tar` 创建一个新包。4090 上现有的离线 tar 为约 211 MiB，SHA-256 为 `9078a74486148e2900d918cd061ccfbbf203e5d3f6542b523d133ea07083c27a`。

## 编译和验证

```bash
bash build.sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python scripts/smoke_test.py --platform 4090 --gpus 0
```

将卡号换成可用的物理 GPU。编译器不在默认路径时设置 `CXX=/path/to/g++`、`NVCC=/path/to/nvcc`。脚本通过 `PATH` 或 `/usr/local/cuda/bin/nvcc` 查找 nvcc，生成的两个共享库位于各自模块目录。可以只运行 `bash build.sh chainsaw` 或 `bash build.sh merizo`。单独运行 UniDoc 仅需 CPU、两个 UniDoc 资产和模块 README 中的两个 Python 包。

## 常见问题

- 提示找不到 `weights_part_*.pt`、`stride` 或 `UniDoc_structure` 时，运行 `python scripts/assets.py --download` 补齐或修复文件。
- 提示 `chainsaw_assign.so` 缺失或加载失败时，在该机器重新执行 `bash build.sh chainsaw`。Merizo GPU 核需要匹配的 NVIDIA 驱动和可加载的 CUDA 运行库，可用 `ldd modules/merizo/merizo_points.so` 检查。
- 显存不足时，先降至每卡一个结构队列。若两个模型同卡的峰值仍超出可用显存，可省略 `--overlap`。这是单独的运行配置，性能数字以各自设置为准。
- 某条结构没有域时，先查 `consensus.tsv` 的 high、medium 计数。若 UniDoc 原始表标记为 `NO_SS`，再查对应 `unidoc/run.log` 中的 `UNIDOC_ERROR`。
