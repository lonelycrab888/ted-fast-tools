# TED-fast-tools

[English](README.md) · 简体中文

TED-fast-tools 使用 *The Encyclopedia of Domains*（TED）的 Merizo、Chainsaw 和 UniDoc 方法，从单链蛋白质 PDB 预测结构域。三个方法均可独立运行，也可以联用，计算 TED 共识边界，并将 high、medium 共识域分别导出为 PDB。

在**同两张 RTX 4090 D、同一批 2,349 个结构**的实测中，`ted-fast` 将端到端耗时从 **3 小时 29 分 48.46 秒降至 21 分 22.17 秒，加速 9.82×**。导出的 13,258 个域 PDB 与原版的 SHA-256 全部一致。[双卡实测记录](benchmarks/evidence/summary.json)

## 下载与安装

### 1. 下载代码，创建环境

```bash
git clone https://github.com/lonelycrab888/ted-fast-tools.git
cd ted-fast-tools
```

如果拿到的是源码 ZIP，解压后进入含 `run_fast.py` 的目录。后续命令均在仓库根目录运行。完整流程适用于 Linux x86_64，需要 Conda、NVIDIA 驱动、带 `nvcc` 的 CUDA Toolkit，以及支持 C++17 的编译器。按 GPU 选择下列一套环境。PyTorch wheel 提供运行时所需的 CUDA 库，系统 Toolkit 用于编译 Merizo 扩展。

| 实测 GPU | Python | PyTorch | 实测 nvcc | 实测驱动 |
| --- | --- | --- | --- | --- |
| RTX 4090 D | 3.10.22 | 2.0.1+cu118 | 12.6.85 | 565.77 |
| Blackwell（RTX 5090） | 3.11.13 | 2.7.1+cu128 | 12.8.93 | 580.178.04 |

**RTX 4090 D**

```bash
conda env create -f environment.yml
conda activate tes-env
python -m pip install -r env/requirements-online.txt
python -m pip check
```

**Blackwell**

```bash
conda env create -f environment-blackwell.yml
conda activate ted-fast-blackwell
python -m pip install -r env/requirements-blackwell.txt
python -m pip check
```

### 2. 下载权重、STRIDE 和 UniDoc 程序

```bash
python scripts/assets.py --download
```

这条命令将 **5 份模型权重和 3 个 Linux x86_64 可执行程序**安装到各自的模块目录，逐份校验 SHA-256，并给程序设置执行权限。再次运行会跳过校验通过的文件。UniDoc 的两个程序来自同一份归档，安装时只下载一次。[文件与校验值清单](assets/manifest.json)

| 下载内容 | 命令 |
| --- | --- |
| 全部 5 份权重，含 Merizo 的 3 个分片和 Chainsaw 的 2 个检查点 | `python scripts/assets.py --download-weights` |
| 两份 STRIDE 和 `UniDoc_structure` | `python scripts/assets.py --download-native` |
| 单个方法的全部运行文件 | `python scripts/assets.py --download --module chainsaw`，也可选 `merizo` 或 `unidoc` |

**手动下载。** 下表的目标路径均相对于 `modules/<方法名>/upstream/`，其中方法名用小写。请先创建所需子目录。Chainsaw 的两份权重都叫 `weights.pt`，要分别保存到表中指定的目录。Chainsaw 和 UniDoc 各用自己的 STRIDE。

| 方法 | `upstream/` 下的目标路径 | 官方下载 |
| --- | --- | --- |
| Merizo | `weights/weights_part_0.pt` | [weights_part_0.pt](https://raw.githubusercontent.com/psipred/Merizo/41d12fb84e6e8fdb586c2c859d12161dc7bb5bfd/weights/weights_part_0.pt) |
| Merizo | `weights/weights_part_1.pt` | [weights_part_1.pt](https://raw.githubusercontent.com/psipred/Merizo/41d12fb84e6e8fdb586c2c859d12161dc7bb5bfd/weights/weights_part_1.pt) |
| Merizo | `weights/weights_part_2.pt` | [weights_part_2.pt](https://raw.githubusercontent.com/psipred/Merizo/41d12fb84e6e8fdb586c2c859d12161dc7bb5bfd/weights/weights_part_2.pt) |
| Chainsaw | `saved_models/secondary_structure_epoch17/version_2/weights.pt` | [weights.pt](https://raw.githubusercontent.com/psipred/ted-tools/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/saved_models/secondary_structure_epoch17/version_2/weights.pt) |
| Chainsaw | `saved_models/ss_c_base_no_excl/version_2/epoch_11/weights.pt` | [weights.pt](https://raw.githubusercontent.com/psipred/ted-tools/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/saved_models/ss_c_base_no_excl/version_2/epoch_11/weights.pt) |
| Chainsaw | `stride/stride` | [TED 的 STRIDE](https://raw.githubusercontent.com/psipred/ted-tools/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/stride/stride) |
| UniDoc | `bin/stride` | [UniDoc_2023.tgz](https://yanglab.qd.sdu.edu.cn/UniDoc/download/UniDoc_2023.tgz) → `UniDoc/bin/stride` |
| UniDoc | `bin/UniDoc_structure` | [UniDoc_2023.tgz](https://yanglab.qd.sdu.edu.cn/UniDoc/download/UniDoc_2023.tgz) → `UniDoc/bin/UniDoc_structure` |

手动安装时，把五份权重和 Chainsaw 的 STRIDE 放到上表位置，再将 `UniDoc_2023.tgz` 下载到仓库根目录。执行以下命令，提取 UniDoc 的两个程序、设置权限，并校验全部八个文件。使用一键安装命令时，这些步骤会自动完成。

```bash
mkdir -p modules/unidoc/upstream/bin
tar -xzf UniDoc_2023.tgz -C modules/unidoc/upstream/bin --strip-components=2 \
  UniDoc/bin/stride UniDoc/bin/UniDoc_structure
chmod 755 modules/chainsaw/upstream/stride/stride \
  modules/unidoc/upstream/bin/stride modules/unidoc/upstream/bin/UniDoc_structure
python scripts/assets.py
```

### 3. 编译扩展

```bash
bash build.sh
```

这一步在目标机器上编译 Merizo 的 CUDA 核和 Chainsaw 的 C++ 域分配库。编译器不在 `PATH` 时，可用 `NVCC` 或 `CXX` 指定路径。只用一个方法时，可单独运行 `bash build.sh merizo` 或 `bash build.sh chainsaw`。UniDoc 在 CPU 上运行，可跳过编译。[安装与离线部署](docs/INSTALL.md) · [Blackwell 部署](docs/blackwell.md)

## 单独运行三个方法

独立入口接受单个 `.pdb` 文件，或一个顶层含 `*.pdb` 的目录。文件名去掉后缀即为结构 ID。每次运行使用**新的输出目录**。已验证的输入是链 A、残基从 1 连续编号的 AlphaFold DB 结构。示例中的 GPU 编号可替换为 `nvidia-smi` 显示的物理卡号。

**Merizo** 在 GPU 上预测结构域。`runs/merizo/chopping_merizo.txt` 是经过 TED 过滤的边界表，可直接交给 UniDoc。

```bash
python modules/merizo/run.py -i examples/pdb -o runs/merizo --gpu 0
```

**Chainsaw** 计算残基对特征，结合 STRIDE 给出的二级结构进行域分配。过滤后的预测结果在 `runs/chainsaw/chopping_chainsaw.txt`。

```bash
python modules/chainsaw/run.py -i examples/pdb -o runs/chainsaw --gpu 0
```

在 Blackwell 上独立运行 Chainsaw 时，以下命令采用与完整 TED 流程相同的精度设置。

```bash
env -u NVIDIA_TF32_OVERRIDE TED_CHAINSAW_DISABLE_TF32=1 \
  python modules/chainsaw/run.py -i examples/pdb -o runs/chainsaw-blackwell --gpu 0
```

**UniDoc** 在 CPU 上运行。如果只安装 UniDoc，可使用 Linux x86_64 的 Python 3.10 环境，安装两个 Python 依赖及该模块的原生程序。

```bash
python -m pip install -r modules/unidoc/requirements.txt
python scripts/assets.py --download --module unidoc
```

独立解析整条链时，使用 `--whole-chain`。

```bash
python modules/unidoc/run.py -i examples/pdb/AF-P0CG08-F1-model_v4.pdb \
  -o runs/unidoc-whole --whole-chain --workers 4
```

需要复现完整 TED 中的 UniDoc 输入时，先运行 Merizo，再通过 `--inherit` 传入其**过滤后的边界表**。表中须包含每个输入结构 ID。UniDoc 使用这些边界选出的残基，结果写入其输出目录的 `chopping_unidoc.txt`。[Merizo 说明](modules/merizo/README.md) · [Chainsaw 说明](modules/chainsaw/README.md) · [UniDoc 说明](modules/unidoc/README.md)

```bash
python modules/unidoc/run.py -i examples/pdb -o runs/unidoc-inherit \
  --inherit runs/merizo/chopping_merizo.txt --workers 4
```

## 联用三个方法运行完整 TED

输入为一个 **PDB 目录**。主程序会启动三个方法，把 Merizo 过滤后的边界传给 UniDoc，再用 TED 共识代码合并三份过滤后的预测表。以下命令用两张 GPU 处理仓库自带的示例，并导出 high、medium 共识域。

```bash
python run_fast.py -i examples/pdb -o runs/ted \
  --gpus 0 1 --overlap --unidoc-batch-size 32 --unidoc-workers 4 --export-domains
```

使用单卡时改为 `--gpus 0`。每张指定的卡至少需要一个输入结构。在 Blackwell 上使用下列启动器，只对 Chainsaw 关闭 TF32，Merizo 沿用已验证的默认精度。

```bash
./run-blackwell.sh -i examples/pdb -o runs/ted-blackwell \
  --gpus 0 1 --overlap --unidoc-batch-size 8 --unidoc-workers 4 --export-domains
```

| 参数 | 作用 |
| --- | --- |
| `--variant ted-fast` | 选择默认的优化流程。 |
| `--overlap` | 允许 Merizo 和 Chainsaw 在同一张卡上并行。显存紧张时可省略。 |
| `--unidoc-batch-size N` | Merizo 每完成一批结构，就将其交给 CPU 上的 UniDoc。默认 `0` 表示等待整个分片。双卡 4090 实测使用 `32`。 |
| `--unidoc-workers N` | 每张 GPU 对应的 UniDoc CPU 工作线程数，默认 `4`。 |
| `--structures-per-gpu N` | 每张 GPU 的独立结构队列数，默认 `1`。增加队列会额外加载模型实例。 |
| `--export-domains` | 为每个 high、medium 共识域写出一个 PDB，并读回校验。 |

### 查看输出

| 运行目录中的文件 | 内容 |
| --- | --- |
| `chopping_merizo.txt`、`chopping_chainsaw.txt`、`chopping_unidoc.txt` | 每个方法一份无表头 TSV。六列依次为结构 ID、序列 MD5、长度、域数、边界和分数。 |
| `consensus.tsv` | 每个结构一行。前三列是 ID、序列 MD5 和长度，随后是 high、medium、low 的域数以及对应的三组边界。`na` 表示该级别无域。 |
| `domains/` 和 `domains.tsv` | 开启导出后生成的 PDB 和清单。清单含父结构 ID、共识等级、边界、残基数、原子数、平均 Cα pLDDT、SHA-256 和文件路径。 |
| `no_domain.tsv` | 开启导出后，列出没有 high 或 medium 域的结构。 |
| `timing.json` 和 `gpu*/` | 记录运行配置、各阶段耗时、逐卡预测和日志。开启导出时，`end_to_end_seconds` 包含导出与校验。 |

边界采用闭区间。逗号分隔不同域，下划线连接同一域的不连续片段。导出保留父结构的残基编号和 ATOM 记录，要求每个残基有一个 Cα，插入码和备选构象字段为空。TED 默认保留长度至少为 5 的片段和总长度至少为 25 的域。随仓库提供的三个示例预计导出 **4 个 high、6 个 medium** 域。如果已有共识表，也可以单独执行导出。[示例与参考输出](examples/README.md)

```bash
python export_domains.py -i examples/pdb --consensus examples/expected/consensus.tsv -o runs/export-only
```

## 相比原版 TED 做了哪些优化？

| 环节 | `ted-fast` 的改动 |
| --- | --- |
| Merizo | 同一进程处理多个结构，复用已加载的模型。使用原生 CUDA 核加速不变点注意力（IPA）的点运算，对反复出现且满足条件的 GRU 输入形状复用 CUDA Graph。 |
| Chainsaw | 在 GPU 推理期间预取 STRIDE 特征，直接从 Cα 坐标计算残基对距离，用编译后的 C++ 实现贪心域分配。 |
| UniDoc | 在同一 Python 进程内准备选定的 PDB 记录和序列信息，由 CPU 线程池把结构交给上游的 STRIDE 和 `UniDoc_structure` 执行。 |
| 流程调度 | 按 PDB 文件大小在 GPU 之间分配结构。每张卡上可重叠运行两个 GPU 模型。Merizo 每完成一批结构，CPU 即可开始该批的 UniDoc 计算。 |

### 2,349 个结构的双卡实测

两次历史运行使用**相同的两张 RTX 4090 D**，输入结构及逐卡顺序相同，两张卡分别处理 1,174 和 1,175 个结构。输入是 `plddt70_v1` 中未匹配到 TED v4 官方边界的全部 2,349 个结构。两组都从启动计时至域 PDB 导出并校验完毕。原版耗时包含当次生产运行中的 UniDoc 兼容性重试。

| 流程 | 端到端耗时 | 加速比 | 节省时间 |
| --- | ---: | ---: | ---: |
| 原版 TED 流程 | 3 小时 29 分 48.46 秒 | 1.00× | — |
| TED-fast-tools `ted-fast` | **21 分 22.17 秒** | **9.82×** | **3 小时 8 分 26.29 秒（89.81%）** |

优化版从启动到生成共识表用时 1,259.46 秒，导出和校验用时 22.71 秒。三个方法表及共识表各有 **2,349 行**，逐结构核对全部一致。**13,258 个域 PDB** 的 SHA-256 全部匹配，包括 5,623 个 high 域和 7,635 个 medium 域。1,920 个结构有保留域，429 个结构没有保留域。[计时记录](benchmarks/evidence/summary.json) · [逐表核对](benchmarks/evidence/table-comparison.json) · [PDB 校验](benchmarks/evidence/native-export-check.json) · [复现逐卡分配](benchmarks/README.md)

另一次 Blackwell 兼容性验证使用 48 个结构和另选的 24 个结构，两组互不重叠。与 4090 参考结果相比，**72/72** 个结构的 high、medium 边界和 **233/233** 个导出 PDB 一致。完整共识表有 70/72 行逐字段一致，两处差异均在 low 共识域。[Blackwell 验证记录](benchmarks/evidence/blackwell-compatibility.json)

### 验证安装

按平台选择回归命令。测试会分别运行三个独立方法、整链 UniDoc 和完整 TED，并和仓库自带的参考输出比较。

```bash
python scripts/smoke_test.py --platform 4090 --gpus 0 1 --output runs/check-4090
```

```bash
python scripts/smoke_test.py --platform blackwell --gpus 0 1 --output runs/check-blackwell
```

## 引用和上游代码

如果在研究中使用 TED 的结构域预测结果，请引用原始论文。

Lau AM, Bordin N, Kandathil SM, Sillitoe I, Waman VP, Wells J, Orengo CA and Jones DT. *Exploring structural diversity across the protein universe with The Encyclopedia of Domains*. Science **386**, eadq4946 (2024). [DOI 10.1126/science.adq4946](https://doi.org/10.1126/science.adq4946)

原始实现见 [TED 官方 GitHub 仓库](https://github.com/psipred/ted-tools)。本项目所用共识代码、权重和程序的来源见[来源与第三方说明](docs/SOURCES.md)。
