# 源码与运行资产来源

本仓库整理自 4090 上通过全量比对的 `TED_fast` ted-fast 工作副本，以及当时实际调用的 `ted_consensus_1.0`。TED 发布归档的 Git 提交为 `6041723c28f1782c2cbb1a4edb9b3523c0a1739c`，安装时的 ZIP SHA-256 为 `3b1c338a7314a8dfe0cfbecad3bbb8a6f1a55e081508808b2df235c8ec012543`。TED 的域过滤与三方法共识源码位于 `consensus/`。[TED 论文](https://www.science.org/doi/10.1126/science.adq4946) · [TED 随附许可](TED-LICENSE)

| 组件 | 本仓库的位置与来源 |
| --- | --- |
| Merizo | TED 随附的模型代码与 AFDB 预测入口，三份权重从 Merizo 上游获取。原始说明和许可保留在 [upstream](../modules/merizo/upstream/README.md) 和 [LICENSE](../modules/merizo/upstream/LICENSE) |
| Chainsaw | TED 随附的网络、配置与 STRIDE 源码，两份检查点从 TED 上游获取。模型原始说明见 [upstream](../modules/chainsaw/upstream/README.md)，STRIDE 的来源见 [README_stride](../modules/chainsaw/upstream/stride/README_stride) |
| UniDoc | 服务器安装的 UniDoc 结构模式程序、STRIDE 和配套源码。安装归档 `UniDoc_2023.tgz` 的 SHA-256 为 `4cf2c4d5ada32a6218cc655bea1751e364cd6d7ecd747d8f26e8e0d73cc30805`。原始命令见 [upstream](../modules/unidoc/upstream/README.md) |
| 域 PDB 导出 | `export_domains.py` 沿用 HumanMotifbench 的父 PDB 读取和逐 ATOM 读回校验函数。历史导出器快照见 [split_ted_v4.py](../benchmarks/history/provenance/split_ted_v4.py) |
| 回归样本 | HumanMotifbench 的 AFDB 人类结构，保留原始 AFDB 模型 ID。数据出处、参考预测和样本组成见 [examples](../examples/README.md) |

推理入口改为相对仓库定位。Chainsaw 的辅助进程使用当前 Python 解释器。UniDoc 参考包装器保留历史短记录补齐和空路径修正，并使用参数数组调用原生程序。优化版的权重、边界规则和核函数来自双卡验证的版本。第三方代码、权重和二进制分别沿用原作者的署名和条款。

## 模型权重原始下载

下列五个链接均指向上游固定的 Git 提交。Merizo 的三个分片共同组成一个模型。Chainsaw 的两份检查点同名，保存时请保留各自的子目录。运行 `python scripts/assets.py --download-weights` 可自动放置文件，并逐份核验 SHA-256。[安装路径和校验值](../assets/manifest.json) · [下载地址清单](../assets/weight-urls.json)

| 模块 | 模块的 `upstream/` 下的安装路径 | 原始文件 |
| --- | --- | --- |
| Merizo | `weights/weights_part_0.pt` | [下载 weights_part_0.pt](https://raw.githubusercontent.com/psipred/Merizo/41d12fb84e6e8fdb586c2c859d12161dc7bb5bfd/weights/weights_part_0.pt) |
| Merizo | `weights/weights_part_1.pt` | [下载 weights_part_1.pt](https://raw.githubusercontent.com/psipred/Merizo/41d12fb84e6e8fdb586c2c859d12161dc7bb5bfd/weights/weights_part_1.pt) |
| Merizo | `weights/weights_part_2.pt` | [下载 weights_part_2.pt](https://raw.githubusercontent.com/psipred/Merizo/41d12fb84e6e8fdb586c2c859d12161dc7bb5bfd/weights/weights_part_2.pt) |
| Chainsaw | `saved_models/secondary_structure_epoch17/version_2/weights.pt` | [下载 weights.pt](https://raw.githubusercontent.com/psipred/ted-tools/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/saved_models/secondary_structure_epoch17/version_2/weights.pt) |
| Chainsaw | `saved_models/ss_c_base_no_excl/version_2/epoch_11/weights.pt` | [下载 weights.pt](https://raw.githubusercontent.com/psipred/ted-tools/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/saved_models/ss_c_base_no_excl/version_2/epoch_11/weights.pt) |

TED 官方安装脚本指定从 Merizo 获取这三个分片。Chainsaw 使用本项目复现的 TED 发布版本中的检查点。[TED 安装脚本](https://github.com/psipred/ted-tools/blob/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/setup.sh)

## 原生程序下载

`python scripts/assets.py --download-native` 安装下列三个 Linux x86_64 可执行文件。`--download` 同时安装模型权重。两个模块各自使用清单指定的 STRIDE。[文件路径和 SHA-256](../assets/manifest.json)

| 模块 | 原生程序 | 官方来源 |
| --- | --- | --- |
| Chainsaw | `stride/stride` | [TED 固定版本的 STRIDE](https://raw.githubusercontent.com/psipred/ted-tools/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/stride/stride) |
| UniDoc | 归档中的 `UniDoc/bin/stride` 和 `UniDoc/bin/UniDoc_structure` | [UniDoc 2023 发布包](https://yanglab.qd.sdu.edu.cn/UniDoc/download/UniDoc_2023.tgz) |
