# Merizo

输入单链 A 的 AFDB PDB，输出 Merizo 原始预测和经过 TED 过滤的边界。该入口单独加载 Merizo。默认 `ted-fast` 使用 FP32，在支持的输入上启用优化的 IPA CUDA 核，并在符合条件的重复 GRU 形状上复用 CUDA Graph。

## 配置

激活项目的 Conda 环境。从仓库根目录检查模型权重并编译内核。首次克隆时可从 Merizo 上游下载本模块的三个权重分片。[权重直链](../../docs/SOURCES.md#模型权重原始下载)[环境配置](../../docs/INSTALL.md)

```bash
python scripts/assets.py --download-weights --module merizo
bash build.sh merizo
```

## 独立运行

以下命令从本目录执行。将 `--gpu 6` 换成空闲物理卡号。省略此项时沿用当前 CUDA 可见设备。

```bash
python run.py -i ../../examples/pdb -o ../../runs/merizo --gpu 6
```

`-i` 也可指向单个 PDB，`-o` 是新目录。`merizo/raw.tsv` 保存原始预测，`chopping_merizo.txt` 是经过过滤的六列无表头 TSV，包含结构 ID、序列 MD5、长度、域数、边界和分数。边界表可直接交给 UniDoc 的 `--inherit`。逐结构错误及符合条件的 GRU 图捕获与回放计数在 `merizo/run.log`。

`merizo_ipa_fast.py` 与 `merizo_points.cu` 对应 IPA 的优化实现，`merizo_gru_graphs.py` 管理 CUDA Graph，`merizo_gru.py` 提供剖析和实验性的置信度 GRU 合批。整套 TED 与此入口共用同一个模型加载器。[上游说明](upstream/README.md)
