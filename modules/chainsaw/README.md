# Chainsaw

从单链 A 的 PDB 计算 Cα 距离和 STRIDE 二级结构特征，预测残基对属于同一结构域的概率，再将结果分配到结构域。独立入口只运行 Chainsaw，并执行与完整 TED 相同的编号转换与边界过滤。

## 配置

激活项目 Conda 环境后，在仓库根目录执行以下命令。脚本从 TED 上游下载两份检查点和 Chainsaw 的 STRIDE。本模块的 C++ 扩展需要支持 C++17 的编译器。[环境配置](../../docs/INSTALL.md)

```bash
python scripts/assets.py --download --module chainsaw
bash build.sh chainsaw
```

## 独立运行

以下命令从本目录执行。输入可为单个 PDB 或目录，输出目录须为新路径。

```bash
python run.py -i ../../examples/pdb -o ../../runs/chainsaw --gpu 7
```

在 RTX 5090 上激活 cu128 环境后，用以下命令复现完整流程对 Chainsaw 的精度设置。[RTX 5090 部署说明](../../docs/blackwell.md)

```bash
env -u NVIDIA_TF32_OVERRIDE TED_CHAINSAW_DISABLE_TF32=1 \
  python run.py -i ../../examples/pdb -o ../../runs/chainsaw-blackwell --gpu 0
```

`chainsaw/raw.tsv` 保存模型原始输出，`chopping_chainsaw.txt` 保存 TED 的六列无表头结果。后者已应用 Chainsaw 所需的残基偏移和长度过滤，应使用这张表与完整流程对照。日志在 `chainsaw/run.log` 与 `chainsaw/filter.log`。

`chainsaw_stride.py` 管理 STRIDE 预取，`chainsaw_inference.py` 负责特征准备，`chainsaw_assign.cpp` 实现域分配。默认检查点位于 `upstream/saved_models/secondary_structure_epoch17/version_2/`。输入 PDB 的临时副本写入运行目录。[上游说明](upstream/README.md)
