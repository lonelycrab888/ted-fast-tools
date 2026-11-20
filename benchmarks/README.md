# 性能基线与复现

历史双卡对照的输入是 `plddt70_v1` 中未能匹配 TED v4 官方边界的全部 2,349 个结构。物理 GPU 6 分配 1,174 条，GPU 7 分配 1,175 条。原版和 ted-fast 使用相同的每卡结构清单及顺序。

| 历史实测 | 数值 |
| --- | ---: |
| TED 原版端到端，含原生产流程的 UniDoc 重试 | 12,588.46 秒 |
| ted-fast 到共识表 | 1,259.46 秒 |
| ted-fast 导出与校验 | 22.71 秒 |
| ted-fast 连续端到端计时 | 1,282.17 秒 |
| 相同 GPU 的端到端加速 | 9.82× |

四张输出表逐结构相同。1,920 个结构保留域，429 个无 high 或 medium 域。5,623 个 high 域和 7,635 个 medium 域的 PDB SHA-256 全部相同。另一次复核使用同一轮推理结果和原版导出代码，分段合计 1,283.55 秒。原始计时、逐表比较和原导出器核对见 [计时](evidence/summary.json)、[表对照](evidence/table-comparison.json)、[导出核对](evidence/native-export-check.json)。

## 按历史逐卡分配启动当前仓库

从仓库根目录运行。此清单只适用于对应的 2,349 个 PDB 和物理 GPU 6、7。行顺序决定各卡处理顺序。常规新数据集省略 `--shard-manifest`，程序会按文件大小分片。

```bash
python run_fast.py -i ../TED_fast/benchmark/full-input -o runs/aligned-dual \
  --gpus 6 7 --shard-manifest benchmarks/evidence/gpu-assignment.tsv \
  --variant ted-fast --structures-per-gpu 1 --overlap --unidoc-workers 4 \
  --unidoc-batch-size 32 --export-domains
```

## 此仓库的发布核对

三个独立方法及完整流程使用固定示例回归。另将历史全量共识表交给新仓库的导出器，从原始父 PDB 重新生成所有保留域，逐 ID 和历史文件的 SHA-256 核对。上表的性能来自整理前的全量实测。[全量导出复核](evidence/package-export.json)

`history/benchmark/` 收录开发时的计时、剖析和一致性检查代码，运行参数已统一为 `ted-fast`，`history/provenance/` 保存历史生产脚本。这些归档脚本依赖原始 benchmark 目录布局和归档数据。4090 上原工作目录为 `/home/zhr/CVPR2027/BiooSeek-motif/data/TED_fast`。归档汇总脚本 `summarize.py` 可通过 `TED_BENCHMARK_DIR` 指向已有的 `model-opt` 目录，通过 `TED_BENCHMARK_VARIANT` 指定原始计时文件中的版本名。日常运行与回归使用当前仓库根目录和 `scripts/smoke_test.py`。

新建 Conda 环境和干净 Git 克隆均已实测，包含从离线资产包恢复、重新编译、单独 CPU 环境运行 UniDoc 以及新克隆上的双卡回归。[打包验证记录](evidence/package-smoke.json)

## RTX 5090 跨架构验证

在 5090-1 上用 48 条结构验证设置，再用 24 条互不重叠的结构复核。以 4090 的归档输出为参考，72 条结构的 high、medium 域计数和边界均匹配，233 个域 PDB 的文件集合和 SHA-256 均匹配。完整共识表 70 行逐字段相同，两条差异位于 low，部分单方法分数也有差异。9.82× 的性能数字来自上面的 4090 双卡历史测试。[5090 验证记录](evidence/blackwell-compatibility.json) · [复核命令](../docs/blackwell.md)
