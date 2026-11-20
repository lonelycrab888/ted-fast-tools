# 固定回归输入

三个 PDB 来自 AlphaFold Protein Structure Database 的人类 v4 结构。文件逐字节复制自本次基准使用的父 PDB，保留其原子记录和残基编号。模型 ID 与数据库条目对应。[AlphaFold DB](https://alphafold.ebi.ac.uk/)

| 模型 | 长度 | 预期 high / medium | 覆盖情况 |
| --- | ---: | ---: | --- |
| AF-A0A1B0GVG4-F1-model_v4 | 234 | 0 / 0 | 三方法有预测，共识只含 low |
| AF-P0CG08-F1-model_v4 | 455 | 1 / 1 | 含不连续域 |
| AF-A2VEC9-F1-model_v4 | 1,400 | 3 / 5 | 长片段，含不连续域 |

`expected/` 的三个方法表、共识表与 10 个 SHA-256 取自完成历史原版比对的双卡全量档案。独立 UniDoc 的 TED 对照使用继承的 Merizo 边界。整链模式单独运行，预测可与该参考表不同。从仓库根目录运行 `python scripts/smoke_test.py --gpus 6 7` 可复核示例。

整链 UniDoc 的 P0CG08 预期结果另存为 `expected/whole_chain_P0CG08.tsv`，已与历史原版的原生程序及包装器单独核对。
