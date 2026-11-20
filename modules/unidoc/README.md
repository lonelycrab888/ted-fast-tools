# UniDoc

UniDoc 在 CPU 上调用 STRIDE 与 `UniDoc_structure` 解析指定链的结构域。该入口支持 **继承边界** 和 **直接解析整条链**。两种模式都使用链 A，支持单个 PDB 或一个 PDB 目录。

## 配置

完整项目的 `tes-env` 已含所需依赖。仅部署 UniDoc 时，在 Linux x86_64 的 Python 3.10 环境中，从仓库根目录执行以下命令。无需配置 CUDA、PyTorch 或编译 GPU 扩展。

```bash
python -m pip install -r modules/unidoc/requirements.txt
python scripts/assets.py --download --module unidoc
```

## 独立解析整链

从本目录运行，无需其他方法的输出。此模式使用全部输入残基。

```bash
python run.py -i ../../examples/pdb -o ../../runs/unidoc-whole --whole-chain --workers 4
```

## 复现 TED 的 UniDoc 输入

传入已有的 `chopping_merizo.txt`。程序按 PDB 文件名去掉扩展名匹配行，读取第 5 列边界，按闭区间保留残基。继承表须含所有输入结构，每个 ID 一行，接受历史 TED 的六列无表头格式。继承值为 `NULL` 或 `NO_SS` 时使用整链。

```bash
python run.py -i ../../examples/pdb -o ../../runs/unidoc-inherit \
  --inherit ../../examples/expected/chopping_merizo.txt --workers 4
```

`--workers` 指并行处理结构数。输出目录下的 `unidoc/raw.tsv` 和 `chopping_unidoc.txt` 分别是原始和过滤结果，后者可与完整 TED 的方法表逐行对照。外部程序执行失败时，原始结果可出现 `NO_SS`，对应 ID 和错误在 `unidoc/run.log`。完整 TED 自动传递 Merizo 的过滤结果。

`unidoc_fast.py` 在独立临时目录准备每个结构，保留短 PDB 记录的兼容处理。`upstream/` 收录原始实现和经过修复的 AFDB 参考包装器。`run_unidoc.py` 供完整流程的 reference 后端使用。[上游说明](upstream/README.md)
