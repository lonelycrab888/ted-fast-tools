# TED-fast-tools

English · [简体中文](README.zh-CN.md)

TED-fast-tools predicts structural domains in single-chain protein PDBs using the Merizo, Chainsaw and UniDoc methods from *The Encyclopedia of Domains* (TED). Run any method on its own, or run all three to produce TED consensus boundaries and export high- and medium-consensus domains as individual PDB files.

On **2,349 structures using the same two RTX 4090 D GPUs**, the `ted-fast` pipeline reduced end-to-end time from **3 h 29 min 48.46 s to 21 min 22.17 s (9.82×)**. All 13,258 exported domain PDBs matched the original workflow by SHA-256. [Benchmark evidence](benchmarks/evidence/summary.json)

## Download and install

### 1. Get the code and create an environment

```bash
git clone https://github.com/lonelycrab888/ted-fast-tools.git
cd ted-fast-tools
```

If you received the source as a ZIP, extract it and open the directory containing `run_fast.py`. Run the remaining commands from the repository root. The full pipeline runs on Linux x86_64 with Conda, an NVIDIA driver, a CUDA Toolkit with `nvcc` and a C++17 compiler. Choose the environment for your GPU. The PyTorch wheel supplies the CUDA runtime used by PyTorch. The system toolkit compiles the Merizo extension.

| Tested GPU | Python | PyTorch | Tested nvcc | Tested driver |
| --- | --- | --- | --- | --- |
| RTX 4090 D | 3.10.22 | 2.0.1+cu118 | 12.6.85 | 565.77 |
| Blackwell (RTX 5090) | 3.11.13 | 2.7.1+cu128 | 12.8.93 | 580.178.04 |

**RTX 4090 D**

```bash
conda env create -f environment.yml
conda activate ted-fast-env
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

### 2. Download the weights, STRIDE and UniDoc executables

```bash
python scripts/assets.py --download
```

This installs **five model weight files and three Linux x86_64 executables** in their module directories. The script checks each file against the SHA-256 manifest, grants execute permission to the programs and skips files that already match. It downloads the UniDoc archive once for both binaries. [Asset manifest](assets/manifest.json)

| To download | Command |
| --- | --- |
| All five weights (three Merizo shards and two Chainsaw checkpoints) | `python scripts/assets.py --download-weights` |
| Both STRIDE binaries and `UniDoc_structure` | `python scripts/assets.py --download-native` |
| Everything needed for one method | `python scripts/assets.py --download --module chainsaw` (also accepts `merizo` or `unidoc`) |

**Manual download.** Each destination below is relative to `modules/<method>/upstream/`, with the method name in lowercase. Create the destination directories as needed. Keep the two identically named Chainsaw checkpoints in their respective subdirectories, and install the STRIDE build supplied for each method.

| Method | Destination under `upstream/` | Official download |
| --- | --- | --- |
| Merizo | `weights/weights_part_0.pt` | [weights_part_0.pt](https://raw.githubusercontent.com/psipred/Merizo/41d12fb84e6e8fdb586c2c859d12161dc7bb5bfd/weights/weights_part_0.pt) |
| Merizo | `weights/weights_part_1.pt` | [weights_part_1.pt](https://raw.githubusercontent.com/psipred/Merizo/41d12fb84e6e8fdb586c2c859d12161dc7bb5bfd/weights/weights_part_1.pt) |
| Merizo | `weights/weights_part_2.pt` | [weights_part_2.pt](https://raw.githubusercontent.com/psipred/Merizo/41d12fb84e6e8fdb586c2c859d12161dc7bb5bfd/weights/weights_part_2.pt) |
| Chainsaw | `saved_models/secondary_structure_epoch17/version_2/weights.pt` | [weights.pt](https://raw.githubusercontent.com/psipred/ted-tools/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/saved_models/secondary_structure_epoch17/version_2/weights.pt) |
| Chainsaw | `saved_models/ss_c_base_no_excl/version_2/epoch_11/weights.pt` | [weights.pt](https://raw.githubusercontent.com/psipred/ted-tools/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/saved_models/ss_c_base_no_excl/version_2/epoch_11/weights.pt) |
| Chainsaw | `stride/stride` | [TED STRIDE binary](https://raw.githubusercontent.com/psipred/ted-tools/6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/stride/stride) |
| UniDoc | `bin/stride` | [UniDoc_2023.tgz](https://yanglab.qd.sdu.edu.cn/UniDoc/download/UniDoc_2023.tgz) → `UniDoc/bin/stride` |
| UniDoc | `bin/UniDoc_structure` | [UniDoc_2023.tgz](https://yanglab.qd.sdu.edu.cn/UniDoc/download/UniDoc_2023.tgz) → `UniDoc/bin/UniDoc_structure` |

For a manual install, save the five weights and the Chainsaw STRIDE binary at the paths above. Download `UniDoc_2023.tgz` to the repository root, then extract the two UniDoc programs, set their permissions and verify all eight files. The one-command installer performs these steps automatically.

```bash
mkdir -p modules/unidoc/upstream/bin
tar -xzf UniDoc_2023.tgz -C modules/unidoc/upstream/bin --strip-components=2 \
  UniDoc/bin/stride UniDoc/bin/UniDoc_structure
chmod 755 modules/chainsaw/upstream/stride/stride \
  modules/unidoc/upstream/bin/stride modules/unidoc/upstream/bin/UniDoc_structure
python scripts/assets.py
```

### 3. Build the extensions

```bash
bash build.sh
```

This builds the Merizo CUDA kernels and the Chainsaw C++ domain-assignment library on the target machine. Set `NVCC` or `CXX` if your compilers are outside `PATH`. For a single method, use `bash build.sh merizo` or `bash build.sh chainsaw`. UniDoc runs on CPU and needs neither extension. [Installation and offline options](docs/INSTALL.md) · [Blackwell deployment](docs/blackwell.md)

## Run a single method

The standalone commands accept one `.pdb` file or a directory containing top-level `*.pdb` files. The filename stem is the structure ID. Use a **new output directory** for each run. The validated examples use AlphaFold DB structures with chain A and residues numbered consecutively from 1. Replace GPU indices with physical indices shown by `nvidia-smi`.

**Merizo** predicts domains on a GPU. Its filtered table, `runs/merizo/chopping_merizo.txt`, can also be passed to UniDoc.

```bash
python modules/merizo/run.py -i examples/pdb -o runs/merizo --gpu 0
```

**Chainsaw** computes residue-pair features and uses STRIDE secondary structure to assign domains. Its filtered predictions are in `runs/chainsaw/chopping_chainsaw.txt`.

```bash
python modules/chainsaw/run.py -i examples/pdb -o runs/chainsaw --gpu 0
```

For standalone Chainsaw on Blackwell, use the same precision settings as the full Blackwell pipeline.

```bash
env -u NVIDIA_TF32_OVERRIDE TED_CHAINSAW_DISABLE_TF32=1 \
  python modules/chainsaw/run.py -i examples/pdb -o runs/chainsaw-blackwell --gpu 0
```

**UniDoc** runs on CPU. If installing *only* UniDoc, use a Linux x86_64 Python 3.10 environment and install its two Python dependencies and native programs.

```bash
python -m pip install -r modules/unidoc/requirements.txt
python scripts/assets.py --download --module unidoc
```

To parse an entire chain independently, choose `--whole-chain`.

```bash
python modules/unidoc/run.py -i examples/pdb/AF-P0CG08-F1-model_v4.pdb \
  -o runs/unidoc-whole --whole-chain --workers 4
```

To reproduce UniDoc's input in the TED pipeline, run Merizo first and pass its **filtered** table with `--inherit`. The table must contain one row for every input ID. UniDoc uses the selected residues and writes `chopping_unidoc.txt` in its output directory. [Merizo](modules/merizo/README.md) · [Chainsaw](modules/chainsaw/README.md) · [UniDoc](modules/unidoc/README.md)

```bash
python modules/unidoc/run.py -i examples/pdb -o runs/unidoc-inherit \
  --inherit runs/merizo/chopping_merizo.txt --workers 4
```

## Run the full TED consensus pipeline

Pass a **directory** of PDBs. The driver starts all three methods, hands filtered Merizo boundaries to UniDoc, then combines the three filtered prediction tables using the TED consensus code. This command runs the bundled examples on two GPUs and exports high- and medium-consensus domains.

```bash
python run_fast.py -i examples/pdb -o runs/ted \
  --gpus 0 1 --overlap --unidoc-batch-size 32 --unidoc-workers 4 --export-domains
```

For a single GPU use `--gpus 0`. Each listed GPU needs at least one input structure. On Blackwell, use the launcher below to disable TF32 specifically for Chainsaw and keep Merizo at its validated precision.

```bash
./run-blackwell.sh -i examples/pdb -o runs/ted-blackwell \
  --gpus 0 1 --overlap --unidoc-batch-size 8 --unidoc-workers 4 --export-domains
```

| Option | Effect |
| --- | --- |
| `--variant ted-fast` | Uses the default optimized pipeline. |
| `--overlap` | Allows Merizo and Chainsaw to run concurrently on the same GPU. Omit it to reduce peak memory. |
| `--unidoc-batch-size N` | Hands groups of completed Merizo structures to CPU UniDoc workers. The default `0` waits for the whole shard. The dual-4090 benchmark used `32`. |
| `--unidoc-workers N` | CPU workers per GPU for UniDoc (default `4`). |
| `--structures-per-gpu N` | Independent structure queues per GPU (default `1`). Extra queues load extra model instances. |
| `--export-domains` | Writes and verifies one PDB per high- or medium-consensus domain. |

### Read the results

| Output in the run directory | Contents |
| --- | --- |
| `chopping_merizo.txt`, `chopping_chainsaw.txt`, `chopping_unidoc.txt` | One headerless TSV per method. Fields are structure ID, sequence MD5, length, domain count, boundaries and score. |
| `consensus.tsv` | One row per structure. The first three fields are ID, sequence MD5 and length, followed by high/medium/low counts and the three corresponding boundary fields. `na` means no domain at that level. |
| `domains/` and `domains.tsv` | With export enabled, the PDBs and a manifest with parent ID, confidence level, boundaries, residue and atom counts, mean Cα pLDDT, SHA-256 and file path. |
| `no_domain.tsv` | With export enabled, structures with zero retained high/medium domains. |
| `timing.json` and `gpu*/` | Runtime settings, timings and per-shard predictions and logs. With export enabled, `end_to_end_seconds` includes export and verification. |

Boundaries are inclusive. A comma separates domains, and an underscore joins discontinuous segments of one domain. Export preserves the parent's residue numbering and ATOM records. It expects one Cα per residue and blank insertion-code and alternate-location fields. TED's default filter keeps segments of at least 5 residues and domains of at least 25 residues. The three bundled examples yield **4 high and 6 medium** domain PDBs. To export an existing consensus table on its own, run the following. [Example structures and reference output](examples/README.md)

```bash
python export_domains.py -i examples/pdb --consensus examples/expected/consensus.tsv -o runs/export-only
```

## What changed from the original TED workflow?

| Component | Optimization in `ted-fast` |
| --- | --- |
| Merizo | Reuses the loaded model across structures. Native CUDA kernels accelerate point operations in invariant point attention (IPA). Repeated eligible GRU shapes reuse CUDA graphs. |
| Chainsaw | Prefetches STRIDE features while GPU inference proceeds, computes pairwise distances directly from Cα coordinates and uses compiled C++ for greedy domain assignment. |
| UniDoc | Prepares the selected PDB records and sequence metadata in one Python process. A CPU worker pool dispatches structures to the original STRIDE and `UniDoc_structure` executables as slots become available. |
| Scheduling | Balances structures across GPUs by PDB file size. On each GPU, model inference can overlap, and the CPU can start UniDoc on completed Merizo predictions while the rest of the shard is still running. |

### Measured performance on 2,349 structures

Both historical runs used the same **two RTX 4090 D GPUs**, the same structures and the same order per GPU (1,174 and 1,175 structures). The inputs were the `plddt70_v1` structures without a matching official TED v4 boundary. Timing covers launch through domain PDB export and verification. The original production run includes its UniDoc compatibility retries.

| Workflow | End-to-end wall time | Speedup | Wall time saved |
| --- | ---: | ---: | ---: |
| Original TED workflow | 3 h 29 min 48.46 s | 1.00× | — |
| TED-fast-tools `ted-fast` | **21 min 22.17 s** | **9.82×** | **3 h 8 min 26.29 s (89.81%)** |

The optimized run spent 1,259.46 s through consensus and 22.71 s on export and verification. All **2,349 rows** matched in each of the three method tables and the consensus table. All **13,258 domain PDBs** matched by SHA-256 (5,623 high and 7,635 medium). In total, 1,920 structures had at least one retained domain and 429 had none. [Timing](benchmarks/evidence/summary.json) · [Table comparisons](benchmarks/evidence/table-comparison.json) · [PDB verification](benchmarks/evidence/native-export-check.json) · [Reproduce the GPU assignment](benchmarks/README.md)

The separate Blackwell compatibility check covered 48 structures plus a disjoint set of 24. Against the 4090 reference, **72/72** high/medium boundary rows and **233/233** exported PDBs matched. The full consensus row matched for 70/72 structures, with two low-consensus differences. [Blackwell validation](benchmarks/evidence/blackwell-compatibility.json)

### Check an installation

Use the smoke test for your platform. It runs the three standalone methods, whole-chain UniDoc and the complete pipeline against the bundled reference outputs.

```bash
python scripts/smoke_test.py --platform 4090 --gpus 0 1 --output runs/check-4090
```

```bash
python scripts/smoke_test.py --platform blackwell --gpus 0 1 --output runs/check-blackwell
```

## Citation and upstream code

If you use TED domain predictions in your work, please cite the original TED paper.

Lau AM, Bordin N, Kandathil SM, Sillitoe I, Waman VP, Wells J, Orengo CA and Jones DT. *Exploring structural diversity across the protein universe with The Encyclopedia of Domains*. Science **386**, eadq4946 (2024). [DOI 10.1126/science.adq4946](https://doi.org/10.1126/science.adq4946)

The upstream implementation is maintained in the [official TED GitHub repository](https://github.com/psipred/ted-tools). The consensus code and runtime assets used here are documented in [source provenance and third-party notices](docs/SOURCES.md).
