# Scripts

The command-line scripts for data, models, fine-tuning and result tables. Run them from the repository root, inside the
environment that `install.sh` creates (`source .venv/bin/activate`). Every script prints its options with `--help`;
the shell scripts describe theirs in their header.

`reproduce.py` runs all of these in the right order for the paper's experiments
([`benchmarks/README.md`](../benchmarks/README.md)). To run the steps one at a time:

1. **Data:** `prepare_genome.sh`, plus the benchmark scripts in `benchmarks/` and `get_real_reads.sh`.
2. **Models:** `download_models.py` for the released models, or `pretrain.py` to train your own.
3. **Fine-tuning:** `finetune.py` for LoCUS, `finetune_baseline.py` for the baselines.
4. **Tables:** `collect_results.py`, then `compare_results.py`.

| Script | What it does | Needs | Produces |
|---|---|---|---|
| `prepare_genome.sh` | downloads GRCh38.p14 (NCBI RefSeq GCF_000001405.40), checks its MD5, and cuts it into 4,096 bp windows of A, C, G and T only; `--download-only` skips the windows | network; `build/split_fasta` (`make tools`, done by `install.sh`) | `data/GRCh38.p14.fasta.gz`, `data/GRCh38.p14.acgt.w4096.s4096/` |
| `get_real_reads.sh` | cuts the NT18 test windows out of public HG002 nanopore reads (sup and hac basecalls) and checks them against the paper's checksums | network, about 25 min; prepares the genome and NT18 if they are missing | `data/benchmarks/real_reads/` (about 170 MB) |
| `download_models.py` | downloads the four LoCUS models (and, with `--baselines`, the three baselines) into the Hugging Face cache | network | cached models, loaded by name afterwards |
| `pretrain.py` | pretrains one model from a configuration in `configs/pretrain/`; `--resume` continues an interrupted run | the genome windows, one GPU, 11-19 h on an RTX A5000 | `runs/<model>/train_metrics.jsonl`, `runs/<model>/checkpoints/` (`final.pt` is the model) |
| `train_bpe.py` | retrains the BPE tokenizer of twin-bpe-2m (the paper's tokenizer is already in `data/tokenizers/`) | the genome windows, about 64 GB of RAM | `runs/tokenizers/bpe-4096-1gb-seed42.json` and a metadata file |
| `finetune.py` | fine-tunes a LoCUS model on one task and fold, and scores the test set | a model (name, folder or checkpoint), a recipe from `configs/finetune/`, prepared benchmark data | `fold-XX.json` (every epoch and the selected epoch's test score), `fold-XX.data_history.jsonl.gz`, and with `--save-best` the weights `fold-XX.best.pt` |
| `finetune_baseline.py` | the same for a baseline (Caduceus-PS, HyenaDNA) with the LoCUS protocol | `install.sh --baselines` | the same files |
| `reproduce.py` | runs the paper's experiments ("plans") on the GPUs you list, skips finished jobs, and builds the tables | `install.sh --all` | `results/` and `results/tables.md` |
| `collect_results.py` | gathers the results into the paper's tables | `results/` | `results/tables.md`, `results/tables.json` |
| `compare_results.py` | compares your tables with the paper's | `results/tables.json`, `reference/paper_tables.json` | a summary of the differences, per table |
| `upload_to_hub.py` | converts pretraining checkpoints to the Hugging Face layout and, with `--upload`, publishes them (authors only) | checkpoints, a Hugging Face token with write access | `config.json`, `model.safetensors` and a model card per model |
