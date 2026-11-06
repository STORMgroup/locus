# LoCUS

LoCUS is a family of small DNA language models. Instead of k-mers or BPE tokens, LoCUS reads DNA as segments produced by
locally consistent parsing (LCP). Each segment depends only on the sequence around it, so a substitution, insertion or
deletion changes just the few segments next to it.

[Installation](#installation) · [Usage](#usage) · [Pretrained models](#pretrained-models) ·
[How LoCUS works](#how-locus-works) · [Reproducing the results](#reproducing-the-results) · [Citing](#citing) ·
[License](#license)

## Installation

You need Linux (x86-64), Python 3.9 or newer, gcc and make. Fine-tuning and pretraining need an NVIDIA GPU; one
fine-tune uses at most about 6 GB of GPU memory.

```bash
git clone https://github.com/STORMgroup/locus.git && cd locus
bash install.sh --models
source .venv/bin/activate
```

`install.sh` creates a virtual environment in `.venv`, installs the pinned package versions (`requirements.txt`), builds
the LCP library, runs the unit tests, and with `--models` downloads the pretrained models. PyTorch from PyPI is built for
CUDA 12.4; for the CUDA 11.8 build that the paper used, run `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash
install.sh --models`. `--baselines` always installs the CUDA 11.8 build, because the prebuilt kernels of the Caduceus
baselines do not load with the PyPI build. If you were given access to the models before their public release, log in
to Hugging Face first (`hf auth login`, or set `HF_TOKEN`).

<details>
<summary>The same steps by hand</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
make                                 # builds build/lcptools (the LCP library) and build/split_fasta
pytest                               # unit tests, CPU only, about 30 seconds
python scripts/download_models.py    # the pretrained models
```

</details>

## Usage

### Embed sequences

```python
import locus

model = locus.load("locus-2m")                              # downloaded from the Hugging Face Hub on first use
vectors = model.embed(["ACGTAGGCTA" * 50])                   # one 128-dimensional vector per sequence
per_base = model.embed(["ACGTAGGCTA" * 50], per_base=True)   # one 128-dimensional vector per base
print(vectors.shape, per_base[0].shape)                      # (1, 128) (500, 128)
```

The sequence vector is the mean over its bases. Both strands are embedded and averaged.

### Fine-tune on a benchmark task

```bash
python benchmarks/nt18/prepare.py      # downloads the revised Nucleotide Transformer benchmark to data/benchmarks/nt18
python scripts/finetune.py --pretrained locus-2m --recipe configs/finetune/locus-2m.yaml \
    --data-root data/benchmarks/nt18 --task H3K27ac_revised --fold 0 --micro-batches 2 \
    --output results/examples/locus-2m/H3K27ac_revised/fold-00.json
```

On an RTX A5000 this takes about 15 minutes. The paper's test MCC for this task is 0.469 ± 0.009 over its 10 folds
(0.487 on fold 0, its highest fold); runs of this example on fold 0 gave 0.448 to 0.484, because repeated GPU runs differ
(see [Reproducibility](benchmarks/README.md#reproducibility)).

The result file records every epoch and the test score of the epoch with the lowest validation loss. `--pretrained`
accepts a model name, a downloaded model folder, or your own checkpoint (below). Add `--save-best` to keep the fine-tuned
weights. `python scripts/finetune.py --help` lists all options.

### Pretrain your own model

```bash
bash scripts/prepare_genome.sh         # downloads GRCh38 and cuts it into 4,096 bp windows
python scripts/pretrain.py --config configs/pretrain/locus-2m.yaml --output-dir runs/locus-2m
```

Pretraining runs on one GPU and takes 9 to 19 hours per model on an RTX A5000. An interrupted run continues with
`--resume`. The pretrained model is `runs/locus-2m/checkpoints/final.pt`, which you can pass anywhere a model name works.

## Pretrained models

| Model | Parameters | Tokens |
|---|---|---|
| `locus-0.5m` | 471,556 | LCP segments |
| `locus-2m` | 1,737,604 | LCP segments |
| `twin-fixed-2m` | 1,737,604 | fixed 10 bp segments every 5 bp |
| `twin-bpe-2m` | 1,737,604 | BPE tokens (vocabulary 4,096) |

All four are pretrained on the human genome (GRCh38, chromosomes 3 to 22) with 4 kb inputs, about 12.5 billion bases
each. `twin-fixed-2m` and `twin-bpe-2m` are LoCUS-2M with another tokenizer; the paper uses them to compare tokenizers.
For the paper's pretraining-split ablation, the same four models are pretrained again without chromosomes 20 and 21
(`configs/pretrain/<model>-no-chr20-21.yaml`); they are released as `locus-0.5m-no-chr20-21`, `locus-2m-no-chr20-21`,
`twin-fixed-2m-no-chr20-21` and `twin-bpe-2m-no-chr20-21`.

The models are in the Hugging Face repository [`STORMResearch/locus`](https://huggingface.co/STORMResearch/locus), one
folder per model. They are downloaded the first time you use them; `python scripts/download_models.py` downloads the
four above ahead of time (`--split` the four split-ablation models). To work offline, set `LOCUS_HUB_DIR` to a directory with the same folders.

## How LoCUS works

1. **Base encoder.** Each base is embedded, and two layers of windowed self-attention (radius 32 and 128 bases) give it
   context.
2. **Segment pooling.** LCP cuts the sequence into overlapping segments of about 10 bp. The base vectors of each segment
   are pooled into one vector, and a small convolution lets each segment see its neighbours.
3. **Segment transformer.** Six transformer layers run over the segments.
4. **Per-base output.** Each base gets the mean of the segments that cover it.

Pretraining masks whole segments and predicts their bases.

## Reproducing the results

Every table in the paper can be regenerated with the scripts in this repository:

```bash
bash install.sh --all                                     # also installs the baseline models and PBSIM3
source .venv/bin/activate
python scripts/reproduce.py all --gpus 0,1,2,3 --jobs-per-gpu 3
python scripts/reproduce.py split-nt18 split-score --gpus 0,1,2,3 --jobs-per-gpu 3   # the split ablation (released split models)
python scripts/compare_results.py                         # how far your numbers are from the paper's
```

`--all` includes the baselines, whose CUDA kernels have prebuilt packages for Python 3.10 and newer; on Python 3.9 they
are compiled, which needs the CUDA toolkit (`nvcc`) and takes about 50 minutes. To switch an existing environment to
another Python, remove it first: `rm -rf .venv && PYTHON=python3.10 bash install.sh --all`.

`scripts/reproduce.py` downloads the data, runs every experiment on the listed GPUs, and writes the tables to
`results/tables.md`. Running a finished command again skips what is done. [`benchmarks/README.md`](benchmarks/README.md)
describes each experiment, the data, the run times, the baselines and how the tables are made.

## Citing

The citation will be added when the paper is published.

## License

GNU General Public License v3.0 (see `LICENSE`). `third_party/` holds the libraries we include, under their own licenses;
[`third_party/README.md`](third_party/README.md) lists every external tool, model and dataset with its version and any
change we made. The genomes, benchmarks, nanopore reads and baseline models are not part of this repository: the scripts
download them from their sources, and they stay under their own licenses and terms of use.
