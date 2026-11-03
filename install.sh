#!/usr/bin/env bash
# Set up LoCUS: a Python environment with the pinned dependencies, the LCP library and the genome splitter, and the unit
# tests. Options add what only some experiments need:
#
#     bash install.sh                  # environment in .venv, build, tests
#     bash install.sh --models         # also download the four pretrained LoCUS models into the Hugging Face cache
#     bash install.sh --baselines      # also the baselines: mamba-ssm and causal-conv1d (Caduceus) and the two Caduceus-PS models
#     bash install.sh --pbsim3         # also build PBSIM3 (the simulated nanopore reads of the tokenizer study)
#     bash install.sh --all            # all of the above
#
# Requirements: Python 3.9 or newer, a C compiler (gcc) and zlib headers; a CUDA GPU for training. The paper used
# Python 3.9 and the CUDA 11.8 build of torch 2.6.0; torch 2.6.0 from PyPI is built for CUDA 12.4. For the paper's build:
#     TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash install.sh
# --baselines always uses the CUDA 11.8 build (unless TORCH_INDEX_URL is set): with torch from PyPI, the prebuilt
# mamba-ssm and causal-conv1d kernels fail to import (undefined symbol c10::detail::torchCheckFail).
#
# PYTHON chooses the interpreter of a new .venv (PYTHON=python3.11 bash install.sh); an existing .venv is kept, so to
# change its Python, remove it first (rm -rf .venv).
#
# --baselines: mamba-ssm and causal-conv1d publish prebuilt wheels for Python 3.10 and newer, which pip fetches (without
# nvcc, the local CUDA build of both packages is switched off so that their setup scripts go straight to the wheels). On
# Python 3.9 they are compiled, which needs the CUDA toolkit (nvcc) of torch's CUDA version and takes about 50 minutes
# (MAX_JOBS sets the number of compile jobs). causal-conv1d 1.7.0 writes type hints that need Python 3.10, so on 3.9 it is
# built from its source with third_party/causal-conv1d-1.7.0-python3.9.patch (the same change as in the paper's
# environment; it rewrites only the type hints).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON_GIVEN="${PYTHON:+1}"
PYTHON="${PYTHON:-python3}"

MODELS=0 BASELINES=0 PBSIM3=0
for option in "$@"; do
    case "$option" in
        --models) MODELS=1 ;;
        --baselines) BASELINES=1 ;;
        --pbsim3) PBSIM3=1 ;;
        --all) MODELS=1 BASELINES=1 PBSIM3=1 ;;
        *) echo "unknown option $option (options: --models --baselines --pbsim3 --all)" >&2; exit 2 ;;
    esac
done

VERSION='import sys; print("%d.%d" % sys.version_info[:2])'
if [[ ! -d .venv ]]; then
    "$PYTHON" -m venv .venv
elif [[ -n "$PYTHON_GIVEN" && "$(.venv/bin/python -c "$VERSION")" != "$("$PYTHON" -c "$VERSION")" ]]; then
    echo "the existing .venv uses Python $(.venv/bin/python -c "$VERSION"), not $PYTHON; remove it first (rm -rf .venv)" >&2
    exit 1
fi
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ $BASELINES == 1 && -z "${TORCH_INDEX_URL:-}" ]]; then
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
fi
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    # pip takes an installed torch 2.6.0 of another build (e.g. from PyPI) for the requested one, so remove it first
    build="${TORCH_INDEX_URL%/}"
    if python -c 'import torch' 2> /dev/null && [[ "$(python -c 'import torch; print(torch.__version__.partition("+")[2])')" != "${build##*/}" ]]; then
        python -m pip uninstall --yes torch
    fi
    python -m pip install torch==2.6.0 --index-url "$TORCH_INDEX_URL"
fi
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

make
python -m pytest -q tests

if [[ $BASELINES == 1 ]]; then
    if python -c 'import sys; sys.exit(sys.version_info >= (3, 10))' && ! command -v nvcc > /dev/null; then
        echo "--baselines on Python 3.9 compiles mamba-ssm and causal-conv1d and needs nvcc (the CUDA toolkit); load it, or" \
             "install with Python 3.10 or newer in a new environment (rm -rf .venv; PYTHON=python3.10 bash install.sh --baselines)" >&2
        exit 1
    fi
    python -m pip install ninja==1.13.2 packaging wheel
    if ! command -v nvcc > /dev/null; then
        export CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAMBA_SKIP_CUDA_BUILD=TRUE
    fi
    if python -c 'import sys; sys.exit(sys.version_info >= (3, 10))'; then
        source_dir=$(mktemp -d)
        curl --fail --location --silent --show-error --output "$source_dir/causal_conv1d-1.7.0.tar.gz" \
            https://files.pythonhosted.org/packages/f4/d1/eba27735f31bd5527d39a3c7f693e6ded7b844cab275db719a15ad37b6cd/causal_conv1d-1.7.0.tar.gz
        echo "3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b  $source_dir/causal_conv1d-1.7.0.tar.gz" | sha256sum --check --quiet
        tar xzf "$source_dir/causal_conv1d-1.7.0.tar.gz" -C "$source_dir"
        patch --quiet -p1 -d "$source_dir/causal_conv1d-1.7.0" < third_party/causal-conv1d-1.7.0-python3.9.patch
        python -m pip install --no-build-isolation "$source_dir/causal_conv1d-1.7.0"
        rm -rf "$source_dir"
    fi
    python -m pip install -r requirements-baselines.txt --no-build-isolation
    # the compiled extensions; mamba_ssm itself imports only where a CUDA driver is present
    python -c "import torch, causal_conv1d_cuda, selective_scan_cuda"
fi
if [[ $PBSIM3 == 1 ]]; then
    make pbsim3
fi
DOWNLOAD=()
[[ $MODELS == 1 ]] && DOWNLOAD+=(--locus)
[[ $BASELINES == 1 ]] && DOWNLOAD+=(--baselines)
if [[ ${#DOWNLOAD[@]} -gt 0 ]]; then
    python scripts/download_models.py "${DOWNLOAD[@]}"
fi
echo "LoCUS is installed; activate the environment with: source .venv/bin/activate"
