#!/bin/bash
#SBATCH -J mt_llm
#SBATCH --output=slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141g:2

# Slurm launch template for the 2-stage pipeline on an HPC node.
# Set MT_LLM_REPO to the path of your checkout, then `sbatch scripts/launch.slurm`.

set -euxo pipefail

MT_LLM_REPO="${MT_LLM_REPO:-$HOME/mt-llm}"
cd "$MT_LLM_REPO"

echo "Starting mt_llm job on $(hostname) at $(date)"

module purge
module load cuda/12.1
module load python/3.10.10
module load py-pip/23.0

if [ ! -f ".venv/bin/activate" ]; then
    python -m venv .venv
fi
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]"

python -c "import torch, transformers, accelerate; \
print('torch', torch.__version__, 'transformers', transformers.__version__, \
'accelerate', accelerate.__version__, 'cuda', torch.cuda.is_available(), \
'gpus', torch.cuda.device_count())"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# 2-stage pipeline; switch to `council` subcommand to run the council.
python -m mt_llm translate \
    --data data/full_data.ende.jsonl \
    --mt-model facebook/nllb-moe-54b \
    --checker-model Qwen/Qwen2.5-7B-Instruct \
    --terminology-mode on \
    --output preds.jsonl

python -m mt_llm evaluate \
    --predictions preds.jsonl \
    --references data/full_data.ende.jsonl

echo "mt_llm job completed at $(date)"
