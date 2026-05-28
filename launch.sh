#!/bin/bash
#SBATCH -J mt_llm_job
#SBATCH --output=slurm-%x.%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 04:00:00

# More host RAM to avoid slurm OOM-kill during large model shard loading
#SBATCH --mem=120G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lpsdlv@gmail.com
#SBATCH --partition=gpu

# More GPUs on the node (adjust to what your cluster actually supports)
#SBATCH --gres=gpu:h200-141g:2

set -euxo pipefail

echo "Starting MT-LLM job on $(hostname) at $(date)"

module purge
module load cuda/12.1
module load python/3.10.10
module load py-pip/23.0

cd /gpfs/helios/home/lenardspatriks/transformers-course/mt-llm

# Create venv if missing
if [ ! -f ".venv/bin/activate" ]; then
  echo "No local .venv found, creating it in $(pwd)"
  python -m venv .venv
fi

source .venv/bin/activate

echo "Interpreter check:"
which python
python -V
python -c "import sys; print(sys.executable)"
python -m pip -V

# Ensure project deps are present
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# Install torch stack only if torch import fails
python -c "import torch" 2>/dev/null || {
  echo "Installing PyTorch stack..."
  python -m pip install torch torchvision torchaudio python-hostlist
}

# Verify key imports and GPU count
python -c "import torch; import transformers; import accelerate; print('OK:', torch.__version__, transformers.__version__, accelerate.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

# Reduce CUDA allocator fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Launching main.py..."
# python main.py
python main.py --data-file data/full_data.ende.jsonl --mt-model facebook/nllb-moe-54b --diagnose --num-examples 5 --terminology-mode on

echo "MT-LLM job completed at $(date)"

