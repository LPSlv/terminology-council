# 2-Stage Translation Pipeline

A simple 2-stage English-to-German translation system using Hugging Face Transformers. The pipeline consists of a machine translation (MT) stage followed by an LLM-based checker/post-editor that enforces terminology accuracy, consistency, and grammatical correctness.

## Overview

The system operates in two stages:

1. **Stage A: Machine Translation** - Translates English to German using a strong MT model from Hugging Face
2. **Stage B: Checker/Post-Editor** - Uses an instruction-tuned LLM to review, correct, and ensure terminology consistency

## Quick Start

```bash
# Basic usage (translate a sentence)
python main.py --text "Your English sentence here"

# With GPU acceleration (recommended for HPC)
python main.py --text "Your sentence" --device cuda

# Multi-GPU support (automatically uses all available GPUs)
python main.py --text "Your sentence" --device auto

# Skip checker stage (MT only)
python main.py --text "Your sentence" --no-checker

# With terminology dictionary
python main.py --text "Your sentence" --terms terms.json

# To change models, edit the MODEL CONFIGURATION section at the top of main.py

# Multi-GPU support (automatic when multiple GPUs available)
python main.py --text "Your sentence" --device auto
```

## Installation

### Step 1: Setup Virtual Environment

```bash
# Option 1: Using uv (if installed)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Option 2: Using standard venv
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First run will download models (~2-5 GB). This may take several minutes depending on your internet connection.

## Usage

### Basic Translation

```bash
python main.py --text "Open the consumption model containing the measures and attributes you want to include in your perspective, and click the Perspectives tab."
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--text "..."` | English sentence to translate (required) | - |
| `--device cpu\|cuda\|auto` | Device for inference | `auto` (multi-GPU) or `cuda` (single GPU) if available |
| `--no-checker` | Skip checker stage (MT only) | Disabled |
| `--terms <path.json>` | Path to terminology dictionary JSON | None |

### Example CLI Runs

```bash
# Basic translation
python main.py --text "Machine learning algorithms require large datasets."

# To use different models, edit the MODEL CONFIGURATION section at the top of main.py

# CPU-only mode
python main.py --text "Your text" --device cpu

# With terminology dictionary
python main.py --text "Your text" --terms my_terms.json

# MT only (no checker)
python main.py --text "Your text" --no-checker

# To use different models, edit the MODEL CONFIGURATION section at the top of main.py
```

## Model Configuration

Models are configured at the top of `main.py` in the `MODEL CONFIGURATION` section. Simply edit the constants to change models:

```python
# ============================================================================
# MODEL CONFIGURATION - Change models here
# ============================================================================
MT_MODEL = "facebook/nllb-200-1.3B"  # Machine Translation model
CHECKER_MODEL = "mistralai/Mistral-Large-Instruct-2411"  # Checker/Post-editor model
MAX_NEW_TOKENS = 512  # Maximum tokens for checker generation
# ============================================================================
```

### Default Models

- **MT Model (default)**: `facebook/nllb-200-1.3B` (~2.6GB)
  - Fallback: `Helsinki-NLP/opus-mt-en-de` (~300MB, if memory limited)
  
- **Checker Model (default)**: `mistralai/Mistral-Large-Instruct-2411` (~123B parameters)
  - Large instruction-tuned model with advanced reasoning capabilities
  - Supports multi-GPU distribution for large models
  - Alternative: `Qwen/Qwen2.5-1.5B-Instruct` (~3GB, smaller model)

### Model Requirements

- **GPU**: Recommended for reasonable performance
  - Single GPU: Works for smaller models (32GB V100 sufficient for Qwen/Qwen2.5-1.5B-Instruct)
  - Multi-GPU: Recommended for large models like Mistral-Large-Instruct-2411
- **CPU**: Works but very slow (not recommended for production)
- **Memory**: Varies by model
  - Small models: ~5-8GB for default models on GPU with fp16
  - Large models: Requires significant GPU memory (multi-GPU recommended)

### Model Selection Guidelines

**For HPC with single GPU (32GB V100):**
- MT: `facebook/nllb-200-1.3B` or `Helsinki-NLP/opus-mt-en-de`
- Checker: `Qwen/Qwen2.5-1.5B-Instruct` (fits comfortably)

**For HPC with multiple GPUs:**
- MT: `facebook/nllb-200-1.3B` or `Helsinki-NLP/opus-mt-en-de`
- Checker: `mistralai/Mistral-Large-Instruct-2411` (use `--device auto` for multi-GPU distribution)

**For memory-limited environments:**
- MT: `Helsinki-NLP/opus-mt-en-de` (smallest, ~300MB)
- Checker: `Qwen/Qwen2.5-1.5B-Instruct` or skip with `--no-checker`

**Note:** Do not use gated models (models requiring Hugging Face access approval).

## Terminology Dictionary

The checker can use a terminology dictionary to enforce exact translations for specific terms.

### Dictionary Format

Create a JSON file with English-to-German term mappings:

```json
{
  "consumption model": "Verbrauchsmodell",
  "perspective": "Perspektive",
  "measures": "Maßnahmen",
  "attributes": "Attribute"
}
```

Or use a nested format (compatible with test data):

```json
{
  "proper_terms": {
    "consumption model": "Verbrauchsmodell",
    "perspective": "Perspektive"
  }
}
```

### Usage

```bash
python main.py --text "Your text" --terms terms.json
```

The checker will:
- Verify that dictionary terms appear correctly in the translation
- Report any terminology issues in the output
- Use exact dictionary translations when present

## Output Format

The system outputs:

1. **Source text**: Original English input
2. **MT translation**: Raw machine translation output
3. **Final translation**: Checker-corrected output (or MT if checker skipped)
4. **Changes**: List of edits made by checker (if any)
5. **Terminology issues**: Terms that don't match the dictionary (if any)
6. **Consistency notes**: Observations about term consistency

### Example Output

```
================================================================================
RESULTS
================================================================================

Source text: Open the consumption model containing the measures and attributes...

MT translation: Öffnen Sie das Verbrauchsmodell, das die Maßnahmen und Attribute...

Final translation (after checker): Öffnen Sie das Verbrauchsmodell, das die Maßnahmen und Attribute...

Changes made (1):
  - Fixed grammatical agreement
    From: "die Maßnahmen"
    To: "die Maßnahmen"

Terminology issues (0):

Summary:
Changes made: 1
  - Fixed grammatical agreement
```

## Checker Output Format

The checker returns JSON with the following structure:

```json
{
  "final_translation": "[corrected German translation]",
  "changes": [
    {
      "from": "[original text]",
      "to": "[corrected text]",
      "reason": "[why changed]"
    }
  ],
  "terminology_issues": [
    {
      "source_term": "[English term]",
      "expected": "[correct German]",
      "found": "[what was in translation]"
    }
  ],
  "consistency_notes": [
    "[consistency observations]"
  ]
}
```

If the checker fails to produce valid JSON, it will auto-retry up to 2 times with a stricter correction prompt.

## Architecture

### Core Components

- **`Translator`** (`translator.py`): Loads and runs MT models
- **`Checker`** (`checker.py`): Loads and runs LLM checker with JSON output
- **`Pipeline`** (`pipeline.py`): Orchestrates the 2-stage workflow
- **`prompts.py`**: Contains checker prompt with strict JSON instructions
- **`utils.py`**: JSON parsing helpers and terminology matching

### Workflow

1. **Load models**: MT model and checker model (if not skipped)
2. **Stage A**: Translate English -> German using MT model
3. **Stage B**: Checker reviews MT output, enforces terminology, outputs JSON
4. **Output**: Display results with summary

### Consistency Memory

The pipeline maintains an in-memory consistency dictionary that tracks terminology decisions across runs. This can be extended to persist to disk in future versions.

## HPC Considerations

### Slurm Job Example

```bash
#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source venv/bin/activate
python main.py --text "Your sentence" --device cuda
```

### Memory Management

- Models use fp16 on GPU where possible
- Models are loaded sequentially (MT first, then checker)
- GPU cache is cleared after each model load
- If checker fails to load, pipeline continues with MT output only

### Deterministic Generation

- Checker uses `do_sample=False` for deterministic outputs
- Random seeds are set for reproducibility
- No sampling/temperature used

## Error Handling

- **MT model fails**: Pipeline exits with error
- **Checker model fails to load**: Pipeline continues with MT output only (logs warning)
- **Checker produces invalid JSON**: Auto-retries up to 2 times with stricter prompt
- **Checker fails after retries**: Falls back to MT output with warning

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB+ recommended)
- **Disk Space**: ~5-10 GB for model downloads (first run only)
- **GPU**: Recommended (32GB V100 sufficient for default models)
- **Dependencies**: See `requirements.txt`

## First Run Notes

- **Model Downloads**: Models are automatically downloaded from Hugging Face on first run
  - Total download size: ~5-10 GB (depending on models)
  - Models are cached in `~/.cache/huggingface/` for future use
  - Works offline after first download
- **GPU Performance**: 
  - Typical translation time: 5-15 seconds total (MT + checker)
  - CPU inference: Very slow (not recommended)

## Troubleshooting

**Out of memory errors:**
- Use smaller models: `--mt-model "Helsinki-NLP/opus-mt-en-de"`
- Skip checker: `--no-checker`
- Use CPU: `--device cpu` (very slow)

**Checker produces invalid JSON:**
- System auto-retries up to 2 times
- If still fails, falls back to MT output

**Model download fails:**
- Check internet connection
- Ensure sufficient disk space
- Try running again (downloads are cached)

## Differences from Previous Council System

This refactored version:
- ✅ Removed multi-member council, voting, anonymization, chairman logic
- ✅ Simplified to 2-stage pipeline (MT -> Checker)
- ✅ Checker outputs structured JSON (not free-form text)
- ✅ Deterministic generation (no sampling)
- ✅ Configurable models via CLI
- ✅ Optional terminology dictionary support
- ✅ Simple consistency memory (can be persisted later)
