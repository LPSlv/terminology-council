# Report

## 1) Key Information
- Title: 2-Stage Translation Pipeline (MT + LLM-checker)
- Team members (name – email): Not stated
- Cross-listed project: Not stated
- External mentor: Not stated
- External collaborators: Not stated

## 2) Abstract (2–4 sentences max)
- Motivation: Need for English-to-German translation system with terminology accuracy, consistency, and grammatical correctness.
- Goals: Implement 2-stage pipeline: machine translation (MT) stage followed by LLM-based checker/post-editor.
- Current findings/progress: System implemented with multi-GPU support, batch processing, and terminology dictionary enforcement.

## 3) Approach (max 6 bullets)
- Main method(s): 2-stage pipeline (MT model → LLM checker/post-editor)
- Baseline(s): MT model alone (can be disabled with --disable-checker flag)
- What is original vs borrowed: Pipeline architecture and checker logic (original); models from Hugging Face (borrowed)
- External code used (name/link if present): Hugging Face Transformers library
- Model(s)/architecture(s) mentioned: facebook/nllb-moe-54b (MT model used), Qwen/Qwen2.5-7B-Instruct (checker model used), also supports facebook/nllb-200-3.3B, facebook/nllb-200-1.3B, Helsinki-NLP/opus-mt-en-de, mistralai/Mistral-Large-Instruct-2411
- Key equations/ideas (only if explicitly written): Not stated

## 4) Experiments

### 4.1 Data (max 4 bullets)
- Dataset(s) + task for each: ende_dev.jsonl (English-to-German machine translation)
- Size/splits (if stated): Not stated
- Preprocessing (if stated): JSONL format with fields: en (source), de (reference), proper_terms (terminology dictionary)
- Data source/citation: Not stated

### 4.2 Evaluation (max 4 bullets)
- Metric(s): chrF++ (sacrebleu), terminology success rate
- How computed / settings (if stated): sacrebleu.sentence_chrf with word_order=2; terminology matching using word boundary regex patterns
- Baseline comparison method: MT-only output vs MT+Checker output

### 4.3 Experimental Details (max 10 bullets, factual only)
- Model config (layers, params, checkpoints, tokenizer, max lengths, etc.): MT model used: facebook/nllb-moe-54b; Checker model used: Qwen/Qwen2.5-7B-Instruct; tokenizer.model_max_length used; max_input_len = model_max_length - max_new_tokens - 10
- Hyperparameters (lr, batch size, epochs, optimizer, seeds, etc.): batch_size=8 (default), seed=0 (default, configurable via --seed), max_new_tokens=256 (default, configurable)
- Training/inference procedure: Inference-only (no training); deterministic generation (do_sample=False); batch processing supported; diagnostics enabled (--diagnose flag)
- Hardware/HPC details (extract all you can):
  - Cluster/node/hostname: Logged via socket.gethostname() in main.py
  - GPUs (type + count + memory if present): h200-141g:2 (from launch.sh), supports multi-GPU with device_map="auto" via accelerate
  - CUDA version: 12.1 (from launch.sh module load)
  - Python version: 3.10.10 (from launch.sh module load)
  - Torch/Transformers/Accelerate versions: torch>=2.0.0, transformers>=4.30.0, accelerate>=0.20.0 (from requirements.txt); versions logged at runtime
  - CUDA_VISIBLE_DEVICES / device_map / dtype: CUDA_VISIBLE_DEVICES logged; device_map="auto" for multi-GPU sharding; dtype: auto/fp16/bf16/fp32 (auto selects bf16 if supported, else fp16 on GPU)
  - Batch size / num examples: batch_size=8 (default), num_examples=5 for diagnostics (used in experiment)
  - Max_new_tokens / max_length settings: max_new_tokens=256 (default, configurable), max_length=None in generation, truncation=True with max_length=1024 for tokenizer
  - Any module loads / env vars relevant (eg PYTORCH_ALLOC_CONF): PYTORCH_ALLOC_CONF=expandable_segments:True (from launch.sh), TRANSFORMERS_OFFLINE for offline mode, HF_HOME and TRANSFORMERS_CACHE for cache directory
- Runtime/timing details (extract all you can):
  - Start time/date: Logged via time.strftime('%Y-%m-%d %H:%M:%S') in log_system_info
  - End time/date: Not stated
  - Total runtime: Not stated
  - Stage timings (loading, inference, metrics): Model loading time logged per model; batch processing shows items/s and s/batch; metrics computation time logged
  - Throughput (items/s, s/batch): Logged as items_per_sec and batch_time in main.py (e.g., "X.X items/s, X.XXs/batch")

### 4.4 Results (max 8 bullets)
- Main quantitative results (tables/values): Not stated
- Best result: Not stated
- Baseline result: Not stated
- Observations/commentary from text (1–3 bullets): System includes diagnostic mode to compare MT vs Checker outputs; checker can fallback to MT output if junk detected; terminology matching uses word boundaries
- What results suggest to try next (if stated): Not stated

## 5) Future Work (max 6 bullets)
- Next steps: Not stated
- Planned experiments: Not stated
- Stretch goals: Not stated
- Risks/unknowns (if stated): Not stated

## 6) References
- List cited datasets/models/papers/tools (just names, no commentary).
- PyTorch (torch>=2.0.0): https://pytorch.org/
- Hugging Face Transformers (transformers>=4.30.0): https://huggingface.co/docs/transformers
- Accelerate (accelerate>=0.20.0): https://huggingface.co/docs/accelerate
- sacrebleu (sacrebleu>=2.0.0): Post (2018) "A Call for Clarity in Reporting BLEU Scores"
- NLLB-200 (facebook/nllb-moe-54b, facebook/nllb-200-3.3B, facebook/nllb-200-1.3B): Costa-jussà et al. (2022) "No Language Left Behind: Scaling Human-Centered Machine Translation"
- OPUS-MT (Helsinki-NLP/opus-mt-en-de): Tiedemann & Thottingal (2020) "OPUS-MT – Building open translation services for the World"
- Qwen2.5 (Qwen/Qwen2.5-7B-Instruct): Qwen Team (2024) "Qwen2.5: A Party of Foundation Models"
- Mistral Large (mistralai/Mistral-Large-Instruct-2411): Mistral AI (2024) "Mistral Large"
- NumPy (numpy>=1.24.0): https://numpy.org/
- SentencePiece (sentencepiece>=0.1.99): Kudo & Richardson (2018) "SentencePiece: A simple and language independent subword tokenizer"

