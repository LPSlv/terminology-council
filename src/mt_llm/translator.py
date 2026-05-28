"""English → German translation via HuggingFace pipeline."""

import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class Translator:
    """Handles machine translation from English to German."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: str = "auto",
        cache_dir: str | None = None,
        compile_model: bool = False,
        max_new_tokens: int = 256,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.compile_model = compile_model
        self.max_new_tokens = max_new_tokens
        self.pipeline = None
        self.tokenizer = None
        self.use_sharding = False
        self._load_model()

    def _is_large_model(self, model_id: str) -> bool:
        """Detect if model likely requires multi-GPU sharding."""
        model_lower = model_id.lower()
        large_indicators = ["moe", "54b", "11b", "13b", "30b", "65b", "70b"]
        return any(indicator in model_lower for indicator in large_indicators)

    def _get_torch_dtype(self):
        if self.dtype == "fp16":
            return torch.float16
        elif self.dtype == "bf16":
            return torch.bfloat16
        elif self.dtype == "fp32":
            return torch.float32
        else:  # auto
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            return torch.float32

    def _is_using_sharding(self, model=None):
        if model is not None:
            return hasattr(model, "hf_device_map") and model.hf_device_map is not None
        return self.use_sharding

    def _load_model(self):
        effective_device = "cuda" if self.device == "auto" else self.device
        use_gpu = (effective_device == "cuda") and torch.cuda.is_available()
        num_gpus = torch.cuda.device_count() if use_gpu else 0

        if self.cache_dir:
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir

        is_large = self._is_large_model(self.model_id)
        should_shard = use_gpu and num_gpus > 1 and is_large

        if should_shard:
            self.use_sharding = True
            self._load_with_sharding()
        else:
            self._load_single_device(use_gpu)

    def _load_with_sharding(self):
        try:
            torch_dtype = self._get_torch_dtype()
            if torch_dtype == torch.float32:
                torch_dtype = torch.bfloat16

            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, device_map="auto", dtype=torch_dtype, low_cpu_mem_usage=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # When using accelerate sharding, don't pass device argument to pipeline
            # The model already has device_map set, pipeline will respect it
            if "nllb" in self.model_id.lower():
                self.pipeline = pipeline(
                    "translation",
                    model=model,
                    tokenizer=self.tokenizer,
                    src_lang="eng_Latn",
                    tgt_lang="deu_Latn",
                )
            else:
                self.pipeline = pipeline("translation", model=model, tokenizer=self.tokenizer)

            # Override generation_config to use max_new_tokens instead of max_length
            if hasattr(self.pipeline.model, "generation_config"):
                # Set max_length to 1024 consistently for pipeline validation
                # max_new_tokens will take precedence during generation
                self.pipeline.model.generation_config.max_length = 1024
                self.pipeline.model.generation_config.max_new_tokens = self.max_new_tokens

            num_gpus = torch.cuda.device_count()
            print(f"  MT model sharded across {num_gpus} GPUs (device_map=auto)")

            gen_config = self.pipeline.model.generation_config
            print(
                f"  Pipeline config: tokenizer.model_max_length={self.tokenizer.model_max_length}, "
                f"generation.max_length={gen_config.max_length}, max_new_tokens={gen_config.max_new_tokens}, "
                f"device_map=auto, visible_devices={torch.cuda.device_count()}"
            )

        except Exception as e:
            import traceback

            print(f"ERROR loading MT model with sharding: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load MT model {self.model_id} with sharding: {e}") from e

    def _load_single_device(self, use_gpu: bool):
        device_map = 0 if use_gpu else -1

        try:
            if "nllb" in self.model_id.lower():
                self.pipeline = pipeline(
                    "translation",
                    model=self.model_id,
                    src_lang="eng_Latn",
                    tgt_lang="deu_Latn",
                    device=device_map,
                )
            else:
                self.pipeline = pipeline("translation", model=self.model_id, device=device_map)

            self.tokenizer = self.pipeline.tokenizer

            # Override generation_config to use max_new_tokens instead of max_length
            if hasattr(self.pipeline.model, "generation_config"):
                # Set max_length to 1024 consistently for pipeline validation
                # max_new_tokens will take precedence during generation
                self.pipeline.model.generation_config.max_length = 1024
                self.pipeline.model.generation_config.max_new_tokens = self.max_new_tokens

            gen_config = self.pipeline.model.generation_config
            device_info = f"device={device_map}" if not self.use_sharding else "device_map=auto"
            visible_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print(
                f"  Pipeline config: tokenizer.model_max_length={self.tokenizer.model_max_length}, "
                f"generation.max_length={gen_config.max_length}, max_new_tokens={gen_config.max_new_tokens}, "
                f"{device_info}, visible_devices={visible_devices}"
            )

            if use_gpu:
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(
                    f"  MT load OOM on single GPU; retrying with device_map=auto across {num_gpus} GPUs"
                )
                self.use_sharding = True
                self._load_with_sharding()
            else:
                raise RuntimeError(f"MT model OOM on single GPU: {e}") from e
        except Exception as e:
            import traceback

            print(f"ERROR loading MT model: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load MT model {self.model_id}: {e}") from e

    def translate(self, source_text: str) -> str:
        """Translate English text to German."""
        if self.pipeline is None:
            raise RuntimeError("MT model is not loaded")

        try:
            with torch.inference_mode():
                # Only pass max_new_tokens, not max_length (to avoid warnings)
                result = self.pipeline(
                    source_text, max_new_tokens=self.max_new_tokens, truncation=True
                )

            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    translation = result[0].get("translation_text", "")
                else:
                    translation = str(result[0])
            else:
                translation = str(result)

            return translation.strip()
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}") from e

    def translate_batch(self, source_texts: list) -> list:
        """Translate a batch of English texts to German."""
        if self.pipeline is None:
            raise RuntimeError("MT model is not loaded")

        if not source_texts:
            return []

        try:
            max_input_len = 0
            for text in source_texts:
                encoded = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=1024
                )
                max_input_len = max(max_input_len, encoded.input_ids.shape[1])

            with torch.inference_mode():
                # Only pass max_new_tokens, not max_length (to avoid warnings)
                results = self.pipeline(
                    source_texts, max_new_tokens=self.max_new_tokens, truncation=True
                )

            translations = []
            if not isinstance(results, list):
                results = [results]

            for result in results:
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        translation = result[0].get("translation_text", "")
                    else:
                        translation = str(result[0])
                elif isinstance(result, dict):
                    translation = result.get("translation_text", str(result))
                else:
                    translation = str(result)
                translations.append(translation.strip())

            return translations
        except Exception as e:
            raise RuntimeError(f"Batch translation failed: {e}") from e
