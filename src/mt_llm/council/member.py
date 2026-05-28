"""A single council member: an instruction-tuned LLM that can translate and peer-review."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mt_llm.council.prompts import (
    build_member_review_prompt,
    build_member_translate_prompt,
)


class Member:
    """One council member backed by an HF causal LM."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        cache_dir: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None

    def _torch_dtype(self):
        if self.dtype == "fp16":
            return torch.float16
        if self.dtype == "bf16":
            return torch.bfloat16
        if self.dtype == "fp32":
            return torch.float32
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def load(self) -> None:
        kwargs = {"low_cpu_mem_usage": True, "dtype": self._torch_dtype()}
        if self.device == "auto" and torch.cuda.device_count() > 1:
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device in ("cuda", "auto") and torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def translate(self, source: str, terms: dict[str, str] | None = None) -> str:
        prompt = build_member_translate_prompt(source, terms=terms)
        return self._generate(prompt)

    def review(
        self,
        source: str,
        own_translation: str,
        other_translations: list[str],
    ) -> str:
        prompt = build_member_review_prompt(source, own_translation, other_translations)
        return self._generate(prompt)
