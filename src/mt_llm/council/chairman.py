"""Chairman: final arbiter for the council."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mt_llm.council.prompts import build_chairman_prompt


class Chairman:
    """Selects (and may rewrite) the best candidate produced by council members."""

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

    def arbitrate(
        self,
        source: str,
        candidates: list[str],
        reviews: list[str] | None,
        terms: dict[str, str] | None,
        allow_rewrite: bool,
    ) -> str:
        prompt = build_chairman_prompt(
            source=source,
            candidates=candidates,
            reviews=reviews,
            terms=terms,
            allow_rewrite=allow_rewrite,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
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
