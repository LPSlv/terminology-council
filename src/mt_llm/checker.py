"""LLM post-editor for the 2-stage translation pipeline."""

import hashlib
import re
import sys
import os
from typing import Dict, Optional, List, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mt_llm.prompts import get_checker_prompt


class Checker:
    """Handles post-editing using an instruction-tuned LLM."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cpu",
        dtype: str = "auto",
        cache_dir: Optional[str] = None,
        compile_model: bool = False,
        terminology_mode: Literal["on", "off"] = "on",
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.compile_model = compile_model
        self.terminology_mode = terminology_mode
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _get_torch_dtype(self):
        if self.dtype == "fp16":
            return torch.float16
        elif self.dtype == "bf16":
            return torch.bfloat16
        elif self.dtype == "fp32":
            return torch.float32
        else:  # auto
            if torch.cuda.is_available():
                # Use bf16 if supported, else fp16
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            return torch.float32
    
    def _load_model(self):
        try:
            use_gpu = self.device == "cuda" and torch.cuda.is_available()
            has_multiple_gpus = torch.cuda.is_available() and torch.cuda.device_count() > 1
            wants_multi_gpu = self.device == "auto" and has_multiple_gpus
            
            # Check if accelerate is available for multi-GPU
            accelerate_available = False
            if wants_multi_gpu:
                try:
                    import accelerate
                    accelerate_available = True
                except ImportError:
                    accelerate_available = False
            
            use_multi_gpu = wants_multi_gpu and accelerate_available
            if wants_multi_gpu and not accelerate_available:
                use_gpu = True
            
            # Set cache directory
            if self.cache_dir:
                os.environ["HF_HOME"] = self.cache_dir
                os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
            
            model_kwargs = {}
            torch_dtype = self._get_torch_dtype()
            
            if use_gpu or use_multi_gpu:
                model_kwargs["dtype"] = torch_dtype
                model_kwargs["low_cpu_mem_usage"] = True
                if use_multi_gpu:
                    model_kwargs["device_map"] = "auto"
                    num_gpus = torch.cuda.device_count()
                    max_memory = {}
                    for i in range(num_gpus):
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        max_memory[i] = int(total_memory * 0.9)
                    model_kwargs["max_memory"] = max_memory
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            
            if use_gpu and not use_multi_gpu:
                self.model = self.model.to("cuda")
                torch.cuda.empty_cache()
            
            # Optional torch.compile
            if self.compile_model and hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception as e:
                    print(f"  Warning: torch.compile failed: {e}")
                
        except Exception as e:
            import traceback
            print(f"ERROR loading checker model: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            raise RuntimeError(f"Failed to load checker model {self.model_id}: {e}")
    
    def check(
        self,
        source_text: str,
        mt_translation: str,
        terms: Optional[Dict[str, str]] = None,
        memory: Optional[Dict[str, str]] = None,
        max_new_tokens: int = 256
    ) -> Dict:
        """Check and post-edit the MT translation."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Checker model is not loaded")
        
        prompt_terms = terms if self.terminology_mode == "on" else None
        prompt_memory = memory if self.terminology_mode == "on" else None
        prompt = get_checker_prompt(
            source_text,
            mt_translation,
            prompt_terms,
            prompt_memory,
            terminology_mode=self.terminology_mode,
        )
        output = self._generate(prompt, max_new_tokens)
        final_translation = self._extract_translation(output, mt_translation)

        mt_hash = hashlib.md5(mt_translation.encode()).hexdigest()[:8]
        checker_hash = hashlib.md5(final_translation.encode()).hexdigest()[:8]
        used_checker = (final_translation != mt_translation)
        
        # Detect if output is junk and fallback
        is_junk = self._is_junk_output(final_translation, mt_translation, source_text)
        if is_junk:
            if used_checker:
                print(f"  [Checker] Output detected as junk (MT_hash={mt_hash}, checker_hash={checker_hash}), using MT fallback")
            final_translation = mt_translation
        elif used_checker:
            print(f"  [Checker] Applied (MT_hash={mt_hash}, checker_hash={checker_hash})")
        
        changes = self._detect_changes(mt_translation, final_translation)
        terminology_issues = (
            self._check_terminology(final_translation, terms) if (self.terminology_mode == "on" and terms) else []
        )
        
        return {
            "final_translation": final_translation,
            "raw_output": output,
            "changes": changes,
            "terminology_issues": terminology_issues,
            "consistency_notes": []
        }
    
    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """Generate text from the model."""
        is_chat_model = "chat" in self.model_id.lower() or "instruct" in self.model_id.lower()
        
        if is_chat_model and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Calculate max input length: model_max_length - max_new_tokens - safety margin
            max_input_len = self.tokenizer.model_max_length - max_new_tokens - 10
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len
            )
        else:
            # Calculate max input length: model_max_length - max_new_tokens - safety margin
            max_input_len = self.tokenizer.model_max_length - max_new_tokens - 10
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len
            )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        torch_dtype = self._get_torch_dtype()
        use_autocast = (self.device in ["cuda", "auto"] and torch.cuda.is_available() and 
                       torch_dtype in [torch.float16, torch.bfloat16])
        
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        max_length=None,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    max_length=None,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_text = generated_text.strip()
        generated_text = re.sub(r'^(Assistant|User|System):\s*', '', generated_text, flags=re.IGNORECASE)
        
        return generated_text.strip()
    
    def _generate_batch(self, prompts: list, max_new_tokens: int) -> list:
        """Generate text from the model for a batch of prompts."""
        if not prompts:
            return []
        
        is_chat_model = "chat" in self.model_id.lower() or "instruct" in self.model_id.lower()
        device = next(self.model.parameters()).device
        tokenized_inputs = []
        
        # Calculate max input length: model_max_length - max_new_tokens - safety margin
        max_input_len = self.tokenizer.model_max_length - max_new_tokens - 10
        
        for prompt in prompts:
            if is_chat_model and hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(
                    formatted_prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_len
                )
            else:
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_len
                )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            tokenized_inputs.append(inputs)
        
        # Pad sequences for batch processing
        max_length = max(inp['input_ids'].shape[1] for inp in tokenized_inputs)
        
        batch_input_ids = []
        batch_attention_mask = []
        original_lengths = []
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        for inputs in tokenized_inputs:
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))[0]
            original_length = input_ids.shape[0]
            original_lengths.append(original_length)
            
            padding_length = max_length - original_length
            if padding_length > 0:
                padded_input_ids = torch.cat([torch.full((padding_length,), pad_token_id, device=device), input_ids])
                padded_attention_mask = torch.cat([torch.zeros((padding_length,), device=device), attention_mask])
            else:
                padded_input_ids = input_ids
                padded_attention_mask = attention_mask
            
            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
        
        batch_input_ids = torch.stack(batch_input_ids)
        batch_attention_mask = torch.stack(batch_attention_mask)
        
        torch_dtype = self._get_torch_dtype()
        use_autocast = (self.device in ["cuda", "auto"] and torch.cuda.is_available() and 
                       torch_dtype in [torch.float16, torch.bfloat16])
        
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    outputs = self.model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        max_new_tokens=max_new_tokens,
                        max_length=None,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            else:
                outputs = self.model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=max_new_tokens,
                    max_length=None,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        generated_texts = []
        for i, output in enumerate(outputs):
            input_length = max_length
            output_length = output.shape[0]
            
            if output_length > input_length:
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = ""
            
            generated_text = generated_text.strip()
            generated_text = re.sub(r'^(Assistant|User|System):\s*', '', generated_text, flags=re.IGNORECASE)
            generated_texts.append(generated_text.strip())
        
        return generated_texts
    
    def check_batch(
        self,
        source_texts: list,
        mt_translations: list,
        terms_list: Optional[list] = None,
        memory: Optional[Dict[str, str]] = None,
        max_new_tokens: int = 256
    ) -> list:
        """Check and post-edit a batch of MT translations."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Checker model is not loaded")
        
        if len(source_texts) != len(mt_translations):
            raise ValueError("source_texts and mt_translations must have the same length")
        
        if terms_list is None:
            terms_list = [None] * len(source_texts)
        elif len(terms_list) != len(source_texts):
            raise ValueError("terms_list must have the same length as source_texts")
        
        prompt_list = []
        for source_text, mt_translation, terms in zip(source_texts, mt_translations, terms_list):
            prompt_terms = terms if self.terminology_mode == "on" else None
            prompt_memory = memory if self.terminology_mode == "on" else None
            prompt = get_checker_prompt(
                source_text,
                mt_translation,
                prompt_terms,
                prompt_memory,
                terminology_mode=self.terminology_mode,
            )
            prompt_list.append(prompt)
        
        raw_outputs = self._generate_batch(prompt_list, max_new_tokens)
        
        results = []
        junk_count = 0
        applied_count = 0
        for i, (source_text, mt_translation, terms, raw_output) in enumerate(zip(source_texts, mt_translations, terms_list, raw_outputs)):
            final_translation = self._extract_translation(raw_output, mt_translation)
            
            # Debug: track checker usage with more detail
            used_checker = (final_translation.strip() != mt_translation.strip())
            mt_hash = hashlib.md5(mt_translation.encode()).hexdigest()[:8]
            checker_hash = hashlib.md5(final_translation.encode()).hexdigest()[:8]
            raw_hash = hashlib.md5(raw_output.encode()).hexdigest()[:8] if raw_output else "empty"
            
            # Detect if output is junk and fallback
            is_junk = self._is_junk_output(final_translation, mt_translation, source_text)
            if is_junk:
                if used_checker:
                    junk_count += 1
                    if i < 2:  # Log first 2 junk cases for debugging
                        print(f"  [Checker] Junk detected: raw_hash={raw_hash}, extracted_hash={checker_hash}, mt_hash={mt_hash}")
                final_translation = mt_translation
            elif used_checker:
                applied_count += 1
                if i < 2:  # Log first 2 applied cases for debugging
                    print(f"  [Checker] Applied: raw_hash={raw_hash}, extracted_hash={checker_hash}, mt_hash={mt_hash}")
            
            changes = self._detect_changes(mt_translation, final_translation)
            terminology_issues = (
                self._check_terminology(final_translation, terms) if (self.terminology_mode == "on" and terms) else []
            )
            
            results.append({
                "final_translation": final_translation,
                "raw_output": raw_output,
                "changes": changes,
                "terminology_issues": terminology_issues,
                "consistency_notes": []
            })
        
        if junk_count > 0 or applied_count > 0:
            print(f"  [Checker batch] Applied={applied_count}, Junk={junk_count}, Unchanged={len(source_texts)-applied_count-junk_count}/{len(source_texts)}")
        
        return results
    
    def _is_junk_output(self, output: str, mt_output: str, source_text: str) -> bool:
        """Detect if checker output is junk (commentary, glossary, repeats source, etc.)."""
        if not output or len(output) < 5:
            return True
        
        # If output equals MT, it's not junk (might be correct)
        if output.strip() == mt_output.strip():
            return False
        
        # Check for obvious commentary/instruction text
        junk_indicators = [
            "this is the corrected translation",
            "here is the corrected",
            "corrected german translation:",
            "review and improve",
            "your tasks:",
            "output only",
            "the translation is",
            "the corrected version"
        ]
        
        output_lower = output.lower()
        if any(indicator in output_lower for indicator in junk_indicators):
            return True
        
        # Check if output contains source language (English) - likely commentary
        # Simple heuristic: if output contains many English words from source
        # But be more lenient - some technical terms might appear in both languages
        source_words = set(source_text.lower().split())
        output_words = set(output_lower.split())
        common_english_words = source_words & output_words
        # Only flag as junk if >50% overlap AND output is significantly longer (likely commentary)
        if len(common_english_words) > len(source_words) * 0.5 and len(output) > len(mt_output) * 1.5:
            return True
        
        # Check for glossary-like patterns (repeated "X: Y" patterns)
        lines = output.split('\n')
        if len(lines) > 3:
            glossary_pattern = re.compile(r'^\s*\w+\s*:\s*\w+', re.IGNORECASE)
            glossary_count = sum(1 for line in lines if glossary_pattern.match(line.strip()))
            if glossary_count > len(lines) * 0.5:  # More than 50% are glossary-like
                return True
        
        return False
    
    def _extract_translation(self, output: str, fallback: str) -> str:
        """Extract translation from checker output, handling glossaries/commentary."""
        if not output:
            return fallback
        
        cleaned = output.strip()
        
        # Remove leading/trailing quotes
        cleaned = re.sub(r'^["\']+|["\']+$', '', cleaned)
        
        # Remove markdown code blocks
        if "```" in cleaned:
            code_block_match = re.search(r'```(?:[a-z]+)?\s*(.*?)\s*```', cleaned, re.DOTALL)
            if code_block_match:
                cleaned = code_block_match.group(1).strip()
        
        # Remove common prefixes
        cleaned = re.sub(r'^(Translation|German|Corrected|Output|Corrected German translation):\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*(Translation|German|Corrected|Output)\.?\s*$', '', cleaned, flags=re.IGNORECASE)
        
        # If output starts with prompt markers, try to extract after them
        prompt_markers = [
            r'Corrected German translation:\s*(.+?)(?:\n\n|\Z)',
            r'German translation:\s*(.+?)(?:\n\n|\Z)',
            r'Translation:\s*(.+?)(?:\n\n|\Z)',
            r'German:\s*(.+?)(?:\n\n|$)',
        ]
        
        for pattern in prompt_markers:
            match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Remove trailing prompt text
                extracted = re.split(r'(?:Review|Your tasks|English:|Corrected German translation:)', extracted, flags=re.IGNORECASE)[0].strip()
                if extracted and len(extracted) > 5:
                    cleaned = extracted
                    break
        
        # Handle multi-line output: keep first plausible paragraph, stop at glossary
        lines = cleaned.split('\n')
        translation_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                # Blank line: if we have translation content, stop here (glossary might follow)
                if translation_lines:
                    break
                continue
            
            # Skip glossary-like lines (X: Y pattern)
            if re.match(r'^\s*\w+\s*:\s*\w+', line):
                if translation_lines:  # Already have translation, this is glossary
                    break
                continue
            
            # Skip obvious commentary lines
            if any(word in line.lower() for word in ['review', 'translation', 'quality', 'accuracy', 'this is', 'here is']):
                if translation_lines:  # Already have translation
                    break
                continue
            
            translation_lines.append(line)
        
        if translation_lines:
            cleaned = ' '.join(translation_lines).strip()
        
        # Final validation
        if not cleaned or len(cleaned) < 5:
            return fallback
        
        # If it doesn't look like German and contains English prompt words, use fallback
        has_german_chars = any(char in cleaned for char in 'äöüßÄÖÜ')
        has_english_prompt = any(word in cleaned.lower() for word in ['review', 'translation', 'quality', 'accuracy', 'consistency', 'terminology', 'english'])
        
        if not has_german_chars and has_english_prompt:
            return fallback
        
        return cleaned
    
    def _detect_changes(self, original: str, corrected: str) -> List[Dict[str, str]]:
        """Detect changes between original and corrected translation."""
        changes = []
        if original.strip() != corrected.strip():
            changes.append({
                "from": original,
                "to": corrected,
                "reason": "Translation improved by checker"
            })
        return changes
    
    def _check_terminology(self, translation: str, terms: Dict[str, str]) -> List[Dict[str, str]]:
        """Check if terminology terms appear correctly in translation."""
        issues = []
        translation_lower = translation.lower()
        
        for en_term, expected_de_term in terms.items():
            pattern = r'\b' + re.escape(expected_de_term.lower()) + r'\b'
            if not re.search(pattern, translation_lower):
                issues.append({
                    "source_term": en_term,
                    "expected": expected_de_term,
                    "found": "not found"
                })
        
        return issues
