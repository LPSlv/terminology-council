"""
Translation pipeline orchestrating MT and optional checker stages.
"""

import sys
import os
import importlib.util
import time
import torch
from typing import Dict, Optional, List

_script_dir = os.path.dirname(os.path.realpath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Import translator and checker modules
_translator_path = os.path.join(_script_dir, 'translator.py')
_checker_path = os.path.join(_script_dir, 'checker.py')

if not os.path.exists(_translator_path):
    _translator_path = os.path.join(os.getcwd(), 'translator.py')
if not os.path.exists(_checker_path):
    _checker_path = os.path.join(os.getcwd(), 'checker.py')

if not os.path.exists(_translator_path):
    raise ImportError(f"Cannot find translator.py in {_script_dir}")
if not os.path.exists(_checker_path):
    raise ImportError(f"Cannot find checker.py in {_script_dir}")

spec = importlib.util.spec_from_file_location("translator", _translator_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load translator module")
translator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(translator_module)
Translator = translator_module.Translator

spec = importlib.util.spec_from_file_location("checker", _checker_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load checker module")
checker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker_module)
Checker = checker_module.Checker


class Pipeline:
    """Orchestrates MT and optional checker stages."""
    
    def __init__(
        self,
        mt_model_id: str,
        checker_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cpu",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        disable_checker: bool = False,
        cache_dir: Optional[str] = None,
        compile_model: bool = False,
        terminology_mode: str = "on"
    ):
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.disable_checker = disable_checker
        self.cache_dir = cache_dir
        self.compile_model = compile_model
        self.mt_model_id = mt_model_id
        self.checker_model_id = checker_model_id
        self.terminology_mode = terminology_mode
        self.translator = None
        self.checker = None
        self.consistency_memory: Dict[str, str] = {}
    
    def load_models(self):
        """Load MT model and optionally checker model."""
        stage_start = time.time()
        
        # Load MT model
        mt_start = time.time()
        self.translator = Translator(
            model_id=self.mt_model_id,
            device=self.device,
            dtype=self.dtype,
            cache_dir=self.cache_dir,
            compile_model=self.compile_model,
            max_new_tokens=self.max_new_tokens
        )
        mt_time = time.time() - mt_start
        print(f"  MT model loaded: {self.mt_model_id} ({mt_time:.1f}s)")
        
        # Load checker model if enabled
        checker_time = 0.0
        if not self.disable_checker:
            if self.device in ["cuda", "auto"] and torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            checker_start = time.time()
            try:
                self.checker = Checker(
                    model_id=self.checker_model_id,
                    device=self.device,
                    dtype=self.dtype,
                    cache_dir=self.cache_dir,
                    compile_model=self.compile_model,
                    terminology_mode=self.terminology_mode
                )
                checker_time = time.time() - checker_start
                print(f"  Checker model loaded: {self.checker_model_id} ({checker_time:.1f}s)")
            except Exception as e:
                checker_time = time.time() - checker_start
                print(f"  Warning: Checker failed to load: {e}")
                self.checker = None
        
        total_time = time.time() - stage_start
        print(f"  Total load time: {total_time:.1f}s")
        
        # Print generation kwargs
        if self.translator and self.translator.tokenizer:
            max_input_len = self.translator.tokenizer.model_max_length
            print(f"Generation kwargs: max_new_tokens={self.max_new_tokens}, truncation=True, tokenizer_max_length={max_input_len}")
        else:
            print(f"Generation kwargs: max_new_tokens={self.max_new_tokens}, truncation=True")
    
    def run(
        self,
        source_text: str,
        terms: Optional[Dict[str, str]] = None,
        skip_checker: bool = False
    ) -> Dict:
        """Run translation pipeline on single text."""
        if self.translator is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # MT stage
        mt_translation = self.translator.translate(source_text)
        
        # Checker stage (if enabled)
        checker_result = None
        final_translation = mt_translation
        
        if not skip_checker and self.checker is not None:
            checker_result = self.checker.check(
                source_text=source_text,
                mt_translation=mt_translation,
                terms=terms,
                memory=self.consistency_memory,
                max_new_tokens=self.max_new_tokens
            )
            final_translation = checker_result.get("final_translation", mt_translation)
            
            # Update consistency memory
            if terms:
                for source_term, expected_term in terms.items():
                    if source_term and expected_term:
                        self.consistency_memory[source_term] = expected_term
        
        return {
            "source_text": source_text,
            "mt_translation": mt_translation,
            "final_translation": final_translation,
            "checker_result": checker_result
        }
    
    def run_batch(
        self,
        source_texts: List[str],
        terms_list: Optional[List[Optional[Dict[str, str]]]] = None,
        skip_checker: bool = False
    ) -> List[Dict]:
        """Run translation pipeline on batch of texts."""
        if self.translator is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        if not source_texts:
            return []
        
        if terms_list is None:
            terms_list = [None] * len(source_texts)
        elif len(terms_list) != len(source_texts):
            raise ValueError("terms_list must have the same length as source_texts")
        
        # MT stage
        mt_translations = self.translator.translate_batch(source_texts)
        
        # Checker stage (if enabled)
        checker_results = None
        # Make a copy to avoid aliasing - checker will modify final_translations but not mt_translations
        final_translations = mt_translations.copy()
        
        if not skip_checker and self.checker is not None:
            checker_results = self.checker.check_batch(
                source_texts=source_texts,
                mt_translations=mt_translations,
                terms_list=terms_list,
                memory=self.consistency_memory,
                max_new_tokens=self.max_new_tokens
            )
            
            # Extract final translations and update consistency memory
            for i, (checker_result, terms) in enumerate(zip(checker_results, terms_list)):
                final_translations[i] = checker_result.get("final_translation", mt_translations[i])
                
                if terms:
                    for source_term, expected_term in terms.items():
                        if source_term and expected_term:
                            self.consistency_memory[source_term] = expected_term
        
        # Build results
        results = []
        for i, (source_text, mt_translation, final_translation) in enumerate(zip(source_texts, mt_translations, final_translations)):
            checker_result = checker_results[i] if checker_results else None
            results.append({
                "source_text": source_text,
                "mt_translation": mt_translation,
                "final_translation": final_translation,
                "checker_result": checker_result
            })
        
        return results
