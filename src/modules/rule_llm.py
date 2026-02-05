# rule_llm.py
# Cost-Performance Balance Strategy: Rule + Embedding + LLM Hybrid
#
# This strategy uses different methods for cost tiers:
# - LOW (0.2): Rule-based matching and keyword extraction
# - MID (0.5): Embedding similarity + rule-based filtering
# - HIGH (1.0): Full LLM inference with complex reasoning

import torch
import json
import re
import numpy as np
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from json_repair import repair_json
from ..utils.llm_utils import get_llm_response
from ..utils.llm_pricing import calculate_cost, normalize_cost
from ..prompts.prompt_pool import (
    MODULE1_FILTER_PROMPT_Direct,
    MODULE1_FILTER_PROMPT_COT,
    MODULE1_FILTER_PROMPT_REACT,
    MODULE2_ENTITY_RELATION_PROMPT_Direct,
    MODULE2_ENTITY_RELATION_PROMPT_COT,
    MODULE2_ENTITY_RELATION_PROMPT_REACT,
    MODULE3_TEMPORAL_RELATION_PROMPT_Direct,
    MODULE3_TEMPORAL_RELATION_PROMPT_COT,
    MODULE3_TEMPORAL_RELATION_PROMPT_REACT,
    MODULE4_SUMMARY_PROMPT_Direct,
    MODULE4_SUMMARY_PROMPT_COT,
    MODULE4_SUMMARY_PROMPT_REACT,
    MODULE5_TOPIC_RELATION_PROMPT_Direct,
    MODULE5_TOPIC_RELATION_PROMPT_COT,
    MODULE5_TOPIC_RELATION_PROMPT_REACT,
)

# REBEL related imports and utility functions
try:
    import torch as torch_rebel
except ImportError:
    torch_rebel = None


def ensure_torch_rebel():
    """Ensure PyTorch is available (for REBEL)"""
    if torch_rebel is None:
        raise RuntimeError("REBEL requires PyTorch. Install with: pip install torch")


def rebel_extract_triplets_from_decoded(decoded_text: str) -> List[Tuple[str, str, str]]:
    """
    REBEL output parser: parse <triplet> <subj> <obj> special tokens
    """
    triplets: List[Tuple[str, str, str]] = []
    subject, relation, object_ = "", "", ""
    current = "x"  # t=subject, s=object, o=relation

    text = decoded_text.strip()
    text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()

    for token in text.split():
        if token == "<triplet>":
            if subject.strip() and relation.strip() and object_.strip():
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            subject, relation, object_ = "", "", ""
            current = "t"
        elif token == "<subj>":
            if subject.strip() and relation.strip() and object_.strip():
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            object_ = ""
            current = "s"
        elif token == "<obj>":
            relation = ""
            current = "o"
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token

    if subject.strip() and relation.strip() and object_.strip():
        triplets.append((subject.strip(), relation.strip(), object_.strip()))

    return triplets


def guess_entity_type(entity: str) -> str:
    """Simple entity type guessing"""
    entity = entity.strip()
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$", entity):
        return "PERSON"
    if any(w in entity for w in ["Corporation", "Company", "Inc", "Ltd", "University", "Corp", "LLC"]):
        return "ORG"
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$", entity) and len(entity.split()) <= 3:
        return "LOC"
    return "UNKNOWN"


def token_sliding_chunks(tokenizer, text: str, max_input_tokens: int = 256, stride: int = 64) -> List[str]:
    """Sliding window chunking based on tokenizer token sequence"""
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if not input_ids:
        return []

    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_input_tokens, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(input_ids):
            break
        start = max(0, end - stride)

    return chunks


# ============================================================================
# Cost Level Enum
# ============================================================================

def _extract_tokens_from_llm_response(ret) -> Tuple[int, int]:
    """
    Extract token counts from get_llm_response return value
    
    Args:
        ret: Return value from get_llm_response, format: [(q_id, response, (input_tokens, output_tokens), success), ...]
    
    Returns:
        (input_tokens, output_tokens), returns (0, 0) on failure
    """
    if len(ret) == 0:
        return 0, 0
    
    _, _, token_info, success = ret[0]
    if not success:
        return 0, 0
    
    if isinstance(token_info, tuple) and len(token_info) == 2:
        return token_info[0], token_info[1]
    
    return 0, 0


def _calculate_output_tokens(text_list: List[str]) -> int:
    """
    Calculate total token count for string list (approximate estimation)

    Args:
        text_list: List of strings

    Returns:
        Total token count (approximate: character count / 4)
    """
    if not text_list:
        return 0

    total_chars = sum(len(text) for text in text_list)
    estimated_tokens = total_chars // 4
    estimated_tokens = estimated_tokens / 1000000
    return max(estimated_tokens, 1)


def _calculate_token_cost(model_name: str, ret) -> float:
    """
    Calculate raw cost (without normalization) based on model name and LLM response

    Args:
        model_name: Model name
        ret: Return value from get_llm_response

    Returns:
        Raw cost (USD), returns 0.0 on failure
    """
    input_tokens, output_tokens = _extract_tokens_from_llm_response(ret)
    if input_tokens == 0 and output_tokens == 0:
        return 0.0
    return calculate_cost(model_name, input_tokens, output_tokens)


class CostLevel:
    LOW = 0      # Low cost: simple rules/keywords
    MID = 1      # Mid cost: embedding computation
    HIGH = 2     # High cost: LLM calls


# ============================================================================
# Module Output Data Structures
# ============================================================================

@dataclass
class ModuleOutput:
    """Output of a single module"""
    content: str
    cost_level: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryModuleOutputs:
    """Stores processing outputs from each module for memories"""
    filter_output: Optional[ModuleOutput] = None
    entity_relation_output: Optional[ModuleOutput] = None
    temporal_relation_output: Optional[ModuleOutput] = None
    topic_relation_output: Optional[ModuleOutput] = None
    summary_output: Optional[ModuleOutput] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {}
        if self.filter_output:
            result['filter'] = {
                'content': self.filter_output.content,
                'cost_level': self.filter_output.cost_level,
                'metadata': self.filter_output.metadata
            }
        if self.entity_relation_output:
            result['entity_relation'] = {
                'content': self.entity_relation_output.content,
                'cost_level': self.entity_relation_output.cost_level,
                'metadata': self.entity_relation_output.metadata
            }
        if self.temporal_relation_output:
            result['temporal_relation'] = {
                'content': self.temporal_relation_output.content,
                'cost_level': self.temporal_relation_output.cost_level,
                'metadata': self.temporal_relation_output.metadata
            }
        if self.topic_relation_output:
            result['topic_relation'] = {
                'content': self.topic_relation_output.content,
                'cost_level': self.topic_relation_output.cost_level,
                'metadata': self.topic_relation_output.metadata
            }
        if self.summary_output:
            result['summary'] = {
                'content': self.summary_output.content,
                'cost_level': self.summary_output.cost_level,
                'metadata': self.summary_output.metadata
            }
        return result

# ============================================================================
# Module 1: Filter Module
# ============================================================================

class Module1_Filter:
    """
    Module 1: Filter Module

    - Low cost: Sparse filtering (simple filtering based on keyword matching)
    - Mid cost: Existing filtering method (embedding similarity + keyword boost)
    - High cost: LLM-based filtering method (using LLM scoring)
    """

    COST_WEIGHTS = {
        CostLevel.LOW: 0.0,  # Module1 LOW cost is free
        CostLevel.MID: 0.0,  # Module1 MID cost is free
        CostLevel.HIGH: 1.0,
    }

    def __init__(
        self,
        encoder: Any = None,
        llm_func: Any = None,
        args: Any = None,
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ):
        self.encoder = encoder
        self.llm_func = llm_func
        self.args = args
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def execute(
        self,
        query: str,
        memories: List[Any],
        query_emb: Optional[np.ndarray] = None,
        cost_level: int = CostLevel.MID
    ) -> Tuple[List[Any], float]:
        """
        Execute filtering operation

        Args:
            query: Query text
            memories: Input memory list
            query_emb: Query embedding
            cost_level: Cost level

        Returns:
            (Filtered memory list, cost value)
        """
        if cost_level == CostLevel.LOW:
            result = self._low_cost_filter(query, memories)
            return result, self.COST_WEIGHTS[CostLevel.LOW]
        elif cost_level == CostLevel.MID:
            result = self._mid_cost_filter(query, memories, query_emb)
            return result, self.COST_WEIGHTS[CostLevel.MID]
        else:  # HIGH
            result, ret = self._high_cost_filter(query, memories)
            model_name = getattr(self.args, 'model', 'meta/llama-3.3-70b-instruct')
            cost = _calculate_token_cost(model_name, ret) if ret else self.COST_WEIGHTS[CostLevel.HIGH]
            return result, cost

    def _low_cost_filter(self, query: str, memories: List[Any]) -> List[Any]:
        """
        Low cost: Sparse filtering - based on keyword matching
        """
        query_words = set(query.lower().split())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of',
                    'and', 'in', 'on', 'for', 'what', 'when', 'where', 'who', 'how'}
        query_words = query_words - stopwords

        scored_memories = []
        for mem in memories:
            content_words = set(mem.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / len(query_words) if query_words else 0
                scored_memories.append((mem, score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)
        result = [m for m, s in scored_memories[:self.top_k]]

        if len(result) == 0 and len(memories) > 0:
            return memories[:self.top_k]
        return result

    def _mid_cost_filter(
        self,
        query: str,
        memories: List[Any],
        query_emb: Optional[np.ndarray] = None
    ) -> List[Any]:
        """
        Mid cost: embedding similarity + keyword boost
        """
        if query_emb is None or len(memories) == 0:
            return self._low_cost_filter(query, memories)

        if isinstance(query_emb, torch.Tensor):
            query_emb = query_emb.cpu().numpy()
        query_emb = query_emb.flatten().astype('float32')
        query_emb_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        scored_memories = []
        for mem in memories:
            if mem.embedding is None:
                continue

            mem_emb = mem.embedding
            if isinstance(mem_emb, torch.Tensor):
                mem_emb = mem_emb.cpu().numpy()
            mem_emb = mem_emb.flatten().astype('float32')
            mem_emb_norm = mem_emb / (np.linalg.norm(mem_emb) + 1e-8)

            sim_score = float(np.dot(query_emb_norm, mem_emb_norm))

            keyword_boost = self._compute_keyword_boost(query, mem.content)
            final_score = sim_score * 0.7 + keyword_boost * 0.3

            if final_score >= self.similarity_threshold:
                mem.relevance_score = final_score
                scored_memories.append((mem, final_score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)
        result = [m for m, s in scored_memories[:self.top_k]]

        if len(result) == 0 and len(memories) > 0:
            return memories[:self.top_k]
        return result

    def _high_cost_filter(self, query: str, memories: List[Any]) -> Tuple[List[Any], Any]:
        """
        High cost: Use LLM scoring for filtering
        """

        memories_text = ""
        for idx, mem in enumerate(memories):
            content_preview = mem.content
            mem_metadata = []
            if hasattr(mem, 'date_time') and mem.date_time:
                mem_metadata.append(f'date_time="{mem.date_time}"')
            if hasattr(mem, 'session_id') and mem.session_id:
                mem_metadata.append(f'session_id="{mem.session_id}"')
            if hasattr(mem, 'dia_id') and mem.dia_id:
                mem_metadata.append(f'dia_id="{mem.dia_id}"')
            metadata_str = " " + " ".join(mem_metadata) if mem_metadata else ""
            memories_text += f"\n<memory index=\"{idx}\"{metadata_str}>\n{content_preview}\n</memory>"

        prompt = MODULE1_FILTER_PROMPT_REACT.format(query=query, memories_text=memories_text)   

        task_args = [(0, prompt, self.args)]
        ret = get_llm_response(args=self.args, task_args=task_args, disable_internal_threading=True)

        if len(ret) == 0:
            return self._mid_cost_filter(query, memories, None), []

        _, response, _, success = ret[0]
        if not success:
            return self._mid_cost_filter(query, memories, None), ret

        try:
            response = response.strip()

            # Preferred: extract scores inside <answer>...</answer> if present
            if "<answer>" in response and "</answer>" in response:
                start = response.find("<answer>") + len("<answer>")
                end = response.find("</answer>", start)
                if end != -1:
                    response = response[start:end].strip()

            if response.startswith("```"):
                parts = response.split("```")
                if len(parts) >= 2:
                    response = parts[1]
                    if response.startswith("json"):
                        response = response[4:]
                    response = response.strip()

            # Try to parse as JSON array first (prompt_pool format: [s0, s1, ...])
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
            else:
                # Fallback: try JSON object
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                else:
                    json_str = response

            # Try to parse JSON, if fails, try to repair it
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    repaired_json = repair_json(json_str)
                    parsed = json.loads(repaired_json)
                except Exception as repair_error:
                    print(f"[Module1_Filter._high_cost_filter] JSON repair failed: {repair_error}, fallback to mid-cost")
                    return self._mid_cost_filter(query, memories, None), ret

            # Extract response field from JSON object, or use array directly
            if isinstance(parsed, dict) and 'response' in parsed:
                scores = parsed['response']
            elif isinstance(parsed, list):
                scores = parsed
            else:
                scores = None

            if not isinstance(scores, list):
                return self._mid_cost_filter(query, memories, None), ret

            scored_memories = []
            for idx, mem in enumerate(memories[:len(scores)]):
                score = scores[idx] / 10.0 if idx < len(scores) else 0
                mem.relevance_score = score
                scored_memories.append((mem, score))

            scored_memories.sort(key=lambda x: x[1], reverse=True)
            return [m for m, s in scored_memories[:self.top_k]], ret
        except Exception as e:
            print(f"[Module1_Filter._high_cost_filter] JSON parsing failed: {e}, fallback to mid-cost")
            return self._mid_cost_filter(query, memories, None), ret

    def _compute_keyword_boost(self, query: str, content: str) -> float:
        """Compute keyword boost"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'on', 'for'}
        query_words = query_words - stopwords

        if not query_words:
            return 0.0

        overlap = len(query_words & content_words)
        return min(overlap / len(query_words), 1.0)


# ============================================================================
# Shared REBEL Model Cache (shared by Module2 & Module3)
# ============================================================================

class _SharedREBELCache:
    """
    Shared REBEL model cache to avoid loading the model twice
    Supports preloading and parallel inference with per-thread tokenizers
    """
    model = None
    tokenizer = None  # Shared tokenizer (read-only after loading)
    device = None
    load_lock = threading.RLock()  # Reentrant lock to allow nested calls
    inference_semaphore = None  # Limits concurrent inference
    enabled = True  # Can be set to False if loading fails

    # Per-thread tokenizer instances for parallel inference
    _thread_tokenizers = {}
    _tokenizer_lock = threading.Lock()

    # Batch processing queue for further optimization
    use_batching = False
    max_batch_size = 8
    batch_timeout = 0.01  # seconds


_shared_rebel_cache = _SharedREBELCache()


def preload_rebel_model(
    model_name: str = "Babelscape/rebel-large",
    max_concurrent_inference: int = 4,
    use_batching: bool = False,
    device: str = "auto"
) -> bool:
    """
    Preload REBEL model at system initialization

    Args:
        model_name: HuggingFace model name
        max_concurrent_inference: Maximum number of concurrent inference calls (0 = unlimited)
        use_batching: Enable batch processing for better throughput
        device: Target device ('auto', 'cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        True if successful, False otherwise
    """
    if not _shared_rebel_cache.enabled:
        print("[REBEL] Preload skipped: REBEL is disabled")
        return False

    with _shared_rebel_cache.load_lock:
        if _shared_rebel_cache.model is not None:
            print("[REBEL] Model already loaded, skipping preload")
            return True

        try:
            ensure_torch_rebel()
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            # print(f"[REBEL] Preloading model: {model_name}")

            # Determine device
            if device == "auto":
                if torch_rebel.cuda.is_available():
                    _shared_rebel_cache.device = torch_rebel.device("cuda:0")
                else:
                    _shared_rebel_cache.device = torch_rebel.device("cpu")
            else:
                _shared_rebel_cache.device = torch_rebel.device(device)

            # Load tokenizer (shared, read-only)
            _shared_rebel_cache.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=False
            )

            # Load model
            _shared_rebel_cache.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=False,
                local_files_only=False
            )

            # Handle meta parameters
            if any(p.is_meta for p in _shared_rebel_cache.model.parameters()):
                print("[REBEL] Detected meta parameters, materializing weights...")
                _shared_rebel_cache.model = _shared_rebel_cache.model.to_empty(
                    device=_shared_rebel_cache.device
                )
                temp_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=False,
                    torch_dtype=torch_rebel.float32,
                    local_files_only=False
                )
                _shared_rebel_cache.model.load_state_dict(temp_model.state_dict())
                del temp_model
            else:
                _shared_rebel_cache.model = _shared_rebel_cache.model.to(_shared_rebel_cache.device)

            _shared_rebel_cache.model.eval()

            # Setup concurrency control
            if max_concurrent_inference > 0:
                _shared_rebel_cache.inference_semaphore = threading.Semaphore(max_concurrent_inference)
                print(f"[REBEL] Concurrency limit: {max_concurrent_inference}")
            else:
                _shared_rebel_cache.inference_semaphore = None
                print("[REBEL] Concurrency: unlimited (use with caution)")

            # Setup batching
            _shared_rebel_cache.use_batching = use_batching
            if use_batching:
                print(f"[REBEL] Batching enabled: max_batch_size={_shared_rebel_cache.max_batch_size}")

            print(f"[REBEL] Model preloaded successfully on {_shared_rebel_cache.device}")
            return True

        except Exception as e:
            print(f"[REBEL] Failed to preload model: {e}")
            _shared_rebel_cache.enabled = False
            return False


def get_thread_tokenizer():
    """
    Get a thread-local tokenizer instance for parallel inference
    Tokenizers are not thread-safe during encoding, so each thread gets its own copy
    """
    thread_id = threading.get_ident()

    with _shared_rebel_cache._tokenizer_lock:
        if thread_id not in _shared_rebel_cache._thread_tokenizers:
            if _shared_rebel_cache.tokenizer is None:
                return None

            # Create a deep copy of tokenizer for each thread
            # Fast tokenizers are NOT thread-safe despite being Rust-based
            # They have internal mutable state that causes "Already borrowed" errors
            try:
                from transformers import AutoTokenizer
                # Load a fresh tokenizer instance for this thread
                model_name = _shared_rebel_cache.tokenizer.name_or_path
                thread_tokenizer = AutoTokenizer.from_pretrained(model_name)
                _shared_rebel_cache._thread_tokenizers[thread_id] = thread_tokenizer
                print(f"[REBEL] Created new tokenizer instance for thread {thread_id}")
            except Exception as e:
                print(f"[REBEL] Warning: Could not create thread tokenizer: {e}")
                # Fallback to shared tokenizer (may cause "Already borrowed" errors)
                _shared_rebel_cache._thread_tokenizers[thread_id] = (
                    _shared_rebel_cache.tokenizer
                )

        return _shared_rebel_cache._thread_tokenizers[thread_id]


# ============================================================================
# Additional Optimizations
# ============================================================================

class REBELResultCache:
    """
    LRU cache for REBEL extraction results
    Thread-safe cache to avoid re-extracting the same text
    """
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, text_hash: str) -> Optional[List[Tuple[str, str, str]]]:
        with self.lock:
            if text_hash in self.cache:
                self.access_count[text_hash] += 1
                self.hits += 1
                return self.cache[text_hash]
            else:
                self.misses += 1
                return None

    def put(self, text_hash: str, result: List[Tuple[str, str, str]]):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                lru_key = min(self.access_count, key=self.access_count.get)
                del self.cache[lru_key]
                del self.access_count[lru_key]

            self.cache[text_hash] = result
            self.access_count[text_hash] = 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }


# Global result cache
_rebel_result_cache = REBELResultCache(max_size=1000)


def enable_rebel_optimizations(
    use_cache: bool = True,
    use_static_cache: bool = True,
    use_quantization: bool = False,
    quantization_dtype: str = "int8"
) -> Dict[str, bool]:
    """
    Enable various REBEL model optimizations

    Args:
        use_cache: Enable result caching (recommended)
        use_static_cache: Enable static KV cache for faster generation (PyTorch 2.0+)
        use_quantization: Enable model quantization (reduces memory, may reduce accuracy)
        quantization_dtype: Quantization dtype ('int8' or 'int4')

    Returns:
        Dict of enabled optimizations
    """
    results = {
        "cache_enabled": use_cache,
        "static_cache_enabled": False,
        "quantization_enabled": False
    }

    if not _shared_rebel_cache.model:
        print("[REBEL] Optimizations: Model not loaded yet, will apply on load")
        return results

    # Static KV cache optimization
    if use_static_cache:
        try:
            # Requires PyTorch 2.0+
            import torch
            if hasattr(torch, '__version__') and int(torch.__version__.split('.')[0]) >= 2:
                # This needs to be set before first generation
                print("[REBEL] Static KV cache optimization: Available in PyTorch 2.0+")
                print("[REBEL] Note: Apply before first inference for best results")
                results["static_cache_enabled"] = True
            else:
                print("[REBEL] Static KV cache requires PyTorch 2.0+")
        except Exception as e:
            print(f"[REBEL] Failed to enable static cache: {e}")

    # Quantization optimization
    if use_quantization and _shared_rebel_cache.model:
        try:
            print(f"[REBEL] Applying {quantization_dtype} quantization...")
            if quantization_dtype == "int8":
                # Dynamic quantization (CPU-friendly)
                import torch
                _shared_rebel_cache.model = torch.quantization.quantize_dynamic(
                    _shared_rebel_cache.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                print("[REBEL] INT8 quantization applied")
                results["quantization_enabled"] = True
            else:
                print(f"[REBEL] Quantization dtype {quantization_dtype} not yet implemented")
        except Exception as e:
            print(f"[REBEL] Failed to apply quantization: {e}")

    return results


# ============================================================================
# Module 2: Entity Relation Module
# ============================================================================

class Module2_EntityRelation:
    """
    Module 2: Entity Relation Extraction

    - Low cost: Pattern-based entity relation extraction (regex)
    - Mid cost: Use spaCy NER + dependency parsing to extract entity relations
    - High cost: Use LLM for deep entity relation extraction
    """

    COST_WEIGHTS = {
        CostLevel.LOW: 0.0,  # Module3 LOW cost is free
        CostLevel.MID: 0.5,  # Module3 MID cost will be calculated as 0.001 * output_tokens
        CostLevel.HIGH: 1.0,
    }

    def __init__(
        self,
        llm_func: Any = None,
        args: Any = None,
        use_spacy: bool = True,
        use_rebel: bool = True,
        rebel_model_name: str = "Babelscape/rebel-large",
        rebel_max_input_tokens: int = 256,
        rebel_stride: int = 64,
        rebel_num_beams: int = 1,
        rebel_num_return_sequences: int = 1,
        rebel_max_gen_length: int = 256,
    ):
        self.llm_func = llm_func
        self.args = args
        self.use_spacy = use_spacy
        self._spacy_nlp = None

        self.use_rebel = use_rebel
        self.rebel_model_name = rebel_model_name
        self.rebel_max_input_tokens = rebel_max_input_tokens
        self.rebel_stride = rebel_stride
        self.rebel_num_beams = rebel_num_beams
        self.rebel_num_return_sequences = rebel_num_return_sequences
        self.rebel_max_gen_length = rebel_max_gen_length

        self.person_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
            r'\b((?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+)\b',
        ]
        self.org_patterns = [
            r'\b([A-Z][a-z]+(?:\s+(?:Inc|Corp|Ltd|LLC|Company|Organization))?)\b',
        ]
        self.location_patterns = [
            r'\b(?:in|at|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        ]

        if use_spacy:
            try:
                import spacy  # noqa: F401
            except ImportError:
                print("[Module2] Warning: spaCy not installed, falling back to regex")
                self.use_spacy = False

        if use_rebel:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: F401
            except ImportError:
                print("[Module2] Warning: transformers not installed, REBEL disabled")
                self.use_rebel = False

    def _calculate_output_tokens(self, text_list: List[str]) -> int:
        """
        Calculate total token count for output string list

        Args:
            text_list: List of strings

        Returns:
            Total token count
        """
        return _calculate_output_tokens(text_list)

    def _get_spacy_model(self):
        """Lazy load spaCy model"""
        if self._spacy_nlp is None and self.use_spacy:
            try:
                import spacy
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("[Module2] Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        return self._spacy_nlp

    def _get_rebel_model(self):
        """Lazy load REBEL model (using shared cache, shared by Module2/3)"""
        if not self.use_rebel or not _shared_rebel_cache.enabled:
            return None, None

        # Check if model is already loaded (fast path, no lock needed)
        if _shared_rebel_cache.model is not None:
            return _shared_rebel_cache.model, get_thread_tokenizer()

        # Use load_lock only for initialization
        with _shared_rebel_cache.load_lock:
            # Double-check after acquiring lock
            if _shared_rebel_cache.model is not None:
                return _shared_rebel_cache.model, get_thread_tokenizer()

            # Call preload function instead of inline loading
            success = preload_rebel_model(
                model_name=self.rebel_model_name,
                max_concurrent_inference=4,  # Allow 4 concurrent inferences
                use_batching=False,
                device="auto"
            )

            if not success:
                self.use_rebel = False
                return None, None

        return _shared_rebel_cache.model, get_thread_tokenizer()

    def _extract_triplets_with_rebel(self, text: str, model, tokenizer) -> List[Tuple[str, str, str]]:
        """
        Extract triplets using REBEL (supports parallel inference + caching)

        Optimizations:
        - Result caching: avoid re-extracting same text
        - Thread-safe tokenizer: per-thread tokenizer instances
        - Semaphore-controlled inference: limit concurrent model calls
        """
        # Check cache first (fast path)
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cached_result = _rebel_result_cache.get(text_hash)
        if cached_result is not None:
            return cached_result

        triplets_all: List[Tuple[str, str, str]] = []

        # Tokenization is thread-safe with per-thread tokenizers
        chunks = token_sliding_chunks(
            tokenizer,
            text,
            max_input_tokens=self.rebel_max_input_tokens,
            stride=self.rebel_stride,
        )

        if not chunks:
            return []

        # Batch processing optimization: process multiple chunks together
        batch_size = min(4, len(chunks))  # Process up to 4 chunks at once

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]

            # Use semaphore to limit concurrent inference (if configured)
            semaphore = _shared_rebel_cache.inference_semaphore
            if semaphore:
                semaphore.acquire()

            try:
                # Tokenize batch (thread-safe with per-thread tokenizer)
                inputs = tokenizer(
                    batch_chunks,
                    max_length=self.rebel_max_input_tokens,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                gen_kwargs = {
                    "max_length": self.rebel_max_gen_length,
                    "length_penalty": 0,
                    "num_beams": self.rebel_num_beams,
                    "num_return_sequences": self.rebel_num_return_sequences,
                }

                # Model inference (PyTorch models are thread-safe for inference in eval mode)
                with torch_rebel.no_grad():
                    generated_tokens = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        **gen_kwargs,
                    )

                # Decode (thread-safe with per-thread tokenizer)
                decoded_list = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                for decoded in decoded_list:
                    triplets_all.extend(rebel_extract_triplets_from_decoded(decoded))

            finally:
                if semaphore:
                    semaphore.release()

        seen = set()
        uniq = []
        for s, r, o in triplets_all:
            key = (s.lower(), r.lower(), o.lower())
            if key not in seen:
                uniq.append((s, r, o))
                seen.add(key)

        # Cache the result
        _rebel_result_cache.put(text_hash, uniq)

        return uniq

    def execute(
        self,
        query: str,
        memories: List[Any],
        cost_level: int = CostLevel.MID
    ) -> Tuple[List[str], float]:
        """
        Execute entity relation mining

        Returns:
            (entity relation list, cost value)
        """
        if cost_level == CostLevel.LOW:
            result = self._low_cost_extract(query, memories)
            return result, self.COST_WEIGHTS[CostLevel.LOW]
        elif cost_level == CostLevel.MID:
            result = self._mid_cost_extract(query, memories)
            # Calculate cost as 0 * output_tokens
            output_tokens = self._calculate_output_tokens(result)
            cost = 0 * output_tokens
            return result, cost
        else:  # HIGH
            result, ret = self._high_cost_extract(query, memories)
            model_name = getattr(self.args, 'model', 'meta/llama-3.3-70b-instruct')
            cost = _calculate_token_cost(model_name, ret) if ret else self.COST_WEIGHTS[CostLevel.HIGH]
            return result, cost

    def _low_cost_extract(self, query: str, memories: List[Any]) -> List[str]:
        """
        Low cost: Enhanced pattern matching and spaCy NER for entity relations

        Optimization: Use preprocessed cache if available
        """
        # Check if all memories have preprocessed entity relations
        cached_relations = []
        has_cache = True
        for mem in memories:
            if hasattr(mem, 'entity_relations') and mem.entity_relations is not None:
                cached_relations.extend(mem.entity_relations)
            else:
                has_cache = False
                break

        if has_cache and cached_relations:
            return list(set(cached_relations))  # Deduplicate

        # Fallback to original extraction if cache not available
        nlp = self._get_spacy_model()
        if nlp is None:
            # Enhanced regex fallback with more comprehensive patterns
            relations = []
            for mem in memories:
                content = mem.content
                date_prefix = f"[{mem.date_time}] " if hasattr(mem, 'date_time') and mem.date_time else ""

                # Enhanced relation patterns covering more types
                relation_patterns = [
                    # Communication verbs
                    (r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(said|told|asked|spoke to|called|texted|emailed|messaged)\s+(?:to\s+)?(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                     'PERSON', 'PERSON', '{0} ({1}) - {2} - {3} ({4})'),
                    # Social interactions
                    (r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(met|visited|saw|encountered|joined)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                     'PERSON', 'PERSON', '{0} ({1}) - {2} - {3} ({4})'),
                    # Work relations
                    (r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(works? (?:at|for)|is employed by|joined)\s+(\b[A-Z][a-z]+(?:\s+(?:Inc|Corp|Ltd|LLC|Company|Organization))?)',
                     'PERSON', 'ORG', '{0} ({1}) - {2} - {3} ({4})'),
                    # Location relations
                    (r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(lives? in|moved to|traveled to|went to)\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                     'PERSON', 'LOC', '{0} ({1}) - {2} - {3} ({4})'),
                    # Ownership/possession
                    (r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(owns?|has|bought|purchased|acquired)\s+(.+?)(?:\.|,|from|in)',
                     'PERSON', 'OBJECT', '{0} ({1}) - {2} - {3} ({4})'),
                ]

                for pattern, subj_type, obj_type, template in relation_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 3:
                            relation = template.format(match[0].strip(), subj_type, match[1].strip(),
                                                      match[2].strip(), obj_type)
                            relations.append(f"{date_prefix}{relation}")

            return list(set(relations))

        # Enhanced spaCy-based extraction with batch processing
        relations = []
        seen_relations = set()

        # Batch process all memories with spaCy pipe for better performance
        contents = [mem.content for mem in memories]
        date_prefixes = [f"[{mem.date_time}] " if hasattr(mem, 'date_time') and mem.date_time else "" for mem in memories]

        # Use nlp.pipe for batch processing (much faster than individual calls)
        for doc, date_prefix in zip(nlp.pipe(contents, batch_size=16), date_prefixes):
            # Build token to entity mapping
            token_to_entity = {}
            for ent in doc.ents:
                for token in ent:
                    token_to_entity[token] = ent

            def get_entity_info(token):
                """Get complete entity information for a token"""
                if token in token_to_entity:
                    ent = token_to_entity[token]
                    return ent.text, ent.label_
                elif token.ent_type_:
                    return token.text, token.ent_type_
                elif token.pos_ in ["PROPN", "NOUN"]:
                    # Try to get compound nouns
                    compound = [token.text]
                    for child in token.children:
                        if child.dep_ in ["compound", "amod"]:
                            compound.insert(0, child.text)
                    return " ".join(compound), "ENTITY"
                return token.text, None

            # Extract subject-verb-object relations
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ == "VERB":
                    subject = token
                    verb = token.head
                    subject_entity, subject_label = get_entity_info(subject)

                    # Only process if subject is a named entity or proper noun
                    if not subject_label or subject_label == "UNKNOWN":
                        continue

                    # Find direct objects
                    for child in verb.children:
                        if child.dep_ in ["dobj", "attr", "pobj"]:
                            obj_entity, obj_label = get_entity_info(child)
                            if obj_label and obj_label != "UNKNOWN":
                                # Build verb phrase (include particles and prepositions)
                                verb_phrase = [verb.lemma_]
                                for v_child in verb.children:
                                    if v_child.dep_ in ["prt", "aux", "auxpass"]:
                                        verb_phrase.append(v_child.text)
                                    elif v_child.dep_ == "prep" and child in v_child.subtree:
                                        verb_phrase.append(v_child.text)

                                relation_str = f"{subject_entity} ({subject_label}) - {' '.join(verb_phrase)} - {obj_entity} ({obj_label})"
                                full_relation = f"{date_prefix}{relation_str}"

                                if relation_str.lower() not in seen_relations:
                                    relations.append(full_relation)
                                    seen_relations.add(relation_str.lower())

            # Extract prepositional phrase relations
            for token in doc:
                if token.dep_ == "prep" and token.head.pos_ in ["VERB", "NOUN"]:
                    prep = token
                    head = token.head

                    # Find prepositional object
                    pobj = None
                    for child in prep.children:
                        if child.dep_ == "pobj":
                            pobj = child
                            break

                    if pobj:
                        head_entity, head_label = get_entity_info(head)
                        pobj_entity, pobj_label = get_entity_info(pobj)

                        if head_label and pobj_label:
                            relation_str = f"{head_entity} ({head_label}) - {prep.text} - {pobj_entity} ({pobj_label})"
                            full_relation = f"{date_prefix}{relation_str}"

                            if relation_str.lower() not in seen_relations:
                                relations.append(full_relation)
                                seen_relations.add(relation_str.lower())

            # Extract possessive relations
            for token in doc:
                if token.dep_ == "poss":
                    possessor = token
                    possessed = token.head
                    possessor_entity, possessor_label = get_entity_info(possessor)
                    possessed_entity, possessed_label = get_entity_info(possessed)

                    if possessor_label:
                        relation_str = f"{possessor_entity} ({possessor_label}) - has/owns - {possessed_entity} ({possessed_label or 'OBJECT'})"
                        full_relation = f"{date_prefix}{relation_str}"

                        if relation_str.lower() not in seen_relations:
                            relations.append(full_relation)
                            seen_relations.add(relation_str.lower())

        return relations

    def _mid_cost_extract(self, query: str, memories: List[Any]) -> List[str]:
        """
        Mid cost: Enhanced REBEL model extraction with temporal context

        Optimization: Use preprocessed cache if available
        """
        # Check if all memories have preprocessed entity relations
        cached_relations = []
        has_cache = True
        for mem in memories:
            if hasattr(mem, 'entity_relations') and mem.entity_relations is not None:
                cached_relations.extend(mem.entity_relations)
            else:
                has_cache = False
                break

        if has_cache and cached_relations:
            return list(set(cached_relations))  # Deduplicate

        # Fallback to original extraction if cache not available
        model, tokenizer = self._get_rebel_model()
        if model is None or tokenizer is None:
            # REBEL not available, fallback to low-cost
            return self._low_cost_extract(query, memories)

        relations: List[str] = []
        seen: Set[str] = set()

        for mem in memories:
            content = getattr(mem, "content", "") or ""
            date_time = getattr(mem, "date_time", None)
            date_prefix = f"[{date_time}] " if date_time else ""

            # Extract triplets using REBEL
            triplets = self._extract_triplets_with_rebel(content, model, tokenizer)

            for (subj, rel, obj) in triplets:
                # Improve entity type detection
                subj_type = self._detect_entity_type(subj)
                obj_type = self._detect_entity_type(obj)

                # Format relation with enhanced information
                relation_str = f"{subj} ({subj_type}) - {rel} - {obj} ({obj_type})"
                full_relation = date_prefix + relation_str

                # Use normalized key for deduplication
                key = relation_str.lower()
                if key not in seen:
                    relations.append(full_relation)
                    seen.add(key)

        return relations

    def _detect_entity_type(self, entity: str) -> str:
        """
        Enhanced entity type detection with more accurate heuristics
        """
        entity = entity.strip()
        if not entity:
            return "UNKNOWN"

        # Check for person names (capitalized words, common titles)
        if re.match(r"^(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+", entity):
            return "PERSON"
        if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$", entity) and len(entity.split()) <= 4:
            return "PERSON"

        # Check for organizations
        org_indicators = ["Corporation", "Company", "Inc", "Ltd", "University", "Corp", "LLC",
                         "Institute", "Foundation", "Association", "Group", "Organization",
                         "Department", "Ministry", "Agency"]
        if any(indicator in entity for indicator in org_indicators):
            return "ORG"

        # Check for locations
        loc_indicators = ["City", "Country", "State", "Province", "County", "District",
                         "Street", "Avenue", "Road", "Boulevard", "Square"]
        if any(indicator in entity for indicator in loc_indicators):
            return "LOC"

        # Check for dates/times
        if re.match(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$", entity):
            return "DATE"
        if re.match(r"^\d{1,2}:\d{2}", entity):
            return "TIME"
        if any(word in entity.lower() for word in ["monday", "tuesday", "wednesday", "thursday",
                                                     "friday", "saturday", "sunday", "january",
                                                     "february", "march", "april", "may", "june",
                                                     "july", "august", "september", "october",
                                                     "november", "december"]):
            return "DATE"

        # Check for money/currency
        if re.match(r"^[\$\€\£\¥]\s*\d", entity) or re.search(r"\d+\s*(?:dollars?|euros?|pounds?|yuan)", entity, re.I):
            return "MONEY"

        # Check for percentages
        if re.search(r"\d+\s*%", entity):
            return "PERCENT"

        # Default: if starts with capital, likely a proper noun
        if entity[0].isupper():
            return "ENTITY"

        return "UNKNOWN"

    def _high_cost_extract(self, query: str, memories: List[Any]) -> Tuple[List[str], Any]:
        """
        High cost: Use LLM for deep entity relation mining
        """

        memories_text = ""
        for idx, mem in enumerate(memories):
            mem_metadata = []
            if hasattr(mem, 'date_time') and mem.date_time:
                mem_metadata.append(f'date_time="{mem.date_time}"')
            if hasattr(mem, 'session_id') and mem.session_id:
                mem_metadata.append(f'session_id="{mem.session_id}"')
            if hasattr(mem, 'dia_id') and mem.dia_id:
                mem_metadata.append(f'dia_id="{mem.dia_id}"')
            metadata_str = " " + " ".join(mem_metadata) if mem_metadata else ""
            memories_text += f"\n<memory{metadata_str}>\n{mem.content}\n</memory>"

        prompt = MODULE2_ENTITY_RELATION_PROMPT_REACT.format(query=query, memories_text=memories_text)

        task_args = [(0, prompt, self.args)]
        ret = get_llm_response(args=self.args, task_args=task_args, disable_internal_threading=True)

        if len(ret) == 0:
            return self._mid_cost_extract(query, memories), []

        _, response, _, success = ret[0]
        if not success:
            return self._mid_cost_extract(query, memories), ret

        try:
            original_response = response
            response = response.strip()

            # Preferred: extract relations inside <answer>...</answer> if present
            if "<answer>" in response and "</answer>" in response:
                start = response.find("<answer>") + len("<answer>")
                end = response.find("</answer>", start)
                if end != -1:
                    response = response[start:end].strip()

            if response.startswith("```"):
                parts = response.split("```")
                if len(parts) >= 2:
                    response = parts[1]
                    if response.startswith("json"):
                        response = response[4:]
                    response = response.strip()

            # Try to parse as JSON array first (prompt_pool format: ["relation1", ...])
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
            else:
                # Fallback: try JSON object
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                else:
                    json_str = response

            # Try to parse JSON, if fails, try to repair it
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    repaired_json = repair_json(json_str)
                    parsed = json.loads(repaired_json)
                except Exception as repair_error:
                    return [original_response.strip()] if original_response.strip() else [], ret

            # Extract response field from JSON object, or use array directly
            if isinstance(parsed, dict) and 'response' in parsed:
                relations = parsed['response']
            elif isinstance(parsed, list):
                relations = parsed
            else:
                relations = None

            if isinstance(relations, list):
                return [str(r) for r in relations], ret
        except Exception as e:
            if 'original_response' in dir() and original_response and original_response.strip():
                return [original_response.strip()], ret
            pass
            
        return self._mid_cost_extract(query, memories), ret


# ============================================================================
# Module 3: Temporal Relation Module
# ============================================================================

class Module3_TemporalRelation:
    """
    Module 3: Temporal Relation Extraction

    - Low cost: Temporal sequence sorting and association (regex)
    - Mid cost: Use spaCy NER + temporal parsing to extract temporal relations
    - High cost: Use LLM for deep temporal relation analysis
    """

    COST_WEIGHTS = {
        CostLevel.LOW: 0.0,  # Module3 LOW cost is free
        CostLevel.MID: 0.5,  # Module3 MID cost will be calculated as 0.001 * output_tokens
        CostLevel.HIGH: 1.0,
    }

    def __init__(
        self,
        llm_func: Any = None,
        args: Any = None,
        use_spacy: bool = True,
        use_rebel: bool = True,
        rebel_model_name: str = "Babelscape/rebel-large",
        rebel_max_input_tokens: int = 256,
        rebel_stride: int = 64,
        rebel_num_beams: int = 1,
        rebel_num_return_sequences: int = 1,
        rebel_max_gen_length: int = 256,
    ):
        self.llm_func = llm_func
        self.args = args
        self.use_spacy = use_spacy
        self._spacy_nlp = None

        self.use_rebel = use_rebel
        self.rebel_model_name = rebel_model_name
        self.rebel_max_input_tokens = rebel_max_input_tokens
        self.rebel_stride = rebel_stride
        self.rebel_num_beams = rebel_num_beams
        self.rebel_num_return_sequences = rebel_num_return_sequences
        self.rebel_max_gen_length = rebel_max_gen_length


        self.date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',  # MM/DD/YYYY
            r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4})\b',
        ]
        self.time_patterns = [
            r'\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b',
            r'\b((?:morning|afternoon|evening|night|noon|midnight))\b',
        ]
        self.relative_time_patterns = [
            r'\b(yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year))\b',
            r'\b(\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|later|before|after))\b',
        ]

        if use_spacy:
            try:
                import spacy  # noqa: F401
            except ImportError:
                print("[Module3] Warning: spaCy not installed, falling back to regex")
                self.use_spacy = False

        if use_rebel:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: F401
            except ImportError:
                print("[Module3] Warning: transformers not installed, REBEL disabled")
                self.use_rebel = False

    def _calculate_output_tokens(self, text_list: List[str]) -> int:
        """
        Calculate total token count for output string list

        Args:
            text_list: List of strings

        Returns:
            Total token count
        """
        return _calculate_output_tokens(text_list) 

    def _get_spacy_model(self):
        """Lazy load spaCy model"""
        if self._spacy_nlp is None and self.use_spacy:
            try:
                import spacy
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("[Module3] Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        return self._spacy_nlp

    def _get_rebel_model(self):
        """Lazy load REBEL model (using shared cache, shared by Module2/3)"""
        if not self.use_rebel or not _shared_rebel_cache.enabled:
            return None, None

        # Check if model is already loaded (fast path, no lock needed)
        if _shared_rebel_cache.model is not None:
            return _shared_rebel_cache.model, get_thread_tokenizer()

        # Use load_lock only for initialization
        with _shared_rebel_cache.load_lock:
            # Double-check after acquiring lock
            if _shared_rebel_cache.model is not None:
                return _shared_rebel_cache.model, get_thread_tokenizer()

            # Call preload function instead of inline loading
            success = preload_rebel_model(
                model_name=self.rebel_model_name,
                max_concurrent_inference=4,  # Allow 4 concurrent inferences
                use_batching=False,
                device="auto"
            )

            if not success:
                self.use_rebel = False
                return None, None

        return _shared_rebel_cache.model, get_thread_tokenizer()


    @staticmethod
    def _is_time_expression(text: str) -> bool:
        """Check if it is a temporal expression"""
        t = text.strip()
        if not t:
            return False

        if re.match(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$", t, re.I):
            return True
        if re.match(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\b", t, re.I):
            return True
        if re.match(r"^\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?$", t):
            return True
        if any(w in t.lower() for w in ["yesterday", "today", "tomorrow", "last week", "next week", "last month", "next month", "ago"]):
            return True
        return False

    @staticmethod
    def _is_temporal_relation(rel: str, obj: str) -> bool:
        """Check if it is a temporal relation"""
        time_keywords = ["time", "date", "when", "during", "at", "on", "in", "before", "after", "start", "end", "begin", "finish"]
        if any(kw in rel.lower() for kw in time_keywords):
            return True
        if Module3_TemporalRelation._is_time_expression(obj):
            return True
        return False

    def _extract_triplets_with_rebel(self, text: str, model, tokenizer) -> List[Tuple[str, str, str]]:
        """
        Extract triplets using REBEL (supports parallel inference + caching)

        Optimizations:
        - Result caching: avoid re-extracting same text
        - Thread-safe tokenizer: per-thread tokenizer instances
        - Semaphore-controlled inference: limit concurrent model calls
        """
        # Check cache first (fast path)
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cached_result = _rebel_result_cache.get(text_hash)
        if cached_result is not None:
            return cached_result

        triplets_all: List[Tuple[str, str, str]] = []

        # Tokenization is thread-safe with per-thread tokenizers
        chunks = token_sliding_chunks(
            tokenizer,
            text,
            max_input_tokens=self.rebel_max_input_tokens,
            stride=self.rebel_stride,
        )

        if not chunks:
            return []

        # Batch processing optimization: process multiple chunks together
        batch_size = min(4, len(chunks))  # Process up to 4 chunks at once

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]

            # Use semaphore to limit concurrent inference (if configured)
            semaphore = _shared_rebel_cache.inference_semaphore
            if semaphore:
                semaphore.acquire()

            try:
                # Tokenize batch (thread-safe with per-thread tokenizer)
                inputs = tokenizer(
                    batch_chunks,
                    max_length=self.rebel_max_input_tokens,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                gen_kwargs = {
                    "max_length": self.rebel_max_gen_length,
                    "length_penalty": 0,
                    "num_beams": self.rebel_num_beams,
                    "num_return_sequences": self.rebel_num_return_sequences,
                }

                # Model inference (PyTorch models are thread-safe for inference in eval mode)
                with torch_rebel.no_grad():
                    generated_tokens = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        **gen_kwargs,
                    )

                # Decode (thread-safe with per-thread tokenizer)
                decoded_list = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                for decoded in decoded_list:
                    triplets_all.extend(rebel_extract_triplets_from_decoded(decoded))

            finally:
                if semaphore:
                    semaphore.release()

        seen = set()
        uniq = []
        for s, r, o in triplets_all:
            key = (s.lower(), r.lower(), o.lower())
            if key not in seen:
                uniq.append((s, r, o))
                seen.add(key)

        # Cache the result
        _rebel_result_cache.put(text_hash, uniq)

        return uniq

    def execute(
        self,
        query: str,
        memories: List[Any],
        cost_level: int = CostLevel.MID
    ) -> Tuple[List[str], float]:
        """
        Execute temporal relation mining

        Returns:
            (temporal relation list, cost value)
        """
        if cost_level == CostLevel.LOW:
            result = self._low_cost_extract(query, memories)
            return result, self.COST_WEIGHTS[CostLevel.LOW]
        elif cost_level == CostLevel.MID:
            result = self._mid_cost_extract(query, memories)
            # Calculate cost as 0 * output_tokens
            output_tokens = self._calculate_output_tokens(result)
            cost = 0 * output_tokens
            return result, cost
        else:  # HIGH
            result, ret = self._high_cost_extract(query, memories)
            model_name = getattr(self.args, 'model', 'meta/llama-3.3-70b-instruct')
            cost = _calculate_token_cost(model_name, ret) if ret else self.COST_WEIGHTS[CostLevel.HIGH]
            return result, cost

    def _low_cost_extract(self, query: str, memories: List[Any]) -> List[str]:
        """
        Low cost: Enhanced pattern matching and spaCy NER for temporal relations

        Optimization: Use preprocessed cache if available
        """
        # Check if all memories have preprocessed temporal relations
        cached_relations = []
        has_cache = True
        for mem in memories:
            if hasattr(mem, 'temporal_relations') and mem.temporal_relations is not None:
                cached_relations.extend(mem.temporal_relations)
            else:
                has_cache = False
                break

        if has_cache and cached_relations:
            return list(set(cached_relations))  # Deduplicate

        # Fallback to original extraction if cache not available
        nlp = self._get_spacy_model()
        if nlp is None:
            # Enhanced regex fallback with more comprehensive patterns
            temporal_relations = []
            seen = set()

            for mem in memories:
                content = mem.content
                date_time = getattr(mem, 'date_time', None) or "Unknown"

                # Enhanced event-time patterns
                event_time_patterns = [
                    # Pattern: "Entity action on/at/in time"
                    (r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:on|at|in|during)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|(?:yesterday|today|tomorrow))',
                     'EVENT: {0} {1} @ {2}'),
                    # Pattern: "On/At/In time, entity action"
                    (r'(?:On|At|In|During)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|(?:yesterday|today|tomorrow)),?\s+(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([a-z]+(?:\s+[a-z]+)?)',
                     'EVENT: {1} {2} @ {0}'),
                    # Pattern: "Entity action time_expression"
                    (r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([a-z]+(?:\s+[a-z]+)?)\s+((?:yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year)))',
                     'EVENT: {0} {1} @ {2}'),
                ]

                for pattern, template in event_time_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 3:
                            event_str = template.format(match[0].strip(), match[1].strip(), match[2].strip())
                            full_event = f"[{date_time}] {event_str}"
                            if event_str.lower() not in seen:
                                temporal_relations.append(full_event)
                                seen.add(event_str.lower())

                # Extract standalone time expressions
                time_patterns = [
                    (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)', 'DATE'),
                    (r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', 'TIME'),
                    (r'\b((?:yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year)))', 'RELATIVE_TIME'),
                ]

                for pattern, label in time_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        time_str = f"{label}: {match}"
                        if time_str.lower() not in seen:
                            temporal_relations.append(f"[{date_time}] {time_str}")
                            seen.add(time_str.lower())

            return temporal_relations

        # Enhanced spaCy-based extraction with batch processing
        temporal_relations = []
        seen = set()

        # Batch process all memories with spaCy pipe for better performance
        contents = [mem.content for mem in memories]
        date_times = [getattr(mem, 'date_time', None) or "Unknown" for mem in memories]

        # Use nlp.pipe for batch processing (much faster than individual calls)
        for doc, date_time in zip(nlp.pipe(contents, batch_size=16), date_times):
            # Extract time entities with better categorization
            time_entities = []
            for ent in doc.ents:
                if ent.label_ in ["DATE", "TIME", "EVENT"]:
                    time_entities.append((ent.text, ent.label_))
                    time_str = f"{ent.label_}: {ent.text}"
                    if time_str.lower() not in seen:
                        temporal_relations.append(f"[{date_time}] {time_str}")
                        seen.add(time_str.lower())

            # Extract event-time associations using dependency parsing
            for token in doc:
                # Find temporal modifiers of verbs
                if token.ent_type_ in ["DATE", "TIME"] and token.head.pos_ == "VERB":
                    verb = token.head
                    time_expr = token.text

                    # Find subject of the verb
                    subject = None
                    for child in verb.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = child.text
                            break

                    if subject:
                        event_str = f"EVENT: {subject} {verb.lemma_} @ {time_expr}"
                        if event_str.lower() not in seen:
                            temporal_relations.append(f"[{date_time}] {event_str}")
                            seen.add(event_str.lower())
                    else:
                        event_str = f"EVENT: {verb.lemma_} @ {time_expr}"
                        if event_str.lower() not in seen:
                            temporal_relations.append(f"[{date_time}] {event_str}")
                            seen.add(event_str.lower())

                # Find temporal prepositions (on, at, in, during, before, after)
                if token.dep_ == "prep" and token.text.lower() in ["on", "at", "in", "during", "before", "after"]:
                    prep = token.text.lower()
                    head = token.head

                    # Find the prepositional object
                    time_obj = None
                    for child in token.children:
                        if child.dep_ == "pobj":
                            # Check if it's a time entity
                            if child.ent_type_ in ["DATE", "TIME"]:
                                time_obj = child.text
                            break

                    if time_obj and head.pos_ == "VERB":
                        # Find subject
                        subject = None
                        for child in head.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                subject = child.text
                                break

                        if subject:
                            event_str = f"EVENT: {subject} {head.lemma_} @ {prep} {time_obj}"
                        else:
                            event_str = f"EVENT: {head.lemma_} @ {prep} {time_obj}"

                        if event_str.lower() not in seen:
                            temporal_relations.append(f"[{date_time}] {event_str}")
                            seen.add(event_str.lower())

        return temporal_relations

    def _mid_cost_extract(self, query: str, memories: List[Any]) -> List[str]:
        """
        Mid cost: Enhanced REBEL model extraction for temporal relations

        Optimization: Use preprocessed cache if available
        """
        # Check if all memories have preprocessed temporal relations
        cached_relations = []
        has_cache = True
        for mem in memories:
            if hasattr(mem, 'temporal_relations') and mem.temporal_relations is not None:
                cached_relations.extend(mem.temporal_relations)
            else:
                has_cache = False
                break

        if has_cache and cached_relations:
            return list(set(cached_relations))  # Deduplicate

        # Fallback to original extraction if cache not available
        model, tokenizer = self._get_rebel_model()
        if model is None or tokenizer is None:
            # REBEL not available, fallback to low-cost
            return self._low_cost_extract(query, memories)

        temporal_relations: List[str] = []
        seen: Set[str] = set()

        for mem in memories:
            content = getattr(mem, "content", "") or ""
            date_time = getattr(mem, "date_time", None) or "Unknown"

            # Extract triplets using REBEL
            triplets = self._extract_triplets_with_rebel(content, model, tokenizer)
            time_entities: Set[str] = set()

            # Extract temporal relations from triplets
            for (subj, rel, obj) in triplets:
                # Check if this is a temporal relation
                if not self._is_temporal_relation(rel, obj):
                    continue

                # Identify the time expression
                time_expr = None
                if self._is_time_expression(obj):
                    time_expr = obj
                elif self._is_time_expression(rel):
                    time_expr = rel

                if time_expr:
                    time_entities.add(time_expr)

                    # Add time entity
                    time_type = self._classify_time_expression(time_expr)
                    time_str = f"{time_type}: {time_expr}"
                    if time_str.lower() not in seen:
                        temporal_relations.append(f"[{date_time}] {time_str}")
                        seen.add(time_str.lower())

                    # Add event-time relation
                    event = f"{subj} {rel}".strip()
                    event_str = f"EVENT: {event} @ {time_expr}"
                    if event_str.lower() not in seen:
                        temporal_relations.append(f"[{date_time}] {event_str}")
                        seen.add(event_str.lower())

            # Regex fallback: extract explicit time expressions from text
            time_patterns = [
                (r'\b((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))', 'DAY'),
                (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)', 'DATE'),
                (r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', 'TIME'),
                (r'\b((?:yesterday|today|tomorrow))', 'RELATIVE_DATE'),
                (r'\b(last\s+(?:week|month|year)|next\s+(?:week|month|year))', 'RELATIVE_TIME'),
            ]

            for pattern, time_type in time_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    time_expr = match.group(1)
                    if time_expr not in time_entities:
                        time_entities.add(time_expr)
                        time_str = f"{time_type}: {time_expr}"
                        if time_str.lower() not in seen:
                            temporal_relations.append(f"[{date_time}] {time_str}")
                            seen.add(time_str.lower())

        return temporal_relations

    def _classify_time_expression(self, time_expr: str) -> str:
        """
        Classify a time expression into categories
        """
        time_expr_lower = time_expr.lower()

        # Check for days of week
        if any(day in time_expr_lower for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            return "DAY"

        # Check for months
        if any(month in time_expr_lower for month in ["january", "february", "march", "april", "may", "june",
                                                        "july", "august", "september", "october", "november", "december"]):
            return "DATE"

        # Check for time of day
        if re.match(r"\d{1,2}:\d{2}", time_expr):
            return "TIME"

        # Check for relative dates
        if any(word in time_expr_lower for word in ["yesterday", "today", "tomorrow"]):
            return "RELATIVE_DATE"

        # Check for relative time periods
        if any(word in time_expr_lower for word in ["last week", "next week", "last month", "next month", "last year", "next year"]):
            return "RELATIVE_TIME"

        # Default
        return "TIME_EXPR"

    def _high_cost_extract(self, query: str, memories: List[Any]) -> Tuple[List[str], Any]:
        """
        High cost: Use LLM for deep temporal relation analysis
        """

        memories_text = ""
        for idx, mem in enumerate(memories):
            mem_metadata = []
            if hasattr(mem, 'date_time') and mem.date_time:
                mem_metadata.append(f'date_time="{mem.date_time}"')
            if hasattr(mem, 'session_id') and mem.session_id:
                mem_metadata.append(f'session_id="{mem.session_id}"')
            if hasattr(mem, 'dia_id') and mem.dia_id:
                mem_metadata.append(f'dia_id="{mem.dia_id}"')
            metadata_str = " " + " ".join(mem_metadata) if mem_metadata else ""
            memories_text += f"\n<memory{metadata_str}>\n{mem.content}\n</memory>"

        prompt = MODULE3_TEMPORAL_RELATION_PROMPT_REACT.format(query=query, memories_text=memories_text)

        task_args = [(0, prompt, self.args)]
        ret = get_llm_response(args=self.args, task_args=task_args, disable_internal_threading=True)
        if len(ret) == 0:
            return self._mid_cost_extract(query, memories), []

        _, response, _, success = ret[0]
        if not success:
            return self._mid_cost_extract(query, memories), ret

        try:
            original_response = response
            response = response.strip()

            # Preferred: extract relations inside <answer>...</answer> if present
            if "<answer>" in response and "</answer>" in response:
                start = response.find("<answer>") + len("<answer>")
                end = response.find("</answer>", start)
                if end != -1:
                    response = response[start:end].strip()

            if response.startswith("```"):
                parts = response.split("```")
                if len(parts) >= 2:
                    response = parts[1]
                    if response.startswith("json"):
                        response = response[4:]
                    response = response.strip()

            # Try to parse as JSON array first (prompt_pool format: ["temporal_fact1", ...])
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
            else:
                # Fallback: try JSON object
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                else:
                    json_str = response

            # Try to parse JSON, if fails, try to repair it
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    repaired_json = repair_json(json_str)
                    parsed = json.loads(repaired_json)
                except Exception as repair_error:
                    return [original_response.strip()] if original_response.strip() else [], ret

            # Extract response field from JSON object, or use array directly
            if isinstance(parsed, dict) and 'response' in parsed:
                relations = parsed['response']
            elif isinstance(parsed, list):
                relations = parsed
            else:
                relations = None

            if isinstance(relations, list):
                return [str(r) for r in relations], ret
        except Exception as e:
            if 'original_response' in dir() and original_response and original_response.strip():
                return [original_response.strip()], ret
            pass

        return self._mid_cost_extract(query, memories), ret


# ============================================================================
# Module 4: Summary Module
# ============================================================================

class Module4_Summary:
    """
    Module 4: Summary Module

    Summarize based on question and retrieved memories, guide how to use memory information to solve problems

    - Low cost: Simple concatenation summary
    - Mid cost: Structured summary based on key information
    - High cost: Use LLM for deep summarization and reasoning
    """

    COST_WEIGHTS = {
        CostLevel.LOW: 0.0,  # Module4 LOW cost is free (no LLM or other operations)
        CostLevel.MID: 0.0,  # Module4 MID cost is free (no LLM or other operations)
        CostLevel.HIGH: 1.0,
    }

    def __init__(
        self,
        llm_func: Any = None,
        args: Any = None
    ):
        self.llm_func = llm_func
        self.args = args

    def execute(
        self,
        query: str,
        memories: List[Any],
        entity_relations: List[str] = None,
        temporal_relations: List[str] = None,
        topic_relations: List[str] = None,
        cost_level: int = CostLevel.MID
    ) -> Tuple[str, float]:
        """
        Execute summarization

        Args:
            query: Query/question
            memories: Retrieved memories
            entity_relations: Output from Module 2
            temporal_relations: Output from Module 3
            topic_relations: Output from Module 5
            cost_level: Cost level

        Returns:
            (Summary text, cost value)
        """
        if cost_level == CostLevel.LOW:
            result = self._low_cost_summary(query, memories, entity_relations, temporal_relations, topic_relations)
            return result, self.COST_WEIGHTS[CostLevel.LOW]
        elif cost_level == CostLevel.MID:
            result = self._mid_cost_summary(query, memories, entity_relations, temporal_relations, topic_relations)
            return result, self.COST_WEIGHTS[CostLevel.MID]
        else:  # HIGH
            result, ret = self._high_cost_summary(query, memories, entity_relations, temporal_relations, topic_relations)
            model_name = getattr(self.args, 'model', 'meta/llama-3.3-70b-instruct')
            cost = _calculate_token_cost(model_name, ret) if ret else self.COST_WEIGHTS[CostLevel.HIGH]
            return result, cost

    def _low_cost_summary(
        self,
        query: str,
        memories: List[Any],
        entity_relations: List[str] = None,
        temporal_relations: List[str] = None,
        topic_relations: List[str] = None
    ) -> str:
        """
        Low cost: Simple concatenation summary
        """
        summary_parts = [f"Query: {query}", "Relevant memories:"]

        for idx, mem in enumerate(memories):
            date_prefix = f"[{mem.date_time}] " if hasattr(mem, 'date_time') and mem.date_time else ""
            content_preview = mem.content
            summary_parts.append(f"{idx + 1}. {date_prefix}{content_preview}")

        # Add entity relations (simple concatenation)
        if entity_relations:
            summary_parts.append("\nEntity Relations:")
            for rel in entity_relations:
                summary_parts.append(f"- {rel}")

        # Add temporal relations (simple concatenation)
        if temporal_relations:
            summary_parts.append("\nTemporal Relations:")
            for temp in temporal_relations:
                summary_parts.append(f"- {temp}")

        # Add topic relations (simple concatenation)
        if topic_relations:
            summary_parts.append("\nTopic Relations:")
            for topic in topic_relations:
                summary_parts.append(f"- {topic}")

        return "\n".join(summary_parts)

    def _mid_cost_summary(
        self,
        query: str,
        memories: List[Any],
        entity_relations: List[str] = None,
        temporal_relations: List[str] = None,
        topic_relations: List[str] = None
    ) -> str:
        """
        Mid cost: Structured summary based on key information
        """
        summary_parts = []

        # Question section
        summary_parts.append(f"=== Question ===\n{query}")

        # Relevant memories
        summary_parts.append("\n=== Relevant Memories ===")
        for idx, mem in enumerate(memories):
            date_prefix = f"[{mem.date_time}] " if hasattr(mem, 'date_time') and mem.date_time else ""
            summary_parts.append(f"{idx + 1}. {date_prefix}{mem.content}")

        # Entity relations
        if entity_relations:
            summary_parts.append("\n=== Key Entities & Relations ===")
            for rel in entity_relations:
                summary_parts.append(f"- {rel}")

        # Temporal relations
        if temporal_relations:
            summary_parts.append("\n=== Timeline ===")
            for temp in temporal_relations:
                summary_parts.append(f"- {temp}")

        # Topic relations
        if topic_relations:
            summary_parts.append("\n=== Topic Relations ===")
            for topic in topic_relations:
                summary_parts.append(f"- {topic}")

        # Simple reasoning guidance
        summary_parts.append("\n=== Suggested Approach ===")
        summary_parts.append("Use the above information to answer the question by:")
        summary_parts.append("1. Identifying relevant entities mentioned")
        summary_parts.append("2. Following the timeline of events")
        summary_parts.append("3. Connecting relationships between entities")
        summary_parts.append("4. Following the conversation topics and transitions")

        return "\n".join(summary_parts)

    def _high_cost_summary(
        self,
        query: str,
        memories: List[Any],
        entity_relations: List[str] = None,
        temporal_relations: List[str] = None,
        topic_relations: List[str] = None
    ) -> Tuple[str, Any]:
        """
        High cost: Use LLM for deep summarization and reasoning
        """
        # Build memories text
        memories_text = ""
        for idx, mem in enumerate(memories):
            mem_metadata = []
            if hasattr(mem, 'date_time') and mem.date_time:
                mem_metadata.append(f'date_time="{mem.date_time}"')
            if hasattr(mem, 'session_id') and mem.session_id:
                mem_metadata.append(f'session_id="{mem.session_id}"')
            if hasattr(mem, 'dia_id') and mem.dia_id:
                mem_metadata.append(f'dia_id="{mem.dia_id}"')
            metadata_str = " " + " ".join(mem_metadata) if mem_metadata else ""
            memories_text += f"\n<memory{metadata_str}>\n{mem.content}\n</memory>"

        # Build entity relations text
        entity_text = ""
        if entity_relations:
            entity_text = "\n".join([f"<entity>{rel}</entity>" for rel in entity_relations])

        # Build temporal relations text
        temporal_text = ""
        if temporal_relations:
            temporal_text = "\n".join([f"<temporal>{temp}</temporal>" for temp in temporal_relations])

        # Build topic relations text
        topic_text = ""
        if topic_relations:
            topic_text = "\n".join([f"<topic>{topic}</topic>" for topic in topic_relations])

        prompt = MODULE4_SUMMARY_PROMPT_REACT.format(query=query, entity_text=entity_text, temporal_text=temporal_text, topic_text=topic_text)

        task_args = [(0, prompt, self.args)]
        # Only one prompt, disable internal ThreadPool to avoid conflicts with outer parallelism
        ret = get_llm_response(args=self.args, task_args=task_args, disable_internal_threading=True)

        if len(ret) == 0:
            return self._mid_cost_summary(query, memories, entity_relations, temporal_relations, topic_relations), []

        _, response, _, success = ret[0]
        if not success or not response:
            return self._mid_cost_summary(query, memories, entity_relations, temporal_relations, topic_relations), ret

        # Preferred: extract text inside <answer>...</answer> if present
        response = response.strip()
        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>", start)
            if end != -1:
                return response[start:end].strip(), ret

        # Try to extract JSON object and get response field (backward compatible)
        try:
            response = response.strip()
            if response.startswith("```"):
                parts = response.split("```")
                if len(parts) >= 2:
                    response = parts[1]
                    if response.startswith("json"):
                        response = response[4:]
                    response = response.strip()

            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
            else:
                json_str = response

            # Try to parse JSON, if fails, try to repair it
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    repaired_json = repair_json(json_str)
                    parsed = json.loads(repaired_json)
                except Exception:
                    # If repair fails, return full response
                    return response.strip(), ret

            # Extract response field from JSON object
            if isinstance(parsed, dict) and 'response' in parsed:
                return str(parsed['response']), ret
        except Exception:
            pass  # Fall through to return full response

        # If JSON parsing fails or no JSON found, return full response
        return response.strip(), ret


# ============================================================================
# Module 5: Topic Relation Module
# ============================================================================

class Module5_TopicRelation:
    """
    Module 5: Topic Relation Extraction

    - Low cost: Enhanced pattern matching and spaCy for topic relations
    - Mid cost: Use REBEL model extraction for topic relations
    - High cost: Use LLM for deep topic relation analysis
    """

    COST_WEIGHTS = {
        CostLevel.LOW: 0.0,  # Module5 LOW cost is free
        CostLevel.MID: 0.5,  # Module5 MID cost will be calculated as 0.001 * output_tokens
        CostLevel.HIGH: 1.0,
    }

    def __init__(
        self,
        llm_func: Any = None,
        args: Any = None,
        use_spacy: bool = True,
        use_rebel: bool = True,
        rebel_model_name: str = "Babelscape/rebel-large",
        rebel_max_input_tokens: int = 256,
        rebel_stride: int = 64,
        rebel_num_beams: int = 1,
        rebel_num_return_sequences: int = 1,
        rebel_max_gen_length: int = 256,
    ):
        self.llm_func = llm_func
        self.args = args
        self.use_spacy = use_spacy
        self._spacy_nlp = None

        self.use_rebel = use_rebel
        self.rebel_model_name = rebel_model_name
        self.rebel_max_input_tokens = rebel_max_input_tokens
        self.rebel_stride = rebel_stride
        self.rebel_num_beams = rebel_num_beams
        self.rebel_num_return_sequences = rebel_num_return_sequences
        self.rebel_max_gen_length = rebel_max_gen_length

        if use_spacy:
            try:
                import spacy  # noqa: F401
            except ImportError:
                print("[Module5] Warning: spaCy not installed, falling back to regex")
                self.use_spacy = False

        if use_rebel:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: F401
            except ImportError:
                print("[Module5] Warning: transformers not installed, REBEL disabled")
                self.use_rebel = False

    def _calculate_output_tokens(self, text_list: List[str]) -> int:
        """
        Calculate total token count for output string list

        Args:
            text_list: List of strings

        Returns:
            Total token count
        """
        return _calculate_output_tokens(text_list)

    def _get_spacy_model(self):
        """Lazy load spaCy model"""
        if self._spacy_nlp is None and self.use_spacy:
            try:
                import spacy
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("[Module5] Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        return self._spacy_nlp

    def _get_rebel_model(self):
        """Lazy load REBEL model (using shared cache, shared by Module2/3/5)"""
        if not self.use_rebel or not _shared_rebel_cache.enabled:
            return None, None

        # Check if model is already loaded (fast path, no lock needed)
        if _shared_rebel_cache.model is not None:
            return _shared_rebel_cache.model, get_thread_tokenizer()

        # Use load_lock only for initialization
        with _shared_rebel_cache.load_lock:
            # Double-check after acquiring lock
            if _shared_rebel_cache.model is not None:
                return _shared_rebel_cache.model, get_thread_tokenizer()

            # Call preload function instead of inline loading
            success = preload_rebel_model(
                model_name=self.rebel_model_name,
                max_concurrent_inference=4,  # Allow 4 concurrent inferences
                use_batching=False,
                device="auto"
            )

            if not success:
                self.use_rebel = False
                return None, None

        return _shared_rebel_cache.model, get_thread_tokenizer()

    def _extract_triplets_with_rebel(self, text: str, model, tokenizer) -> List[Tuple[str, str, str]]:
        """
        Extract triplets using REBEL (supports parallel inference + caching)

        Optimizations:
        - Result caching: avoid re-extracting same text
        - Thread-safe tokenizer: per-thread tokenizer instances
        - Semaphore-controlled inference: limit concurrent model calls
        """
        # Check cache first (fast path)
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cached_result = _rebel_result_cache.get(text_hash)
        if cached_result is not None:
            return cached_result

        triplets_all: List[Tuple[str, str, str]] = []

        # Tokenization is thread-safe with per-thread tokenizers
        chunks = token_sliding_chunks(
            tokenizer,
            text,
            max_input_tokens=self.rebel_max_input_tokens,
            stride=self.rebel_stride,
        )

        if not chunks:
            return []

        # Batch processing optimization: process multiple chunks together
        batch_size = min(4, len(chunks))  # Process up to 4 chunks at once

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]

            # Use semaphore to limit concurrent inference (if configured)
            semaphore = _shared_rebel_cache.inference_semaphore
            if semaphore:
                semaphore.acquire()

            try:
                # Tokenize batch (thread-safe with per-thread tokenizer)
                inputs = tokenizer(
                    batch_chunks,
                    max_length=self.rebel_max_input_tokens,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                gen_kwargs = {
                    "max_length": self.rebel_max_gen_length,
                    "length_penalty": 0,
                    "num_beams": self.rebel_num_beams,
                    "num_return_sequences": self.rebel_num_return_sequences,
                }

                # Model inference (PyTorch models are thread-safe for inference in eval mode)
                with torch_rebel.no_grad():
                    generated_tokens = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        **gen_kwargs,
                    )

                # Decode (thread-safe with per-thread tokenizer)
                decoded_list = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                for decoded in decoded_list:
                    triplets_all.extend(rebel_extract_triplets_from_decoded(decoded))

            finally:
                if semaphore:
                    semaphore.release()

        seen = set()
        uniq = []
        for s, r, o in triplets_all:
            key = (s.lower(), r.lower(), o.lower())
            if key not in seen:
                uniq.append((s, r, o))
                seen.add(key)

        # Cache the result
        _rebel_result_cache.put(text_hash, uniq)

        return uniq

    @staticmethod
    def _is_topic_relation(rel: str, obj: str) -> bool:
        """Check if it is a topic relation"""
        topic_keywords = ["about", "regarding", "concerning", "topic", "subject", "discuss", "mention", "talk"]
        if any(kw in rel.lower() for kw in topic_keywords):
            return True
        return False

    def execute(
        self,
        query: str,
        memories: List[Any],
        cost_level: int = CostLevel.MID
    ) -> Tuple[List[str], float]:
        """
        Execute topic relation mining

        Returns:
            (topic relation list, cost value)
        """
        if cost_level == CostLevel.LOW:
            result = self._low_cost_extract(query, memories)
            return result, self.COST_WEIGHTS[CostLevel.LOW]
        elif cost_level == CostLevel.MID:
            result = self._mid_cost_extract(query, memories)
            output_tokens = self._calculate_output_tokens(result)
            cost = 0 * output_tokens
            return result, cost
        else:  # HIGH
            result, ret = self._high_cost_extract(query, memories)
            model_name = getattr(self.args, 'model', 'meta/llama-3.3-70b-instruct')
            cost = _calculate_token_cost(model_name, ret) if ret else self.COST_WEIGHTS[CostLevel.HIGH]
            return result, cost

    def _low_cost_extract(self, query: str, memories: List[Any]) -> List[str]:
        """
        Low cost: Enhanced pattern matching and spaCy for topic relations

        Optimization: Use preprocessed cache if available
        """
        # Check if all memories have preprocessed topic relations
        cached_relations = []
        has_cache = True
        for mem in memories:
            if hasattr(mem, 'topic_relations') and mem.topic_relations is not None:
                cached_relations.extend(mem.topic_relations)
            else:
                has_cache = False
                break

        if has_cache and cached_relations:
            return list(set(cached_relations))  # Deduplicate

        # Fallback to original extraction if cache not available
        nlp = self._get_spacy_model()
        if nlp is None:
            # Enhanced regex fallback with more comprehensive patterns
            topic_relations = []
            seen = set()

            # Extract potential topics using keyword patterns
            topic_patterns = [
                r'\b(?:about|regarding|concerning|on|talking about)\s+([A-Z][a-z]+(?:\s+[a-z]+)*)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was|is|are|were)\s+(?:discussed|mentioned|talked about)\b',
                r'\b(?:the topic of|subject of)\s+([A-Z][a-z]+(?:\s+[a-z]+)*)\b',
            ]

            for mem in memories:
                content = mem.content
                date_time = getattr(mem, 'date_time', None) or "Unknown"
                date_prefix = f"[{date_time}] "

                for pattern in topic_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        topic = match.strip()
                        if len(topic.split()) >= 1:  # At least one word topic
                            topic_str = f"Topic discussed: {topic}"
                            full_topic = date_prefix + topic_str
                            if topic_str.lower() not in seen:
                                topic_relations.append(full_topic)
                                seen.add(topic_str.lower())

            # Extract topic transitions
            for mem in memories:
                content = mem.content
                date_time = getattr(mem, 'date_time', None) or "Unknown"
                date_prefix = f"[{date_time}] "

                # Look for topic change indicators
                transition_patterns = [
                    r'\b(?:then|after that|later|next|moving on to|switching to|changing to)\s+(?:the topic of\s+)?([A-Z][a-z]+(?:\s+[a-z]+)*)\b',
                    r'\b(?:from|about)\s+([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:to|and then to)\s+([A-Z][a-z]+(?:\s+[a-z]+)*)\b',
                ]

                for pattern in transition_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            topic_str = f"Topic transition: {match[0]} → {match[1]}"
                        else:
                            topic_str = f"Topic transition to: {match}"
                        full_topic = date_prefix + topic_str
                        if topic_str.lower() not in seen:
                            topic_relations.append(full_topic)
                            seen.add(topic_str.lower())

            return topic_relations

        # Enhanced spaCy-based extraction with batch processing
        topic_relations = []
        seen = set()

        # Batch process all memories with spaCy pipe for better performance
        contents = [mem.content for mem in memories]
        date_times = [getattr(mem, 'date_time', None) or "Unknown" for mem in memories]

        # Use nlp.pipe for batch processing (much faster than individual calls)
        for doc, date_time in zip(nlp.pipe(contents, batch_size=16), date_times):
            date_prefix = f"[{date_time}] "

            # Extract noun phrases as potential topics
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 1 and chunk.text[0].isupper():
                    topic_str = f"Topic discussed: {chunk.text}"
                    full_topic = date_prefix + topic_str
                    if topic_str.lower() not in seen:
                        topic_relations.append(full_topic)
                        seen.add(topic_str.lower())

            # Find topic relationships through discourse markers
            for token in doc:
                if token.lower_ in ['about', 'regarding', 'concerning', 'on', 'talking']:
                    # Find following noun phrase
                    for child in token.subtree:
                        if child.pos_ == 'NOUN' or child.dep_ == 'pobj':
                            topic = child.text
                            topic_str = f"Topic discussed: {topic}"
                            full_topic = date_prefix + topic_str
                            if topic_str.lower() not in seen:
                                topic_relations.append(full_topic)
                                seen.add(topic_str.lower())
                            break

                # Topic transitions
                elif token.lower_ in ['then', 'after', 'later', 'next', 'moving', 'switching', 'changing']:
                    # Look for topic after transition
                    for sibling in token.head.children:
                        if sibling.pos_ == 'NOUN' or sibling.dep_ == 'pobj':
                            topic = sibling.text
                            topic_str = f"Topic transition to: {topic}"
                            full_topic = date_prefix + topic_str
                            if topic_str.lower() not in seen:
                                topic_relations.append(full_topic)
                                seen.add(topic_str.lower())
                            break

        return topic_relations

    def _mid_cost_extract(self, query: str, memories: List[Any]) -> List[str]:
        """
        Mid cost: Enhanced REBEL model extraction for topic relations

        Optimization: Use preprocessed cache if available
        """
        # Check if all memories have preprocessed topic relations
        cached_relations = []
        has_cache = True
        for mem in memories:
            if hasattr(mem, 'topic_relations') and mem.topic_relations is not None:
                cached_relations.extend(mem.topic_relations)
            else:
                has_cache = False
                break

        if has_cache and cached_relations:
            return list(set(cached_relations))  # Deduplicate

        # Fallback to original extraction if cache not available
        model, tokenizer = self._get_rebel_model()
        if model is None or tokenizer is None:
            # REBEL not available, fallback to low-cost
            return self._low_cost_extract(query, memories)

        topic_relations: List[str] = []
        seen: Set[str] = set()

        for mem in memories:
            content = getattr(mem, "content", "") or ""
            date_time = getattr(mem, "date_time", None) or "Unknown"
            date_prefix = f"[{date_time}] "

            # Extract triplets using REBEL
            triplets = self._extract_triplets_with_rebel(content, model, tokenizer)

            # Extract topic relations from triplets
            for (subj, rel, obj) in triplets:
                # Check if this is a topic relation
                if self._is_topic_relation(rel, obj):
                    topic_str = f"Topic discussed: {subj} - {rel} - {obj}"
                    full_topic = date_prefix + topic_str
                    if topic_str.lower() not in seen:
                        topic_relations.append(full_topic)
                        seen.add(topic_str.lower())
                else:
                    # General relation can also be considered as topic context
                    topic_str = f"Topic context: {subj} - {rel} - {obj}"
                    full_topic = date_prefix + topic_str
                    if topic_str.lower() not in seen:
                        topic_relations.append(full_topic)
                        seen.add(topic_str.lower())

        return topic_relations

    def _high_cost_extract(self, query: str, memories: List[Any]) -> Tuple[List[str], Any]:
        """
        High cost: Use LLM for deep topic relationship analysis
        """
        # Build memories text
        memories_text = ""
        for idx, mem in enumerate(memories):
            mem_metadata = []
            if hasattr(mem, 'date_time') and mem.date_time:
                mem_metadata.append(f'date_time="{mem.date_time}"')
            if hasattr(mem, 'session_id') and mem.session_id:
                mem_metadata.append(f'session_id="{mem.session_id}"')
            if hasattr(mem, 'dia_id') and mem.dia_id:
                mem_metadata.append(f'dia_id="{mem.dia_id}"')
            metadata_str = " " + " ".join(mem_metadata) if mem_metadata else ""
            memories_text += f"\n<memory{metadata_str}>\n{mem.content}\n</memory>"

        prompt = MODULE5_TOPIC_RELATION_PROMPT_REACT.format(query=query, memories_text=memories_text)

        task_args = [(0, prompt, self.args)]
        # Only one prompt, disable internal ThreadPool to avoid conflicts with outer parallelism
        ret = get_llm_response(args=self.args, task_args=task_args, disable_internal_threading=True)

        if len(ret) == 0:
            return self._mid_cost_extract(query, memories), []

        _, response, _, success = ret[0]
        if not success:
            return self._mid_cost_extract(query, memories), ret

        # Parse response
        try:
            original_response = response
            response = response.strip()

            # Preferred: extract relations inside <answer>...</answer> if present
            if "<answer>" in response and "</answer>" in response:
                start = response.find("<answer>") + len("<answer>")
                end = response.find("</answer>", start)
                if end != -1:
                    response = response[start:end].strip()

            if response.startswith("```"):
                parts = response.split("```")
                if len(parts) >= 2:
                    response = parts[1]
                    if response.startswith("json"):
                        response = response[4:]
                    response = response.strip()

            # Try to parse as JSON array first
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
            else:
                # Fallback: try JSON object
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                else:
                    json_str = response

            # Try to parse JSON, if fails, try to repair it
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    repaired_json = repair_json(json_str)
                    parsed = json.loads(repaired_json)
                except Exception as repair_error:
                    return [original_response.strip()] if original_response.strip() else [], ret

            # Extract response field from JSON object, or use array directly
            if isinstance(parsed, dict) and 'response' in parsed:
                relations = parsed['response']
            elif isinstance(parsed, list):
                relations = parsed
            else:
                relations = None

            if isinstance(relations, list):
                return [str(r) for r in relations], ret
        except Exception as e:
            if 'original_response' in dir() and original_response and original_response.strip():
                return [original_response.strip()], ret
            pass

        return self._mid_cost_extract(query, memories), ret


# ============================================================================
# Pipeline Executor with Cost Tracking
# ============================================================================

class ModularPipelineExecutor:
    """
    Modular Pipeline Executor

    Execution flow:
    1. General Pipeline (5 modules with RL-based cost selection)
    2. Module4_Summary (produces final output, integrates everything)
    """

    def __init__(
        self,
        module1: Module1_Filter,
        module2: Module2_EntityRelation,
        module3: Module3_TemporalRelation,
        module5: Module5_TopicRelation,
        module4: Module4_Summary,
        actor_critic: Any,  # ActorCriticNetwork
        device: torch.device,
        encoder: Any = None  # For text to embedding conversion
    ):
        # General Modules
        self.module1 = module1
        self.module2 = module2
        self.module3 = module3
        self.module5 = module5
        self.module4 = module4
        self.actor_critic = actor_critic
        self.device = device
        # encoder can be obtained from module1 or passed separately
        self.encoder = encoder if encoder is not None else getattr(module1, 'encoder', None)

        # Action counter
        self.action_counts = {
            'module1_low': 0, 'module1_mid': 0, 'module1_high': 0,
            'module2_low': 0, 'module2_mid': 0, 'module2_high': 0,
            'module3_low': 0, 'module3_mid': 0, 'module3_high': 0,
            'module5_low': 0, 'module5_mid': 0, 'module5_high': 0,
            'module4_low': 0, 'module4_mid': 0, 'module4_high': 0,
        }

        # Total cost tracking
        self.total_cost = 0.0

    def _aggregate_embeddings(self, memories: List[Any], dim: int = 768) -> torch.Tensor:
        """Aggregate memory embeddings into a single embedding (average pooling)"""
        if len(memories) > 0:
            embs = [m.embedding for m in memories if m.embedding is not None]
            if embs:
                embs_np = np.stack(embs)
                return torch.from_numpy(embs_np).float().mean(dim=0).to(self.device)
        return torch.zeros(dim).to(self.device)

    def _text_list_to_embedding(self, texts: List[str], dim: int = 768) -> torch.Tensor:
        """Convert text list to aggregated embedding"""
        if not texts or len(texts) == 0:
            return torch.zeros(dim).to(self.device)
        
        combined_text = "\n".join(texts)
        
        if self.encoder is not None:
            try:
                from rag_utils import get_data_embeddings
                embeddings = get_data_embeddings(self.encoder, [combined_text])
                if embeddings is not None and len(embeddings) > 0:
                    return torch.from_numpy(embeddings[0]).float().to(self.device)
            except Exception as e:
                print(f"[_text_list_to_embedding] Error: {e}")
        
        return torch.zeros(dim).to(self.device)

    def execute(
        self,
        query: str,
        query_emb: torch.Tensor,
        initial_memories: List[Any],
        actions: Optional[Tuple[int, int, int, int, int]] = None,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline

        Execution order:
        a) General Pipeline (5 modules with RL control)
           - Module1: query + initial_memories → filtered_memories
           - Module2: query + filtered_memories → entity_relations
           - Module3: query + filtered_memories → temporal_relations
           - Module5: query + filtered_memories → topic_relations
           - Module4: query + entity/temporal/topic embeddings → summary

        Returns:
            Dict with all outputs
        """
        self.actor_critic.eval()
        query_emb = query_emb.to(self.device)
        query_emb_batch = query_emb.unsqueeze(0)

        total_cost = 0.0
        log_probs_float = []

        initial_memory_emb = self._aggregate_embeddings(initial_memories).unsqueeze(0)  # [1, 768]

        if actions is None:
            with torch.no_grad():
                m1_action_t, m1_lp, _, state_value = self.actor_critic.get_module1_action(
                    query_emb_batch, initial_memory_emb, deterministic=deterministic
                )
                m1_action = m1_action_t.item()
                log_probs_float.append(m1_lp.item())
                state_value_float = state_value.item()
        else:
            m1_action = actions[0]
            log_probs_float.append(0.0)
            state_value_float = 0.0

        self._update_action_count('module1', m1_action)
        filtered_memories, cost1 = self.module1.execute(
            query, initial_memories,
            query_emb.cpu().numpy() if isinstance(query_emb, torch.Tensor) else query_emb,
            cost_level=m1_action
        )
        total_cost += cost1

        filtered_memory_emb = self._aggregate_embeddings(filtered_memories).unsqueeze(0)  # [1, 768]

        if actions is None:
            with torch.no_grad():
                m2_action_t, m2_lp, _ = self.actor_critic.get_module2_action(
                    query_emb_batch, filtered_memory_emb, deterministic=deterministic
                )
                m3_action_t, m3_lp, _ = self.actor_critic.get_module3_action(
                    query_emb_batch, filtered_memory_emb, deterministic=deterministic
                )
                m5_action_t, m5_lp, _ = self.actor_critic.get_module5_action(
                    query_emb_batch, filtered_memory_emb, deterministic=deterministic
                )
                m2_action, m3_action, m5_action = m2_action_t.item(), m3_action_t.item(), m5_action_t.item()
                log_probs_float.extend([m2_lp.item(), m3_lp.item(), m5_lp.item()])
        else:
            m2_action, m3_action, m5_action = actions[1], actions[2], actions[3]
            log_probs_float.extend([0.0, 0.0, 0.0])

        self._update_action_count('module2', m2_action)
        entity_relations, cost2 = self.module2.execute(query, filtered_memories, cost_level=m2_action)
        total_cost += cost2

        self._update_action_count('module3', m3_action)
        temporal_relations, cost3 = self.module3.execute(query, filtered_memories, cost_level=m3_action)
        total_cost += cost3

        self._update_action_count('module5', m5_action)
        topic_relations, cost5 = self.module5.execute(query, filtered_memories, cost_level=m5_action)
        total_cost += cost5

        entity_emb = self._text_list_to_embedding(entity_relations).unsqueeze(0)  # [1, 768]
        temporal_emb = self._text_list_to_embedding(temporal_relations).unsqueeze(0)  # [1, 768]
        topic_emb = self._text_list_to_embedding(topic_relations).unsqueeze(0)  # [1, 768]

        if len(entity_relations) == 0 and len(temporal_relations) == 0 and len(topic_relations) == 0:
            aggregated_memory_emb = filtered_memory_emb
        else:
            # Average pool entity, temporal, and topic embeddings
            valid_embs = []
            if len(entity_relations) > 0:
                valid_embs.append(entity_emb)
            if len(temporal_relations) > 0:
                valid_embs.append(temporal_emb)
            if len(topic_relations) > 0:
                valid_embs.append(topic_emb)

            if valid_embs:
                aggregated_memory_emb = torch.mean(torch.stack(valid_embs), dim=0)
            else:
                aggregated_memory_emb = filtered_memory_emb

        if actions is None:
            with torch.no_grad():
                m4_action_t, m4_lp, _ = self.actor_critic.get_module4_action(
                    query_emb_batch, aggregated_memory_emb, deterministic=deterministic
                )
                m4_action = m4_action_t.item()
                log_probs_float.append(m4_lp.item())
        else:
            m4_action = actions[4]
            log_probs_float.append(0.0)

        # ========== Step 6: Module4 FINAL (Integrates everything) ==========
        # Module4 produces final summary output
        self._update_action_count('module4', m4_action)
        summary, cost4 = self.module4.execute(
            query, filtered_memories,
            entity_relations=entity_relations,
            temporal_relations=temporal_relations,
            topic_relations=topic_relations,
            cost_level=m4_action
        )
        total_cost += cost4
        self.total_cost += total_cost

        return {
            # General Pipeline outputs
            'filtered_memories': filtered_memories,
            'entity_relations': entity_relations,
            'temporal_relations': temporal_relations,
            'topic_relations': topic_relations,
            'summary': summary,
            'actions': (m1_action, m2_action, m3_action, m5_action, m4_action),
            'log_probs': log_probs_float,
            'state_value': state_value_float,
            'total_cost': total_cost,
            'individual_costs': (cost1, cost2, cost3, cost5, cost4),
            'initial_memory_emb': initial_memory_emb.squeeze(0).cpu(),
            'filtered_memory_emb': filtered_memory_emb.squeeze(0).cpu(),
            'entity_emb': entity_emb.squeeze(0).cpu(),
            'temporal_emb': temporal_emb.squeeze(0).cpu(),
            'topic_emb': topic_emb.squeeze(0).cpu(),
            'aggregated_memory_emb': aggregated_memory_emb.squeeze(0).cpu(),  # CRITICAL: for Module4
        }

    def _update_action_count(self, module: str, action: int):
        """Update action count"""
        level = ['low', 'mid', 'high'][action]
        key = f'{module}_{level}'
        self.action_counts[key] += 1
