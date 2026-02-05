# model_tier.py
# Cost-Performance Balance Strategy: Model Size Tiers
#
# This version uses different LLM sizes with the same prompt for cost tiers:
# - LOW (0.2): Small LLM (Qwen/Qwen2.5-0.5B-Instruct)
# - MID (0.5): Medium LLM (Qwen/Qwen2.5-3B-Instruct)
# - HIGH (1.0): Large LLM (Qwen/Qwen2.5-7B-Instruct)
#
# All use the same prompt, but different model sizes

import torch
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from json_repair import repair_json
from ..utils.llm_utils import get_llm_response
from ..utils.llm_pricing import calculate_cost, normalize_cost
from ..prompts.prompt_pool import (
    MODULE1_FILTER_PROMPT_Direct,
    MODULE2_ENTITY_RELATION_PROMPT_Direct,
    MODULE3_TEMPORAL_RELATION_PROMPT_Direct,
    MODULE4_SUMMARY_PROMPT_Direct,
    MODULE1_FILTER_PROMPT_COT,
    MODULE1_FILTER_PROMPT_REACT,
    MODULE2_ENTITY_RELATION_PROMPT_COT,
    MODULE2_ENTITY_RELATION_PROMPT_REACT,
    MODULE3_TEMPORAL_RELATION_PROMPT_COT,
    MODULE3_TEMPORAL_RELATION_PROMPT_REACT,
    MODULE4_SUMMARY_PROMPT_COT,
    MODULE4_SUMMARY_PROMPT_REACT,
    MODULE5_TOPIC_RELATION_PROMPT_Direct,
    MODULE5_TOPIC_RELATION_PROMPT_COT,
    MODULE5_TOPIC_RELATION_PROMPT_REACT,
)


# ============================================================================
# Cost Level Enum
# ============================================================================

class CostLevel:
    LOW = 0      # Small LLM (fastest, lowest cost)
    MID = 1      # Medium LLM (balanced)
    HIGH = 2     # Large LLM (slowest, highest quality)


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
# Model Size Selection Helper
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


def get_model_for_cost_level(args, cost_level: int) -> str:
    """
    Get the appropriate model name based on cost level

    Args:
        args: Configuration args with model settings
        cost_level: Cost level (LOW/MID/HIGH)

    Returns:
        Model name to use
    """
    if cost_level == CostLevel.LOW:
        return getattr(args, 'small_model', 'meta/llama-3.2-3b-instruct')
    elif cost_level == CostLevel.MID:
        return getattr(args, 'medium_model', 'meta/llama-3.1-8b-instruct')
    elif cost_level == CostLevel.HIGH:
        return getattr(args, 'large_model', 'meta/llama-3.3-70b-instruct')
    else:  # Default to HIGH
        return getattr(args, 'large_model', 'meta/llama-3.3-70b-instruct')


# ============================================================================
# Module 1: Filter Module
# ============================================================================

class Module1_Filter:
    """
    Module 1: Filter Module

    Uses different sized LLMs with the same prompt:
    - Low cost: Small LLM
    - Mid cost: Medium LLM
    - High cost: Large LLM
    """

    # Cost weights deprecated - now using token-based cost calculation

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

    def _get_args_with_model(self, cost_level: int):
        """Get a modified args object with the appropriate model for the cost level"""
        import copy
        modified_args = copy.copy(self.args)
        modified_args.model = get_model_for_cost_level(self.args, cost_level)
        return modified_args

    def execute(
        self,
        query: str,
        memories: List[Any],
        query_emb: Optional[np.ndarray] = None,
        cost_level: int = CostLevel.MID
    ) -> Tuple[List[Any], float]:
        """Execute filtering operation"""
        result, ret = self._filter_with_llm(query, memories, cost_level)
        model_name = get_model_for_cost_level(self.args, cost_level)
        cost = _calculate_token_cost(model_name, ret) if ret else 0.0
        return result, cost

    def _filter_with_llm(self, query: str, memories: List[Any], cost_level: int) -> Tuple[List[Any], Any]:
        """Use LLM to filter memories - same prompt, different model sizes"""
        if len(memories) == 0:
            return [], []

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

        # Use appropriate model based on cost level
        modified_args = self._get_args_with_model(cost_level)
        task_args = [(0, prompt, modified_args)]
        ret = get_llm_response(args=modified_args, task_args=task_args, disable_internal_threading=True)
        return self._parse_scores(ret, memories), ret

    def _parse_scores(self, ret, memories: List[Any]) -> List[Any]:
        """Parse LLM response to extract scores"""
        if len(ret) == 0:
            return memories[:self.top_k]

        _, response, _, success = ret[0]
        if not success:
            return memories[:self.top_k]

        try:
            original_response = response
            response = response.strip()

            # Check if response is empty
            if not response:
                print(f"[Module1_Filter._parse_scores] Empty response, returning top_k memories")
                return memories[:self.top_k]

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

            # Check if json_str is empty before parsing
            if not json_str or not json_str.strip():
                # Try to extract comma-separated numbers from entire response
                numbers = re.findall(r'\b\d+\b', original_response)
                if numbers and len(numbers) >= len(memories):
                    try:
                        scores = [int(n) for n in numbers[:len(memories)]]
                        scored_memories = []
                        for idx, mem in enumerate(memories[:len(scores)]):
                            score = float(scores[idx]) / 10.0 if idx < len(scores) else 0.0
                            mem.relevance_score = score
                            scored_memories.append((mem, score))
                        scored_memories.sort(key=lambda x: x[1], reverse=True)
                        return [m for m, s in scored_memories[:self.top_k]]
                    except Exception:
                        pass
                print(f"[Module1_Filter._parse_scores] Empty JSON string extracted. Original response (first 500 chars): {original_response[:500]}")
                return memories[:self.top_k]

            # Try to parse JSON, if fails, try to extract numbers
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as json_error:
                # Try to extract comma-separated numbers (e.g., "9, 6, 6, 9, 10")
                # First try from json_str
                numbers = re.findall(r'\b\d+\b', json_str)
                if numbers and len(numbers) >= len(memories):
                    try:
                        scores = [int(n) for n in numbers[:len(memories)]]
                        scored_memories = []
                        for idx, mem in enumerate(memories[:len(scores)]):
                            score = float(scores[idx]) / 10.0 if idx < len(scores) else 0.0
                            mem.relevance_score = score
                            scored_memories.append((mem, score))
                        scored_memories.sort(key=lambda x: x[1], reverse=True)
                        return [m for m, s in scored_memories[:self.top_k]]
                    except Exception:
                        pass
                
                # Try JSON repair
                try:
                    repaired_json = repair_json(json_str)
                    parsed = json.loads(repaired_json)
                except Exception as repair_error:
                    # Last resort: extract numbers from entire response
                    numbers = re.findall(r'\b\d+\b', original_response)
                    if numbers and len(numbers) >= len(memories):
                        try:
                            scores = [int(n) for n in numbers[:len(memories)]]
                            scored_memories = []
                            for idx, mem in enumerate(memories[:len(scores)]):
                                score = float(scores[idx]) / 10.0 if idx < len(scores) else 0.0
                                mem.relevance_score = score
                                scored_memories.append((mem, score))
                            scored_memories.sort(key=lambda x: x[1], reverse=True)
                            return [m for m, s in scored_memories[:self.top_k]]
                        except Exception:
                            pass
                    print(f"[Module1_Filter._parse_scores] JSON repair failed: {repair_error}")
                    print(f"[Module1_Filter._parse_scores] Original response (first 500 chars): {original_response[:500]}")
                    print(f"[Module1_Filter._parse_scores] Extracted JSON string (first 200 chars): {json_str[:200]}")
                    return memories[:self.top_k]

            # Extract response field from JSON object, or use array directly
            if isinstance(parsed, dict) and 'response' in parsed:
                scores = parsed['response']
            elif isinstance(parsed, list):
                scores = parsed
            else:
                scores = None

            if not isinstance(scores, list):
                return memories[:self.top_k]

            scored_memories = []
            for idx, mem in enumerate(memories[:len(scores)]):
                if idx < len(scores):
                    score_value = scores[idx]
                    # Handle both number and dict formats
                    if isinstance(score_value, dict):
                        # Extract score from dict (e.g., {"score": 8})
                        score = float(score_value.get('score', 0)) / 10.0
                    elif isinstance(score_value, (int, float)):
                        score = float(score_value) / 10.0
                    else:
                        score = 0.0
                else:
                    score = 0.0
                mem.relevance_score = score
                scored_memories.append((mem, score))

            scored_memories.sort(key=lambda x: x[1], reverse=True)
            return [m for m, s in scored_memories[:self.top_k]]
        except Exception as e:
            print(f"[Module1_Filter._parse_scores] JSON parsing failed: {e}, returning top_k memories")
            return memories[:self.top_k]


# ============================================================================
# Module 2: Entity Relation Module
# ============================================================================

class Module2_EntityRelation:
    """
    Module 2: Entity Relation Extraction

    Uses different sized LLMs with the same prompt:
    - Low cost: Small LLM
    - Mid cost: Medium LLM
    - High cost: Large LLM
    """

    # Cost weights deprecated - now using token-based cost calculation

    def __init__(
        self,
        llm_func: Any = None,
        args: Any = None
    ):
        self.llm_func = llm_func
        self.args = args

    def _get_args_with_model(self, cost_level: int):
        """Get a modified args object with the appropriate model for the cost level"""
        import copy
        modified_args = copy.copy(self.args)
        modified_args.model = get_model_for_cost_level(self.args, cost_level)
        return modified_args

    def execute(
        self,
        query: str,
        memories: List[Any],
        cost_level: int = CostLevel.MID
    ) -> Tuple[List[str], float]:
        """Execute entity relation extraction"""
        result, ret = self._extract_with_llm(query, memories, cost_level)
        model_name = get_model_for_cost_level(self.args, cost_level)
        cost = _calculate_token_cost(model_name, ret) if ret else 0.0
        return result, cost

    def _extract_with_llm(self, query: str, memories: List[Any], cost_level: int) -> Tuple[List[str], Any]:
        """Extract entities - same prompt, different model sizes"""
        if len(memories) == 0:
            return [], []

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

        # Use appropriate model based on cost level
        modified_args = self._get_args_with_model(cost_level)
        task_args = [(0, prompt, modified_args)]
        ret = get_llm_response(args=modified_args, task_args=task_args, disable_internal_threading=True)

        return self._parse_relations(ret), ret

    def _parse_relations(self, ret) -> List[str]:
        """Parse LLM response"""
        if len(ret) == 0:
            return []

        _, response, _, success = ret[0]
        if not success:
            return []

        try:
            original_response = response
            response = response.strip()

            # Check for empty response
            if not response:
                print(f"[Module2_EntityRelation._parse_relations] Empty response, returning empty list")
                return []

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

            # Check for empty json_str
            if not json_str or not json_str.strip():
                print(f"[Module2_EntityRelation._parse_relations] Empty JSON string extracted. Original response (first 500 chars): {original_response}")
                return []

            # Try to parse JSON, if fails, try to repair it
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as json_error:
                try:
                    repaired_json = repair_json(json_str)
                    parsed = json.loads(repaired_json)
                except Exception as repair_error:
                    print(f"[Module2_EntityRelation._parse_relations] JSON repair failed.")
                    return [original_response.strip()] if original_response.strip() else []

            # Extract response field from JSON object, or use array directly
            if isinstance(parsed, dict) and 'response' in parsed:
                relations = parsed['response']
            elif isinstance(parsed, list):
                relations = parsed
            else:
                relations = None

            if isinstance(relations, list):
                return [str(r) for r in relations]
        except Exception as e:
            if 'original_response' in dir() and original_response and original_response.strip():
                return [original_response.strip()]
            return []


# ============================================================================
# Module 3: Temporal Relation Module
# ============================================================================

class Module3_TemporalRelation:
    """
    Module 3: Temporal Relation Extraction

    Uses different sized LLMs with the same prompt:
    - Low cost: Small LLM
    - Mid cost: Medium LLM
    - High cost: Large LLM
    """

    # Cost weights deprecated - now using token-based cost calculation

    def __init__(
        self,
        llm_func: Any = None,
        args: Any = None
    ):
        self.llm_func = llm_func
        self.args = args

    def _get_args_with_model(self, cost_level: int):
        """Get a modified args object with the appropriate model for the cost level"""
        import copy
        modified_args = copy.copy(self.args)
        modified_args.model = get_model_for_cost_level(self.args, cost_level)
        return modified_args

    def execute(
        self,
        query: str,
        memories: List[Any],
        cost_level: int = CostLevel.MID
    ) -> Tuple[List[str], float]:
        """Execute temporal relation extraction"""
        result, ret = self._extract_with_llm(query, memories, cost_level)
        model_name = get_model_for_cost_level(self.args, cost_level)
        cost = _calculate_token_cost(model_name, ret) if ret else 0.0
        return result, cost

    def _extract_with_llm(self, query: str, memories: List[Any], cost_level: int) -> Tuple[List[str], Any]:
        """Extract temporal info - same prompt, different model sizes"""
        if len(memories) == 0:
            return [], []

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

        # Use appropriate model based on cost level
        modified_args = self._get_args_with_model(cost_level)
        task_args = [(0, prompt, modified_args)]
        ret = get_llm_response(args=modified_args, task_args=task_args, disable_internal_threading=True)

        return self._parse_relations(ret), ret

    def _parse_relations(self, ret) -> List[str]:
        """Parse LLM response"""
        if len(ret) == 0:
            return []

        _, response, _, success = ret[0]
        if not success:
            return []

        try:
            original_response = response
            response = response.strip()

            # Check for empty response
            if not response:
                print(f"[Module3_TemporalRelation._parse_relations] Empty response, returning empty list")
                return []

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

            # Check for empty json_str
            if not json_str or not json_str.strip():
                print(f"[Module3_TemporalRelation._parse_relations] Empty JSON string extracted. Original response (first 500 chars): {original_response[:500] if original_response else 'EMPTY'}")
                return []

            # Try to parse JSON, if fails, try to repair it
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as json_error:
                try:
                    repaired_json = repair_json(json_str)
                    parsed = json.loads(repaired_json)
                except Exception as repair_error:
                    print(f"[Module2_EntityRelation._parse_relations] JSON repair failed.")
                    return [original_response.strip()] if original_response.strip() else []

            # Extract response field from JSON object, or use array directly
            if isinstance(parsed, dict) and 'response' in parsed:
                relations = parsed['response']
            elif isinstance(parsed, list):
                relations = parsed
            else:
                relations = None

            if isinstance(relations, list):
                return [str(r) for r in relations]
        except Exception as e:
            if 'original_response' in dir() and original_response and original_response.strip():
                return [original_response.strip()]
            return []


# ============================================================================
# Module 4: Summary Module
# ============================================================================

class Module4_Summary:
    """
    Module 4: Summary Module

    Uses different sized LLMs with the same prompt:
    - Low cost: Small LLM
    - Mid cost: Medium LLM
    - High cost: Large LLM
    """

    # Cost weights deprecated - now using token-based cost calculation

    def __init__(
        self,
        llm_func: Any = None,
        args: Any = None
    ):
        self.llm_func = llm_func
        self.args = args

    def _get_args_with_model(self, cost_level: int):
        """Get a modified args object with the appropriate model for the cost level"""
        import copy
        modified_args = copy.copy(self.args)
        modified_args.model = get_model_for_cost_level(self.args, cost_level)
        return modified_args

    def execute(
        self,
        query: str,
        memories: List[Any],
        entity_relations: List[str] = None,
        temporal_relations: List[str] = None,
        topic_relations: List[str] = None,
        cost_level: int = CostLevel.MID
    ) -> Tuple[str, float]:
        """Execute summarization"""
        result, ret = self._summarize_with_llm(query, memories, entity_relations, temporal_relations, topic_relations, cost_level)
        model_name = get_model_for_cost_level(self.args, cost_level)
        cost = _calculate_token_cost(model_name, ret) if ret else 0.0
        return result, cost

    def _summarize_with_llm(
        self,
        query: str,
        memories: List[Any],
        entity_relations: List[str],
        temporal_relations: List[str],
        topic_relations: List[str] = None,
        cost_level: int = CostLevel.MID
    ) -> Tuple[str, Any]:
        """Summarize - same prompt, different model sizes"""
        if len(memories) == 0:
            return "", []

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

        # Use appropriate model based on cost level
        modified_args = self._get_args_with_model(cost_level)
        task_args = [(0, prompt, modified_args)]
        ret = get_llm_response(args=modified_args, task_args=task_args, disable_internal_threading=True)

        return self._parse_summary(ret), ret

    def _parse_summary(self, ret) -> str:
        """Parse LLM response"""
        if len(ret) == 0:
            return ""

        _, response, _, success = ret[0]
        if not success or not response:
            return ""

        # Preferred: extract text inside <answer>...</answer> if present
        response = response.strip()
        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>", start)
            if end != -1:
                return response[start:end].strip()

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
                    return response.strip()

            # Extract response field from JSON object
            if isinstance(parsed, dict) and 'response' in parsed:
                return str(parsed['response'])
        except Exception:
            pass  # Fall through to return full response

        # If JSON parsing fails or no JSON found, return full response
        return response.strip()


# ============================================================================
# Module 5: Topic Relation Module
# ============================================================================

class Module5_TopicRelation:
    """
    Module 5: Topic Relation Extraction

    Uses different sized LLMs with the same prompt:
    - Low cost: Small LLM
    - Mid cost: Medium LLM
    - High cost: Large LLM
    """

    # Cost weights deprecated - now using token-based cost calculation

    def __init__(
        self,
        llm_func: Any = None,
        args: Any = None
    ):
        self.llm_func = llm_func
        self.args = args

    def _get_args_with_model(self, cost_level: int):
        """Get a modified args object with the appropriate model for the cost level"""
        import copy
        modified_args = copy.copy(self.args)
        modified_args.model = get_model_for_cost_level(self.args, cost_level)
        return modified_args

    def execute(
        self,
        query: str,
        memories: List[Any],
        cost_level: int = CostLevel.MID
    ) -> Tuple[List[str], float]:
        """Execute topic relation extraction"""

        result, ret = self._extract_with_llm(query, memories, cost_level)
        model_name = get_model_for_cost_level(self.args, cost_level)
        cost = _calculate_token_cost(model_name, ret) if ret else 0.0
        return result, cost

    def _extract_with_llm(self, query: str, memories: List[Any], cost_level: int) -> Tuple[List[str], Any]:
        """Extract topic relations - same prompt, different model sizes"""
        if len(memories) == 0:
            return [], []

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

        # Use appropriate model based on cost level
        modified_args = self._get_args_with_model(cost_level)
        task_args = [(0, prompt, modified_args)]
        ret = get_llm_response(args=modified_args, task_args=task_args, disable_internal_threading=True)

        return self._parse_relations(ret), ret

    def _parse_relations(self, ret) -> List[str]:
        """Parse LLM response"""
        if len(ret) == 0:
            return []

        _, response, _, success = ret[0]
        if not success:
            return []

        try:
            original_response = response
            response = response.strip()

            # Check for empty response
            if not response:
                print(f"[Module5_TopicRelation._parse_relations] Empty response, returning empty list")
                return []

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

            # Check for empty json_str
            if not json_str or not json_str.strip():
                print(f"[Module5_TopicRelation._parse_relations] Empty JSON string extracted. Original response (first 500 chars): {original_response[:500] if original_response else 'EMPTY'}")
                return []

            # Try to parse JSON, if fails, try to repair it
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as json_error:
                try:
                    repaired_json = repair_json(json_str)
                    parsed = json.loads(repaired_json)
                except Exception as repair_error:
                    print(f"[Module5_TopicRelation._parse_relations] JSON repair failed.")      
                    return [original_response.strip()] if original_response.strip() else []

            # Extract response field from JSON object, or use array directly
            if isinstance(parsed, dict) and 'response' in parsed:
                relations = parsed['response']
            elif isinstance(parsed, list):
                relations = parsed
            else:
                relations = None

            if isinstance(relations, list):
                return [str(r) for r in relations]
        except Exception as e:
            if 'original_response' in dir() and original_response and original_response.strip():
                return [original_response.strip()]
            return []


# ============================================================================
# Pipeline Executor with Cost Tracking
# ============================================================================

class ModularPipelineExecutor:
    """
    Modular Pipeline Executor

    Execution flow:
    1. General Pipeline (4 modules with RL-based cost selection)
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

        initial_memory_emb = self._aggregate_embeddings(initial_memories).unsqueeze(0)

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

        filtered_memory_emb = self._aggregate_embeddings(filtered_memories).unsqueeze(0)

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

        entity_emb = self._text_list_to_embedding(entity_relations).unsqueeze(0)
        temporal_emb = self._text_list_to_embedding(temporal_relations).unsqueeze(0)
        topic_emb = self._text_list_to_embedding(topic_relations).unsqueeze(0)

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
            # Return intermediate embeddings for PPO training
            # CRITICAL: These embeddings MUST match what was used in action selection
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
