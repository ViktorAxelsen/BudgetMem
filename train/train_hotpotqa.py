import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import pickle
import copy
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import openai
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import wandb

from src.utils.llm_utils import get_tokenizer, MAX_CONTEXT_LENGTH, get_llm_response, get_llm_response_via_api, get_llm_api_stats, reset_llm_api_stats
from src.utils.rag_utils import get_embeddings, get_embeddings_with_model, build_faiss_index, faiss_knn_search, init_context_model, init_query_model, init_data_embedding_model, get_data_embeddings
from src.utils.llm_pricing import (
    normalize_costs_batch,
    align_reward_cost_scales,
    align_reward_cost_scales_batch,
    add_to_scale_alignment_history,
    get_scale_alignment_stats,
    reset_scale_alignment_history
)
from src.utils.eval_utils import *
from src.prompts.prompt_pool import *
from src.config import get_locomo_args
from src.trainer.ppo_trainer import PPOTrainer, ActorCriticNetwork, Experience, ExperienceBuffer
# nv_api = "tgp_v1__YwYN06yIUcp51rJBckBA-54y_f8jawFJc25U7Hx9AI"
# nv_base = "https://api.together.xyz/v1"
def load_modules(cost_strategy='rule_llm'):
    """
    Dynamically load modules based on cost-performance strategy

    Args:
        cost_strategy: One of 'rule_llm', 'prompt_tier', 'model_tier'

    Returns:
        Tuple of (CostLevel, ModuleOutput, MemoryModuleOutputs,
                  Module1_Filter, Module2_EntityRelation, Module3_TemporalRelation,
                  Module5_TopicRelation, Module4_Summary, ModularPipelineExecutor)
    """
    if cost_strategy == 'rule_llm':
        from src.modules.rule_llm import (
            CostLevel, ModuleOutput, MemoryModuleOutputs,
            Module1_Filter, Module2_EntityRelation, Module3_TemporalRelation, Module5_TopicRelation, Module4_Summary,
            ModularPipelineExecutor
        )
    elif cost_strategy == 'prompt_tier':
        from src.modules.prompt_tier import (
            CostLevel, ModuleOutput, MemoryModuleOutputs,
            Module1_Filter, Module2_EntityRelation, Module3_TemporalRelation, Module5_TopicRelation, Module4_Summary,
            ModularPipelineExecutor
        )
    elif cost_strategy == 'model_tier':
        from src.modules.model_tier import (
            CostLevel, ModuleOutput, MemoryModuleOutputs,
            Module1_Filter, Module2_EntityRelation, Module3_TemporalRelation, Module5_TopicRelation, Module4_Summary,
            ModularPipelineExecutor
        )
    else:
        raise ValueError(f"Unknown cost strategy: {cost_strategy}")

    return (CostLevel, ModuleOutput, MemoryModuleOutputs,
            Module1_Filter, Module2_EntityRelation, Module3_TemporalRelation, Module5_TopicRelation, Module4_Summary,
            ModularPipelineExecutor)



random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


QUESTION_LOG_PATH = os.environ.get(
    "QUESTION_LOG_PATH",
    os.path.join("./logs", f"question_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
)


def log_question_trace(payload: Dict):
    os.makedirs(os.path.dirname(QUESTION_LOG_PATH) or ".", exist_ok=True)
    with open(QUESTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

@dataclass
class QuestionTrackingRecord:
    """Tracking record for a single question"""
    sample_id: str
    question_index: int
    question: str
    ground_truth: str
    prediction: str
    reward: float
    epoch: int  # Epoch number during training, -1 during testing
    retrieved_memories: List[str]  # Retrieved memory content list
    filtered_memories: List[str]  # Filtered memory content list
    entity_relations: List[str]  # Module 2 output
    temporal_relations: List[str]  # Module 3 output
    topic_relations: List[str]  # Module 5 output
    summary: str  # Module 4 output
    final_prompt: str  # Final generated prompt
    actions: Tuple[int, int, int, int, int]  # (m1, m2, m3, m5, m4) actions

class QuestionTracker:
    """Tracks historical prediction records for questions"""
    def __init__(self):
        # key: (sample_id, question_index), value: List[QuestionTrackingRecord]
        self.records: Dict[Tuple[str, int], List[QuestionTrackingRecord]] = {}
    
    def add_record(self, record: QuestionTrackingRecord):
        """Add a record"""
        key = (record.sample_id, record.question_index)
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(record)
    
    def get_failed_questions(self, min_failures: int = 2) -> List[Tuple[str, int, List[QuestionTrackingRecord]]]:
        """
        Get questions with consecutive failures exceeding the specified threshold
        
        Args:
            min_failures: Minimum number of consecutive failures (default 2)
        
        Returns:
            List of (sample_id, question_index, records) tuples
        """
        failed_questions = []
        for (sample_id, q_idx), records in self.records.items():
            # Sort by epoch
            sorted_records = sorted(records, key=lambda r: r.epoch)
            
            # Check for consecutive failures
            consecutive_failures = 0
            max_consecutive = 0
            for record in sorted_records:
                if record.reward < 1.0:  # reward < 1.0 indicates prediction error
                    consecutive_failures += 1
                    max_consecutive = max(max_consecutive, consecutive_failures)
                else:
                    consecutive_failures = 0
            
            # If consecutive failures exceed threshold, add to results
            if max_consecutive >= min_failures:
                failed_questions.append((sample_id, q_idx, sorted_records))
        
        return failed_questions
    
    def save_failed_questions_report(self, output_file: str, min_failures: int = 2):
        """Save detailed report of failed questions"""
        failed_questions = self.get_failed_questions(min_failures)
        
        report = {
            'total_failed_questions': int(len(failed_questions)),
            'min_failures_threshold': int(min_failures),
            'failed_questions': []
        }
        
        for sample_id, q_idx, records in failed_questions:
            # Calculate consecutive failures
            consecutive_failures = 0
            max_consecutive = 0
            for record in records:
                if record.reward < 1.0:
                    consecutive_failures += 1
                    max_consecutive = max(max_consecutive, consecutive_failures)
                else:
                    consecutive_failures = 0
            
            question_data = {
                'sample_id': str(sample_id),
                'question_index': int(q_idx),
                'question': str(records[0].question) if records else "",
                'ground_truth': str(records[0].ground_truth) if records else "",
                'max_consecutive_failures': int(max_consecutive),
                'total_attempts': int(len(records)),
                'total_failures': int(sum(1 for r in records if r.reward < 1.0)),
                'records': []
            }
            
            # Add detailed information for each record
            for record in records:
                # Ensure all values are Python native types (avoid JSON serialization errors from numpy types)
                reward_value = float(record.reward) if hasattr(record.reward, 'item') else float(record.reward)
                is_correct_value = bool(reward_value >= 1.0)
                
                record_data = {
                    'epoch': int(record.epoch),
                    'prediction': str(record.prediction),
                    'reward': reward_value,
                    'is_correct': is_correct_value,
                    'actions': {
                        'module1': str(['low', 'mid', 'high'][record.actions[0]]),
                        'module2': str(['low', 'mid', 'high'][record.actions[1]]),
                        'module3': str(['low', 'mid', 'high'][record.actions[2]]),
                        'module5': str(['low', 'mid', 'high'][record.actions[3]]),  
                        'module4': str(['low', 'mid', 'high'][record.actions[4]])
                    },
                    'retrieved_memories': [str(m) for m in record.retrieved_memories],
                    'filtered_memories': [str(m) for m in record.filtered_memories],
                    'entity_relations': [str(r) for r in record.entity_relations] if record.entity_relations else [],
                    'temporal_relations': [str(r) for r in record.temporal_relations] if record.temporal_relations else [],
                    'summary': str(record.summary) if record.summary else '',
                    'final_prompt': str(record.final_prompt)
                }
                question_data['records'].append(record_data)
            
            report['failed_questions'].append(question_data)
        
        # Save to file
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report

global_question_tracker = QuestionTracker()

@dataclass
class MemoryItem:
    """
    Single memory item - enhanced version

    Supports storing outputs from each module for richer memory content during retrieval
    """
    content: str  # Memory text content (entire session content)
    embedding: Optional[np.ndarray] = None  # Vector representation of memory
    session_id: int = 0  # Session ID
    date_time: str = ""  # Date and time
    processed: bool = False  # Whether processed by modules
    is_original: bool = True  # Whether original session (not pipeline-generated enhanced memory)
    original_dialogs: Optional[List[str]] = field(default=None)  # Original dialog list

    module_outputs: Optional["MemoryModuleOutputs"] = None

    # Convenient access to module outputs
    filter_output: Optional[str] = None          # Module 1: Filtered content
    entity_relations: Optional[List[str]] = None  # Module 2: Entity relation list
    temporal_relations: Optional[List[str]] = None  # Module 3: Temporal relation list
    topic_relations: Optional[List[str]] = None     # Module 5: Topic relation list
    summary: Optional[str] = None                 # Module 4: Summary content

    # Cost tracking
    total_cost: float = 0.0  # Total cost for processing this memory
    cost_breakdown: Optional[Dict[str, float]] = None  # Cost breakdown by module

    def get_module_output(self, module_name: str) -> Optional[str]:
        """Get output from specified module"""
        if module_name == 'filter':
            return self.filter_output
        elif module_name == 'entity_relation':
            return "\n".join(self.entity_relations) if self.entity_relations else None
        elif module_name == 'temporal_relation':
            return "\n".join(self.temporal_relations) if self.temporal_relations else None
        elif module_name == 'topic_relation':
            return "\n".join(self.topic_relations) if self.topic_relations else None
        elif module_name == 'summary':
            return self.summary
        return None

    def get_enriched_content(self) -> str:
        """
        Get enriched memory content (includes relations and summary)
        
        Used to provide richer memory information during retrieval, especially for summary_memory
        """
        parts = []
        
        # Original content
        if self.date_time:
            parts.append(f"[Date: {self.date_time}]")
        parts.append(f"[Original Content]\n{self.content}")
        
        # Entity relations
        if self.entity_relations:
            parts.append(f"\n[Key Entities]")
            parts.extend(self.entity_relations[:10])  # Limit quantity to avoid excessive length
        
        # Temporal relations
        if self.temporal_relations:
            parts.append(f"\n[Timeline]")
            parts.extend(self.temporal_relations[:10])  # Limit quantity to avoid excessive length

        # Topic relations
        if self.topic_relations:
            parts.append(f"\n[Topic Relations]")
            parts.extend(self.topic_relations[:10])  # Limit quantity to avoid excessive length

        # Summary
        if self.summary:
            parts.append(f"\n[Summary]\n{self.summary}")
        
        return "\n".join(parts)


def build_summary_memory_content(
    entity_relations: Optional[List[str]],
    temporal_relations: Optional[List[str]],
    summary: str
) -> str:
    """Build summary memory content including relations, timeline, and summary text."""
    parts = []
    if entity_relations:
        parts.append("[Entities]")
        parts.extend(entity_relations)
    if temporal_relations:
        parts.append("[Timeline]")
        parts.extend(temporal_relations)
    if summary:
        parts.append("[Summary]")
        parts.append(summary)
    return "\n".join(parts).strip()


class GlobalMemoryPool:
    """
    Global memory pool - enhanced version

    Supports retrieval by module outputs for richer memory content
    """
    def __init__(self):
        self.memories: List[MemoryItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self._valid_indices: List[int] = []  # Mapping from FAISS index to memories index

    def __len__(self):
        """Return number of memories in the pool"""
        return len(self.memories)

    def add_memory(self, memory: MemoryItem):
        """Add memory to global pool"""
        self.memories.append(memory)

    def update_memory(self, session_id: int, new_content: str, new_embedding: np.ndarray):
        """Update memory content and embedding for specified session"""
        for memory in self.memories:
            if memory.session_id == session_id:
                memory.content = new_content
                memory.embedding = new_embedding
                memory.processed = True
                break

    def update_memory_with_modules(
        self,
        session_id: int,
        entity_relations: List[str] = None,
        temporal_relations: List[str] = None,
        summary: str = None,
        total_cost: float = 0.0,
        cost_breakdown: Dict[str, float] = None
    ):
        """
        Update module outputs for specified session

        Args:
            session_id: Session ID
            entity_relations: Entity relation list
            temporal_relations: Temporal relation list
            summary: Summary content
            total_cost: Total cost
            cost_breakdown: Cost breakdown
        """
        for memory in self.memories:
            if memory.session_id == session_id:
                if entity_relations:
                    memory.entity_relations = entity_relations
                if temporal_relations:
                    memory.temporal_relations = temporal_relations
                if summary:
                    memory.summary = summary
                memory.total_cost = total_cost
                memory.cost_breakdown = cost_breakdown
                memory.processed = True
                break

    def build_index(self):
        """Build FAISS index"""
        if len(self.memories) > 0:
            # Collect valid embeddings and their indices in memories
            embeddings_list = []
            valid_indices = []

            for i, m in enumerate(self.memories):
                if m.embedding is not None:
                    embeddings_list.append(m.embedding)
                    valid_indices.append(i)

            if embeddings_list:
                self.embeddings = np.stack(embeddings_list)
                self._faiss_index = build_faiss_index(self.embeddings, metric='ip')
                self._valid_indices = valid_indices
                print(f"[GlobalMemoryPool] Built FAISS index with {len(embeddings_list)} valid embeddings "
                      f"out of {len(self.memories)} total memories")
            else:
                print(f"[GlobalMemoryPool] Warning: No valid embeddings found in {len(self.memories)} memories")

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MemoryItem]:
        """Retrieve relevant memories from global pool"""
        if self._faiss_index is None or len(self.memories) == 0:
            return []

        if not self._valid_indices:
            print("[GlobalMemoryPool] Warning: No valid indices, returning empty list")
            return []

        # Limit top_k to not exceed actual retrievable quantity
        actual_top_k = min(top_k, len(self._valid_indices))

        q = query_embedding.reshape(1, -1).astype('float32')
        _, indices = faiss_knn_search(self._faiss_index, q, top_k=actual_top_k, metric='ip')

        # Use mapping to convert FAISS index to memories index and filter out -1
        retrieved_memories = []
        for faiss_idx in indices[0]:
            # Check if FAISS index is valid
            if faiss_idx < 0 or faiss_idx >= len(self._valid_indices):
                if faiss_idx != -1:  # -1 is FAISS normal return value (indicates no result), no warning needed
                    print(f"[GlobalMemoryPool] Warning: Invalid FAISS index {faiss_idx} (valid range: 0-{len(self._valid_indices)-1}), skipping")
                continue
            
            # Get actual index in memories
            memory_idx = self._valid_indices[faiss_idx]
            
            # Check again if memory_idx is within valid range (prevent index out of bounds)
            if memory_idx < 0 or memory_idx >= len(self.memories):
                print(f"[GlobalMemoryPool] Warning: Invalid memory index {memory_idx} (valid range: 0-{len(self.memories)-1}), skipping")
                continue
            
            retrieved_memories.append(self.memories[memory_idx])

        return retrieved_memories

    def retrieve_with_enriched_content(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        include_modules: List[str] = None
    ) -> List[Tuple[MemoryItem, str]]:
        """
        Retrieve memories and return enriched content

        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            include_modules: Module outputs to include ['entity_relation', 'temporal_relation', 'summary']

        Returns:
            List of (MemoryItem, enriched_content) tuples
        """
        if include_modules is None:
            include_modules = ['entity_relation', 'temporal_relation', 'summary']

        retrieved = self.retrieve(query_embedding, top_k)
        results = []

        for mem in retrieved:
            parts = []

            # Base content
            if mem.date_time:
                parts.append(f"[Date: {mem.date_time}]")
            parts.append(mem.content)

            # Add module outputs as needed
            if 'entity_relation' in include_modules and mem.entity_relations:
                parts.append("\n[Entities & Relations]")
                parts.extend(mem.entity_relations[:10])

            if 'temporal_relation' in include_modules and mem.temporal_relations:
                parts.append("\n[Timeline]")
                parts.extend(mem.temporal_relations[:10])

            if 'summary' in include_modules and mem.summary:
                parts.append(f"\n[Summary]\n{mem.summary}")

            enriched_content = "\n".join(parts)
            results.append((mem, enriched_content))

        return results

    def get_all_entity_relations(self) -> List[str]:
        """Get all Entity relations from memories"""
        all_relations = []
        for mem in self.memories:
            if mem.entity_relations:
                all_relations.extend(mem.entity_relations)
        return list(set(all_relations))

    def get_all_temporal_relations(self) -> List[str]:
        """Get all Temporal relations from memories"""
        all_relations = []
        for mem in self.memories:
            if mem.temporal_relations:
                all_relations.extend(mem.temporal_relations)
        return all_relations

    def get_total_processing_cost(self) -> float:
        """Get total processing cost for all memories"""
        return sum(mem.total_cost for mem in self.memories)





def get_llm_judge_reward(question: str, ground_truth: str, prediction: str, args) -> float:
    """
    Evaluate answer quality using LLM judge and return reward (0.0/0.5/1.0)

    Args:
        question: Question text
        ground_truth: Ground truth answer
        prediction: Model prediction
        args: Arguments object

    Returns:
        reward score (0.0, 0.5, or 1.0)
    """
    judge_prompt = LLM_JUDGE_GENERAL_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=prediction
    )

    judge_args = copy.deepcopy(args)
    judge_args.max_new_tokens = 512
    judge_args.model = "openai/gpt-oss-120b"
    judge_args.temperature = 0.0
    judge_args.api = True
    api_base = "https://integrate.api.nvidia.com/v1"
    api = "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"
    # Call LLM judge
    try:
        response, _, _ = get_llm_response_via_api(
            prompt=judge_prompt,
            MAX_TOKENS=judge_args.max_new_tokens,
            LLM_MODEL=judge_args.model,
            TAU=judge_args.temperature,
            base_url=api_base,
            api_key=api,
        )

        if response is None:
            print(f"[LLM Judge] API returned None, returning 0.0")
            return 0.0

        reward = parse_judge_response(response)
        return reward
    except Exception as e:
        print(f"[LLM Judge] Error: {e}, returning 0.0")
        return 0.0



def construct_global_memory_hotpotqa(data: Dict, retriever: str, context_tokenizer, context_encoder, args,
                                    data_tokenizer=None, data_encoder=None, max_tokens_per_chunk: int = 256) -> GlobalMemoryPool:
    global_memory = GlobalMemoryPool()

    # Get document list
    documents = data.get('documents', [])
    sample_id = data.get('sample_id', 'unknown')
    date_time = data.get('conversation', {}).get('session_1_date_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Check if document list is empty
    if not documents or len(documents) == 0:
        print(f"Warning: Sample {sample_id} has no documents, returning empty memory pool")
        return global_memory

    
    if data_encoder is not None:
        # SentenceTransformer object, get underlying tokenizer
        tokenizer = data_encoder.tokenizer
    elif data_tokenizer is not None:
        # If data_tokenizer is SentenceTransformer, also get its tokenizer
        tokenizer = getattr(data_tokenizer, 'tokenizer', data_tokenizer)
    elif context_tokenizer is not None:
        tokenizer = context_tokenizer
    else:
        # Finally fallback to args.model tokenizer (not recommended, may be inconsistent with embedding)
        tokenizer = get_tokenizer(args.model) if hasattr(args, 'model') and args.model else get_tokenizer('gpt-3.5-turbo')
    
    # Unify tokenizer interface
    def count_tokens(text):
        if hasattr(tokenizer, 'encode'):
            return len(tokenizer.encode(text, add_special_tokens=False))
        else:
            return len(tokenizer(text))

    all_chunk_contents = []
    all_chunk_metadata = []
    current_chunk_parts = []
    current_chunk_doc_indices = []  # Explicitly maintain document indices contained in current chunk
    current_chunk_tokens = 0
    chunk_idx = 0

    # Split documents by token count
    for doc_num, doc_content in enumerate(documents, 1):
        doc_text = f"Document {doc_num}:\n{doc_content}"
        doc_tokens = count_tokens(doc_text)
        doc_idx = doc_num - 1  # Actual index of document in list

        # If current chunk plus this document exceeds limit, save current chunk first
        if current_chunk_tokens + doc_tokens > max_tokens_per_chunk and current_chunk_parts:
            chunk_content = "\n\n".join(current_chunk_parts)
            all_chunk_contents.append(chunk_content)
            all_chunk_metadata.append({
                'session_id': 1,
                'date_time': date_time,
                'chunk_idx': chunk_idx,
                'doc_indices': current_chunk_doc_indices.copy()  # Use explicitly maintained document index list
            })
            chunk_idx += 1
            current_chunk_parts = []
            current_chunk_doc_indices = []
            current_chunk_tokens = 0

        # If single document exceeds limit, need to split document itself
        if doc_tokens > max_tokens_per_chunk:
            # Simple split: split by sentences
            sentences = doc_content.split('. ')
            current_sentences = []
            header_text = f"Document {doc_num}:\n"
            current_sent_tokens = count_tokens(header_text)

            for sent in sentences:
                sent_text = sent + '. '
                sent_tokens = count_tokens(sent_text)

                if current_sent_tokens + sent_tokens > max_tokens_per_chunk and current_sentences:
                    chunk_content = f"Document {doc_num}:\n" + ' '.join(current_sentences)
                    all_chunk_contents.append(chunk_content)
                    all_chunk_metadata.append({
                        'session_id': 1,
                        'date_time': date_time,
                        'chunk_idx': chunk_idx,
                        'doc_indices': [doc_idx]  # Single document chunk only contains this one document
                    })
                    chunk_idx += 1
                    current_sentences = []
                    current_sent_tokens = count_tokens(f"Document {doc_num}:\n")

                current_sentences.append(sent_text)
                current_sent_tokens += sent_tokens

            if current_sentences:
                chunk_content = f"Document {doc_num}:\n" + ' '.join(current_sentences)
                all_chunk_contents.append(chunk_content)
                all_chunk_metadata.append({
                    'session_id': 1,
                    'date_time': date_time,
                    'chunk_idx': chunk_idx,
                    'doc_indices': [doc_idx]  # Single document chunk only contains this one document
                })
                chunk_idx += 1
        else:
            # Add to current chunk
            current_chunk_parts.append(doc_text)
            current_chunk_doc_indices.append(doc_idx)  # Explicitly record document index
            current_chunk_tokens += doc_tokens

    # Save last chunk
    if current_chunk_parts:
        chunk_content = "\n\n".join(current_chunk_parts)
        all_chunk_contents.append(chunk_content)
        all_chunk_metadata.append({
            'session_id': 1,
            'date_time': date_time,
            'chunk_idx': chunk_idx,
            'doc_indices': current_chunk_doc_indices.copy()  # Use explicitly maintained document index list
        })

    print(f"Chunking {len(documents)} documents into {len(all_chunk_contents)} chunks (max {max_tokens_per_chunk} tokens per chunk) for sample {sample_id}")

    if data_encoder is not None:
        embeddings = get_data_embeddings(data_encoder, all_chunk_contents, batch_size=16)
    else:
        embeddings = get_embeddings_with_model(retriever, all_chunk_contents, context_tokenizer, context_encoder)

    # Create memory items
    for chunk_content, embedding, metadata in zip(all_chunk_contents, embeddings, all_chunk_metadata):
        unique_id = metadata['session_id'] * 1000 + metadata['chunk_idx']
        memory = MemoryItem(
            content=chunk_content,
            embedding=embedding,
            session_id=unique_id,
            date_time=metadata['date_time'],
            processed=False,
            original_dialogs=[documents[i] for i in metadata['doc_indices']] if metadata['doc_indices'] else [],
            is_original=True
        )
        global_memory.add_memory(memory)

    global_memory.build_index()
    print(f"Global memory constructed with {len(global_memory)} chunks from {len(documents)} documents")

    return global_memory


def preprocess_memory_extractions_parallel(
    global_memory: GlobalMemoryPool,
    module2,
    module3,
    module5=None,
    max_workers: int = None
) -> None:
    """
    Parallel preprocessing of all memory items to extract entity, temporal, and topic relations.
    This optimization caches extraction results to avoid repeated computation
    during training since low/mid cost extractions are query-independent.

    Args:
        global_memory: GlobalMemoryPool containing all memories
        module2: Module2_EntityRelation instance
        module3: Module3_TemporalRelation instance
        module5: Module5_TopicRelation instance (optional)
        max_workers: Maximum number of worker threads (default: min(CPU count, 8))
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers is None:
        max_workers = min(os.cpu_count() or 16, 16)  # Cap at 16 for rule_llm strategy

    # Filter memories that need preprocessing
    memories_to_process = [
        (i, mem) for i, mem in enumerate(global_memory.memories)
        if mem.entity_relations is None or mem.temporal_relations is None or (module5 is not None and mem.topic_relations is None)
    ]

    if not memories_to_process:
        return

    total_memories = len(memories_to_process)

    def process_single_memory(idx_mem_tuple):
        """Process a single memory item - extract entities, temporal, and topic relations"""
        i, memory = idx_mem_tuple

        # Extract entity relations using module2's low+mid cost methods
        # Note: query is not used in low/mid cost methods, so we pass empty string
        try:
            # These methods now use cache internally, but first time we need to extract
            entity_relations_low = module2._low_cost_extract("", [memory])
            entity_relations_mid = module2._mid_cost_extract("", [memory])

            all_entity_relations = list(set(entity_relations_low + entity_relations_mid))
            memory.entity_relations = all_entity_relations
        except Exception as e:
            print(f"[Preprocessing] Warning: Failed to extract entities for memory {i}: {e}")
            memory.entity_relations = []

        try:
            temporal_relations_low = module3._low_cost_extract("", [memory])
            temporal_relations_mid = module3._mid_cost_extract("", [memory])
            all_temporal_relations = list(set(temporal_relations_low + temporal_relations_mid))
            memory.temporal_relations = all_temporal_relations
        except Exception as e:
            print(f"[Preprocessing] Warning: Failed to extract temporal info for memory {i}: {e}")
            memory.temporal_relations = []

        if module5 is not None:
            try:
                topic_relations_low = module5._low_cost_extract("", [memory])
                topic_relations_mid = module5._mid_cost_extract("", [memory])
                all_topic_relations = list(set(topic_relations_low + topic_relations_mid))
                memory.topic_relations = all_topic_relations
            except Exception as e:
                print(f"[Preprocessing] Warning: Failed to extract topic relations for memory {i}: {e}")
                memory.topic_relations = []

        return i

    processed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_memory, idx_mem): idx_mem[0]
            for idx_mem in memories_to_process
        }

        for future in as_completed(future_to_idx):
            try:
                idx = future.result()
                processed_count += 1

                if processed_count % 20 == 0 or processed_count == total_memories:
                    print(f"[Preprocessing] Progress: {processed_count}/{total_memories} memories processed")

            except Exception as e:
                idx = future_to_idx[future]
                print(f"[Preprocessing] Error processing memory {idx}: {e}")


def preprocess_all_samples_memories_hotpotqa(
    samples: List[Dict],
    args,
    context_tokenizer,
    context_encoder,
    data_tokenizer,
    data_encoder,
    Module2_EntityRelation,
    Module3_TemporalRelation,
    Module5_TopicRelation=None,
    max_workers: int = 32,
    max_tokens_per_chunk: int = 256
) -> Dict[str, GlobalMemoryPool]:
    from tqdm import tqdm

    print(f"\n{'='*60}")
    print(f"PREPROCESSING ALL HOTPOTQA SAMPLES' MEMORIES")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)}")
    print(f"Max tokens per chunk: {max_tokens_per_chunk}")
    print(f"Max workers for parallel extraction: {max_workers}")
    print()

    # Initialize modules once (shared across all samples)
    module2 = Module2_EntityRelation(args=args)
    module3 = Module3_TemporalRelation(args=args)
    module5 = Module5_TopicRelation(args=args) if Module5_TopicRelation is not None else None

    sample_memory_pools = {}

    for data in tqdm(samples, desc="Preprocessing samples", unit="sample"):
        sample_id = data['sample_id']

        # Step 1: Construct global memory pool for this HotpotQA sample
        global_memory = construct_global_memory_hotpotqa(
            data, args.retriever, context_tokenizer, context_encoder, args,
            data_tokenizer, data_encoder,
            max_tokens_per_chunk=max_tokens_per_chunk
        )

        # Step 2: Parallel extract entities, temporal, and topic relations for all memories
        preprocess_memory_extractions_parallel(global_memory, module2, module3, module5=module5, max_workers=max_workers)

        # Step 3: Store preprocessed memory pool
        sample_memory_pools[sample_id] = global_memory

    print(f"\n✓ Preprocessing completed for {len(sample_memory_pools)} samples")
    print(f"{'='*60}\n")

    return sample_memory_pools


def save_preprocessed_memories(
    sample_memory_pools: Dict[str, GlobalMemoryPool],
    dataset_name: str,
    cost_strategy: str,
    split: str = 'train',
    base_dir: str = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lx/memory/res_code/res_data"
) -> str:
    """
    Save preprocessed memory pools to disk for future reuse.

    Args:
        sample_memory_pools: Dictionary mapping sample_id to GlobalMemoryPool
        dataset_name: Name of the dataset (extracted from data file path)
        cost_strategy: Cost strategy used (e.g., 'rule_llm')
        split: Data split ('train' or 'test')
        base_dir: Base directory for saving preprocessed data

    Returns:
        Path to the saved file
    """
    import os
    import pickle

    save_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"preprocessed_memories_{split}_{cost_strategy}.pkl"
    save_path = os.path.join(save_dir, filename)

    print(f"\n[Saving] Saving preprocessed memories to: {save_path}")
    save_data = {
        'version': '1.0',
        'dataset_name': dataset_name,
        'cost_strategy': cost_strategy,
        'num_samples': len(sample_memory_pools),
        'sample_memory_pools': sample_memory_pools
    }

    try:
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Get file size for reporting
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"[Saving] ✓ Successfully saved {len(sample_memory_pools)} samples ({file_size_mb:.2f} MB)")
        print(f"[Saving] Location: {save_path}\n")

        return save_path
    except Exception as e:
        print(f"[Saving] Error: Failed to save preprocessed memories: {e}")
        return None


def load_preprocessed_memories(
    dataset_name: str,
    cost_strategy: str,
    split: str = 'train',
    base_dir: str = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lx/memory/res_code/res_data"
) -> Optional[Dict[str, GlobalMemoryPool]]:
    import os
    import pickle

    filename = f"preprocessed_memories_{split}_{cost_strategy}.pkl"
    load_path = os.path.join(base_dir, dataset_name, filename)

    if not os.path.exists(load_path):
        print(f"[Loading] No cached preprocessed memories found at: {load_path}")
        return None

    print(f"\n[Loading] Found cached preprocessed memories at: {load_path}")

    try:
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)

        if save_data.get('dataset_name') != dataset_name:
            print(f"[Loading] Warning: Dataset name mismatch. Expected '{dataset_name}', got '{save_data.get('dataset_name')}'")
            return None

        if save_data.get('cost_strategy') != cost_strategy:
            print(f"[Loading] Warning: Cost strategy mismatch. Expected '{cost_strategy}', got '{save_data.get('cost_strategy')}'")
            return None

        sample_memory_pools = save_data.get('sample_memory_pools')
        num_samples = len(sample_memory_pools) if sample_memory_pools else 0
        file_size_mb = os.path.getsize(load_path) / (1024 * 1024)
        print(f"[Loading] ✓ Successfully loaded {num_samples} samples ({file_size_mb:.2f} MB)")
        print(f"[Loading] Version: {save_data.get('version', 'unknown')}\n")

        return sample_memory_pools

    except Exception as e:
        print(f"[Loading] Error: Failed to load preprocessed memories: {e}")
        print(f"[Loading] Will proceed with fresh preprocessing...\n")
        return None


def extract_dataset_name(data_file_path: str) -> str:
    """
    Extract dataset name from data file path.

    Examples:
        '../data/locomo/train.json' -> 'locomo_train'
        '/path/to/dataset.json' -> 'dataset'
    """
    import os

    basename = os.path.basename(data_file_path)
    dataset_name = os.path.splitext(basename)[0]
    parent_dir = os.path.basename(os.path.dirname(data_file_path))

    if parent_dir and parent_dir not in ['.', '..', 'data']:
        return f"{parent_dir}_{dataset_name}"

    return dataset_name



def train_epoch(
    samples: List[Dict],
    actor_critic: ActorCriticNetwork,
    ppo_trainer: PPOTrainer,
    context_tokenizer,
    context_encoder,
    query_tokenizer,
    query_encoder,
    args,
    Module1_Filter,
    Module2_EntityRelation,
    Module3_TemporalRelation,
    Module5_TopicRelation,
    Module4_Summary,
    ModularPipelineExecutor,
    data_tokenizer=None,
    data_encoder=None,
    epoch: int = 0,
    global_batch_idx: int = 0,
    sample_memory_pools: Optional[Dict[str, GlobalMemoryPool]] = None,
    model_save_dir: str = None
) -> Tuple[Dict[str, float], int]:
    """
    Train one epoch - new 5-module system

    Uses Module1(Filter), Module2(Entity), Module3(Temporal), Module4(Summary), Module5(Topic)
    Each module has 3 cost levels: low/mid/high

    Args:
        global_batch_idx: Global batch counter, starts from 0 and increments, used as wandb x-axis
        sample_memory_pools: Optional preprocessed memory pools for all samples
                            (maps sample_id -> GlobalMemoryPool)

    Returns:
        Metrics dictionary and updated global_batch_idx
    """

    total_reward = 0.0
    num_questions = 0
    all_metrics = []
    global_action_counts = {
        'module1_low': 0, 'module1_mid': 0, 'module1_high': 0,
        'module2_low': 0, 'module2_mid': 0, 'module2_high': 0,
        'module3_low': 0, 'module3_mid': 0, 'module3_high': 0,
        'module5_low': 0, 'module5_mid': 0, 'module5_high': 0,
        'module4_low': 0, 'module4_mid': 0, 'module4_high': 0,
    }
    all_pipeline_outputs = []  # Store all sample pipeline outputs
    all_question_results = []  # Store all sample question results

    # Track best step reward for checkpoint saving
    best_step_reward = -float('inf')
    best_step = 0

    for data in tqdm(samples, desc="Training"):
        print(f"\n{'='*60}")
        print(f"Processing sample: {data['sample_id']}")
        print(f"{'='*60}")

        sample_id = data['sample_id']

        if sample_memory_pools and sample_id in sample_memory_pools:
            global_memory = sample_memory_pools[sample_id]
            print(f"Using preprocessed global memory with {len(global_memory)} chunks")
        else:
            # Fallback: construct on-the-fly (for backward compatibility)
            global_memory = construct_global_memory_hotpotqa(
                data, args.retriever, context_tokenizer, context_encoder, args,
                data_tokenizer, data_encoder,
                max_tokens_per_chunk=args.chunk_max_tokens
            )
            print(f"Constructed global memory with {len(global_memory)} chunks")

        module1 = Module1_Filter(
            encoder=data_encoder,
            args=args,
            top_k=args.module_topk
        )
        module2 = Module2_EntityRelation(
            args=args
        )
        module3 = Module3_TemporalRelation(
            args=args
        )
        module5 = Module5_TopicRelation(
            args=args
        )
        module4 = Module4_Summary(
            args=args
        )

        pipeline = ModularPipelineExecutor(
            module1=module1,
            module2=module2,
            module3=module3,
            module5=module5,
            module4=module4,
            actor_critic=actor_critic,
            device=DEVICE,
            encoder=data_encoder
        )

        # Step 3: Prepare questions (HotpotQA each sample has only one question)
        out_data = {'sample_id': data['sample_id'], 'qa': copy.deepcopy(data['qa'])}
        questions = []
        for qa in data['qa']:
            questions.append(qa['question'])

        # Step 4: Batch compute question embeddings
        if data_encoder is not None:
            all_query_embs = get_data_embeddings(data_encoder, questions, batch_size=16)
        else:
            all_query_embs = get_embeddings_with_model(args.retriever, questions, query_tokenizer, query_encoder)

        def process_single_question(i, qa, question, query_emb_np, answer):
            """
            Process a single question complete pipeline (retrieval, pipeline, evaluation)

            Args:
                i: Question index
                qa: Question-answer dictionary
                question: Question text
                query_emb_np: Question embedding
                answer: Ground truth answer (passed from out_data to avoid concurrent access)

            Returns: Dictionary containing all necessary information for subsequent PPO updates and memory writes
            """
            if query_emb_np.ndim > 1:
                query_emb_np = query_emb_np.flatten()

            query_emb = torch.from_numpy(query_emb_np).float()

            # Retrieve relevant memories
            effective_top_k = max(args.top_k, 10)
            retrieved_memories = global_memory.retrieve(query_emb_np, top_k=effective_top_k)
            # Execute pipeline
            pipeline_result = pipeline.execute(
                query=question,
                query_emb=query_emb,
                initial_memories=retrieved_memories,
                deterministic=False
            )
            filtered_memories = pipeline_result['filtered_memories']
            entity_relations = pipeline_result['entity_relations']
            temporal_relations = pipeline_result['temporal_relations']
            summary = pipeline_result['summary']
            actions = pipeline_result['actions']
            log_probs = pipeline_result['log_probs']
            state_value = pipeline_result['state_value']
            total_cost_q = pipeline_result['total_cost']
            topic_relations = pipeline_result['topic_relations']        
            context_parts = []
            
            if summary:
                context_parts.append("\n<Summary>")
                context_parts.append(summary)
                context_parts.append("</Summary>")
            
            query_context = '\n'.join(context_parts)
            
            
            input_prompt = HOTPOTQA_ANSWER_PROMPT.format(
                    context=query_context, question=question
                )
            
            # Get LLM answer
            disable_threading = args.parallel_questions > 1
            task_args = [(i, input_prompt, args)]
            ret = get_llm_response(args=args, task_args=task_args, disable_internal_threading=disable_threading)

            # Calculate reward
            prediction = ""
            answer_str = str(answer)

            if len(ret) > 0:
                idx, response, _, success = ret[0]
                if success:
                    prediction = response.strip()
                    if "<answer>" in prediction and "</answer>" in prediction:
                        start = prediction.find("<answer>") + len("<answer>")
                        end = prediction.find("</answer>", start)
                        if end > start:
                            prediction = prediction[start:end].strip()
                    if args.llm_judge:
                        judge_question = qa['question']
                        reward = get_llm_judge_reward(judge_question, answer_str, prediction, args)
                    else:
                        
                        reward = f1_score(prediction, answer_str)
                else:
                    reward = 0.0
            else:
                reward = 0.0

            if len(retrieved_memories) > 0:
                memory_embs_np = np.stack([m.embedding for m in retrieved_memories[:10]])
                memory_embs = torch.from_numpy(memory_embs_np).float()
            else:
                memory_embs = torch.zeros(10, 768)

            if memory_embs.shape[0] < 10:
                padding = torch.zeros(10 - memory_embs.shape[0], memory_embs.shape[1])
                memory_embs = torch.cat([memory_embs, padding], dim=0)
                
            return {
                'question_index': i,
                'question': question,
                'query_emb': query_emb,
                'memory_embs': memory_embs,
                'actions': actions,
                'log_probs': log_probs,
                'state_value': state_value,
                'reward': reward,
                'total_cost': total_cost_q,
                'retrieved_memories': retrieved_memories,
                'filtered_memories': filtered_memories,
                'entity_relations': entity_relations,
                'temporal_relations': temporal_relations,
                'topic_relations': topic_relations,
                'summary': summary,
                'pipeline_result': pipeline_result,
                'prediction': prediction,
                'final_prompt': input_prompt,
                'ground_truth': answer_str
            }

        # Step 6: Process single question (HotpotQA each sample has only one question)
        sample_rewards = []
        sample_pipeline_outputs = []
        question_results = []

        # Process single question
        i = 0  # Question index
        qa = data['qa'][i]
        question = questions[i]
        query_emb_np = all_query_embs[i]
        answer = out_data['qa'][i]['answer']

        result = process_single_question(i, qa, question, query_emb_np, answer)
        
        # Perform PPO update on single question
        print(f"\nPerforming PPO update on question result...")

        question_raw_costs = [result['total_cost']]

        # Batch normalize current question costs
        if question_raw_costs:
            question_normalized_costs = normalize_costs_batch(question_raw_costs, add_to_history=True)

            # Update normalized costs in result
            if len(question_normalized_costs) > 0:
                result['total_cost'] = question_normalized_costs[0]

        # After normalization, perform reward and cost scale alignment for current question
        question_base_reward = result['reward']
        question_base_cost = result['total_cost']

        question_aligned_rewards, question_aligned_costs = align_reward_cost_scales_batch(
            f1_rewards=[question_base_reward],
            costs=[question_base_cost]
        )

        # Update aligned values in result
        if len(question_aligned_rewards) > 0:
            result['aligned_reward'] = question_aligned_rewards[0]
            result['aligned_cost'] = question_aligned_costs[0]

        question_results.append(result)

        batch_buffer = ExperienceBuffer()
        for result in question_results:
            i = result['question_index']
            reward = result['reward']
            actions = result['actions']
            m1_action, m2_action, m3_action, m5_action, m4_action = actions

            if 'prediction' in result:
                out_data['qa'][i]['prediction'] = result['prediction']

            retrieved_memory_contents = []
            for mem in result['retrieved_memories']:
                if hasattr(mem, 'get_enriched_content'):
                    retrieved_memory_contents.append(mem.get_enriched_content())
                else:
                    content = mem.content
                    if hasattr(mem, 'date_time') and mem.date_time:
                        content = f"[Date: {mem.date_time}] {content}"
                    retrieved_memory_contents.append(content)

            filtered_memory_contents = []
            for mem in result['filtered_memories']:
                if hasattr(mem, 'get_enriched_content'):
                    filtered_memory_contents.append(mem.get_enriched_content())
                else:
                    content = mem.content
                    if hasattr(mem, 'date_time') and mem.date_time:
                        content = f"[Date: {mem.date_time}] {content}"
                    filtered_memory_contents.append(content)

            log_question_trace({
                "sample_id": str(data["sample_id"]),
                "epoch": int(epoch),
                "question_index": int(i),
                "question": str(result.get("question", "")),
                "retrieved_memories": [str(x) for x in retrieved_memory_contents],
                "pipeline_outputs": {
                    "filtered_memories": [str(x) for x in filtered_memory_contents],
                    "entity_relations": [str(x) for x in (result.get("entity_relations") or [])],
                    "temporal_relations": [str(x) for x in (result.get("temporal_relations") or [])],
                    "topic_relations": [str(x) for x in (result.get("topic_relations") or [])],
                    "summary": str(result.get("summary") or ""),
                    "actions": [int(a) for a in actions],
                    "reward": float(reward) if hasattr(reward, "item") else float(reward),
                },
            })
            tracking_record = QuestionTrackingRecord(
                sample_id=data['sample_id'],
                question_index=i,
                question=result['question'],
                ground_truth=result.get('ground_truth', str(out_data['qa'][i]['answer'])),
                prediction=result.get('prediction', ''),
                reward=reward,
                epoch=epoch,
                retrieved_memories=retrieved_memory_contents,
                filtered_memories=filtered_memory_contents,
                entity_relations=result.get('entity_relations', []),
                temporal_relations=result.get('temporal_relations', []),
                topic_relations=result.get('topic_relations', []),
                summary=result.get('summary', ''),
                final_prompt=result.get('final_prompt', ''),
                actions=actions
            )
            global_question_tracker.add_record(tracking_record)

            # Record reward - execute once
            sample_rewards.append(reward)
            total_reward += reward
            num_questions += 1

            sample_pipeline_outputs.append({
                'sample_id': data['sample_id'],
                'question_index': i,
                'question': result['question'],
                'actions': {
                    'module1': ['low', 'mid', 'high'][m1_action],
                    'module2': ['low', 'mid', 'high'][m2_action],
                    'module3': ['low', 'mid', 'high'][m3_action],
                    'module5': ['low', 'mid', 'high'][m5_action],
                    'module4': ['low', 'mid', 'high'][m4_action]
                },
                'actions_raw': [int(m1_action), int(m2_action), int(m3_action), int(m5_action), int(m4_action)],
                'retrieved_sessions_count': len(result['retrieved_memories']),
                'f1_reward': reward,
                'filtered_memories_count': len(result['filtered_memories']),
                'entity_relations_count': len(result['entity_relations']) if result['entity_relations'] else 0,
                'temporal_relations_count': len(result['temporal_relations']) if result['temporal_relations'] else 0,
                'has_summary': bool(result['summary'])
            })

           
            aligned_reward = result.get('aligned_reward', reward)
            aligned_cost = result.get('aligned_cost', result['total_cost'])

            pipeline_result = result['pipeline_result']
            exp = Experience(
                query_emb=result['query_emb'],
                memory_embs=result['memory_embs'],
                actions=list(result['actions']),  # Order: [m1, m2, m3, m5, m4]
                log_probs=result['log_probs'],  # Order: [m1, m2, m3, m5, m4] from rollout-time policy
                reward=aligned_reward,  # Use scale-aligned reward
                cost=aligned_cost,      # Use aligned cost
                value=result['state_value'],
                done=True,
                initial_memory_emb=pipeline_result.get('initial_memory_emb'),
                filtered_memory_emb=pipeline_result.get('filtered_memory_emb'),
                entity_emb=pipeline_result.get('entity_emb'),
                temporal_emb=pipeline_result.get('temporal_emb'),
                topic_emb=pipeline_result.get('topic_emb'),
                aggregated_memory_emb=pipeline_result.get('aggregated_memory_emb'),  # CRITICAL: for Module4
            )
            batch_buffer.add(exp)

        # PPO update
        batch_metrics = None
        if len(batch_buffer) > 0:
            batch_metrics = ppo_trainer.update(batch_buffer)
            if batch_metrics:
                all_metrics.append(batch_metrics)

        # Record question-level wandb logs
        question_rewards = [result['reward'] for result in question_results]
        question_action_counts = {
            'module1_low': 0, 'module1_mid': 0, 'module1_high': 0,
            'module2_low': 0, 'module2_mid': 0, 'module2_high': 0,
            'module3_low': 0, 'module3_mid': 0, 'module3_high': 0,
            'module5_low': 0, 'module5_mid': 0, 'module5_high': 0,
            'module4_low': 0, 'module4_mid': 0, 'module4_high': 0,
        }

        for result in question_results:
            actions = result['actions']
            m1_action, m2_action, m3_action, m5_action, m4_action = actions
            action_levels = ['low', 'mid', 'high']
            question_action_counts[f'module1_{action_levels[m1_action]}'] += 1
            question_action_counts[f'module2_{action_levels[m2_action]}'] += 1
            question_action_counts[f'module3_{action_levels[m3_action]}'] += 1
            question_action_counts[f'module5_{action_levels[m5_action]}'] += 1
            question_action_counts[f'module4_{action_levels[m4_action]}'] += 1

        # Calculate total actions for each module
        question_module1_total = (question_action_counts['module1_low'] +
                                question_action_counts['module1_mid'] +
                                question_action_counts['module1_high'])
        question_module2_total = (question_action_counts['module2_low'] +
                                question_action_counts['module2_mid'] +
                                question_action_counts['module2_high'])
        question_module3_total = (question_action_counts['module3_low'] +
                                question_action_counts['module3_mid'] +
                                question_action_counts['module3_high'])
        question_module5_total = (question_action_counts['module5_low'] +
                                question_action_counts['module5_mid'] +
                                question_action_counts['module5_high'])
        question_module4_total = (question_action_counts['module4_low'] +
                                question_action_counts['module4_mid'] +
                                question_action_counts['module4_high'])

        # Calculate question-level metrics
        question_rewards = [result['reward'] for result in question_results]  # Raw F1 rewards
        question_costs = [result['total_cost'] for result in question_results]  # cost efficiencies
        question_aligned_rewards = [result.get('aligned_reward', result['reward']) for result in question_results]  # Aligned F1 rewards
        question_aligned_costs = [result.get('aligned_cost', result['total_cost']) for result in question_results]  # Aligned costs

        question_avg_reward = np.mean(question_rewards) if question_rewards else 0.0  # Raw F1 average
        question_avg_cost = np.mean(question_costs) if question_costs else 0.0  # Cost efficiency average
        question_avg_aligned_reward = np.mean(question_aligned_rewards) if question_aligned_rewards else 0.0  # Aligned F1 average
        question_avg_aligned_cost = np.mean(question_aligned_costs) if question_aligned_costs else 0.0  # Aligned cost average

        reward_weight = getattr(args, 'reward_weight', 1.0)
        cost_weight = getattr(args, 'cost_weight', 0.0)
        question_rl_train_reward = np.mean([reward_weight * r + cost_weight * c for r, c in zip(question_aligned_rewards, question_aligned_costs)]) if question_aligned_rewards and question_aligned_costs else 0.0

        question_log = {
            'question/ppo/policy_loss': batch_metrics.get('policy_loss', 0) if batch_metrics else 0,
            'question/ppo/value_loss': batch_metrics.get('value_loss', 0) if batch_metrics else 0,
            'question/ppo/entropy': batch_metrics.get('entropy', 0) if batch_metrics else 0,
            'question/ppo/approx_kl': batch_metrics.get('approx_kl', 0) if batch_metrics else 0,
            'question/ppo/clip_fraction': batch_metrics.get('clip_fraction', 0) if batch_metrics else 0,
            'question/ppo/grad_norm': batch_metrics.get('grad_norm', 0) if batch_metrics else 0,
            'question/ppo/explained_variance': batch_metrics.get('explained_variance', 0) if batch_metrics else 0,
            'question/ppo/avg_cost': batch_metrics.get('avg_cost', 0) if batch_metrics else 0,
            'question/ppo/num_updates': batch_metrics.get('num_updates', 0) if batch_metrics else 0,

            f'question/train/{args.llm_judge}_f1_avg_reward': question_avg_reward,
            'question/train/cost_avg_reward': question_avg_cost,
            'question/train/f1_aligned_avg_reward': question_avg_aligned_reward,
            'question/train/cost_aligned_avg': question_avg_aligned_cost,
            'question/rl/train_reward': question_rl_train_reward,
            'question/train/f1_max_reward': max(question_rewards) if question_rewards else 0.0,
            'question/train/f1_min_reward': min(question_rewards) if question_rewards else 0.0,
            'question/train/f1_std_reward': float(np.std(question_rewards)) if question_rewards else 0.0,
            'question/train/num_questions': len(question_results),
            'question/train/epoch': epoch,

            'question/modules/module1/low_ratio': question_action_counts['module1_low'] / max(question_module1_total, 1),
            'question/modules/module1/mid_ratio': question_action_counts['module1_mid'] / max(question_module1_total, 1),
            'question/modules/module1/high_ratio': question_action_counts['module1_high'] / max(question_module1_total, 1),
            'question/modules/module2/low_ratio': question_action_counts['module2_low'] / max(question_module2_total, 1),
            'question/modules/module2/mid_ratio': question_action_counts['module2_mid'] / max(question_module2_total, 1),
            'question/modules/module2/high_ratio': question_action_counts['module2_high'] / max(question_module2_total, 1),

            'question/modules/module3/low_ratio': question_action_counts['module3_low'] / max(question_module3_total, 1),
            'question/modules/module3/mid_ratio': question_action_counts['module3_mid'] / max(question_module3_total, 1),
            'question/modules/module3/high_ratio': question_action_counts['module3_high'] / max(question_module3_total, 1),

            'question/modules/module5/low_ratio': question_action_counts['module5_low'] / max(question_module5_total, 1),
            'question/modules/module5/mid_ratio': question_action_counts['module5_mid'] / max(question_module5_total, 1),
            'question/modules/module5/high_ratio': question_action_counts['module5_high'] / max(question_module5_total, 1),

            'question/modules/module4/low_ratio': question_action_counts['module4_low'] / max(question_module4_total, 1),
            'question/modules/module4/mid_ratio': question_action_counts['module4_mid'] / max(question_module4_total, 1),
            'question/modules/module4/high_ratio': question_action_counts['module4_high'] / max(question_module4_total, 1),
        }

        wandb.log(question_log, step=global_batch_idx)
        global_batch_idx += 1

        # Check if need to save step checkpoint (save every 100 steps, and update best checkpoint)
        if model_save_dir and global_batch_idx % 100 == 0:
            # Calculate current batch average reward
            batch_rewards = [r.get('aligned_reward', r['reward']) for r in question_results if 'aligned_reward' in r or 'reward' in r]
            if batch_rewards:
                current_batch_reward = sum(batch_rewards) / len(batch_rewards)

                # 1. Save a checkpoint every 100 steps (do not delete old ones)
                checkpoint_path = os.path.join(model_save_dir, f"checkpoint_step_{global_batch_idx}_{args.cost_strategy}.pt")
                torch.save({
                    'step': global_batch_idx,
                    'epoch': epoch,
                    'num_actions_per_module': int(getattr(actor_critic, 'num_actions_per_module', 4)),
                    'model_state_dict': actor_critic.state_dict(),
                    'optimizer_state_dict': ppo_trainer.optimizer.state_dict(),
                    'cost_strategy': args.cost_strategy,
                    'batch_reward': current_batch_reward
                }, checkpoint_path)
                print(f"Checkpoint saved at step {global_batch_idx} (reward: {current_batch_reward:.4f}): {checkpoint_path}")

                # 2. If current batch reward is better, update best checkpoint
                if current_batch_reward > best_step_reward:
                    best_step_reward = current_batch_reward
                    best_step = global_batch_idx

                    # Save best checkpoint (use fixed filename, will overwrite previous)
                    best_checkpoint_path = os.path.join(model_save_dir, f"checkpoint_step_best_{args.cost_strategy}.pt")
                    torch.save({
                        'step': global_batch_idx,
                        'epoch': epoch,
                        'num_actions_per_module': int(getattr(actor_critic, 'num_actions_per_module', 4)),
                        'model_state_dict': actor_critic.state_dict(),
                        'optimizer_state_dict': ppo_trainer.optimizer.state_dict(),
                        'cost_strategy': args.cost_strategy,
                        'batch_reward': current_batch_reward
                    }, best_checkpoint_path)
                    print(f"New best checkpoint saved at step {global_batch_idx} (reward: {current_batch_reward:.4f}): {best_checkpoint_path}")

        # Accumulate action counts
        sample_action_counts = {
            'module1_low': 0, 'module1_mid': 0, 'module1_high': 0,
            'module2_low': 0, 'module2_mid': 0, 'module2_high': 0,
            'module3_low': 0, 'module3_mid': 0, 'module3_high': 0,
            'module5_low': 0, 'module5_mid': 0, 'module5_high': 0,
            'module4_low': 0, 'module4_mid': 0, 'module4_high': 0,
        }
        for result in question_results:
            actions = result['actions']
            m1_action, m2_action, m3_action, m5_action, m4_action = actions
            action_levels = ['low', 'mid', 'high']
            global_action_counts[f'module1_{action_levels[m1_action]}'] += 1
            global_action_counts[f'module2_{action_levels[m2_action]}'] += 1
            global_action_counts[f'module3_{action_levels[m3_action]}'] += 1
            global_action_counts[f'module5_{action_levels[m5_action]}'] += 1
            global_action_counts[f'module4_{action_levels[m4_action]}'] += 1
            sample_action_counts[f'module1_{action_levels[m1_action]}'] += 1
            sample_action_counts[f'module2_{action_levels[m2_action]}'] += 1
            sample_action_counts[f'module3_{action_levels[m3_action]}'] += 1
            sample_action_counts[f'module5_{action_levels[m5_action]}'] += 1
            sample_action_counts[f'module4_{action_levels[m4_action]}'] += 1

        # Print sample-level statistics
        avg_sample_reward = np.mean(sample_rewards) if sample_rewards else 0.0
        print(f"\nSample {data['sample_id']} - Avg Reward: {avg_sample_reward:.4f}, "
              f"Max: {max(sample_rewards):.4f}, Min: {min(sample_rewards):.4f}")

        print(f"\nSample {data['sample_id']} Action Distribution:")
        total_sample_actions = sum(sample_action_counts.values())
        for action, count in sample_action_counts.items():
            ratio = count / max(total_sample_actions, 1)
            print(f"  {action}: {count} ({ratio:.2%})")

        all_pipeline_outputs.extend(sample_pipeline_outputs)
        all_question_results.extend(question_results)

    # Calculate average metrics
    avg_reward = total_reward / max(num_questions, 1)
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m.get(key, 0) for m in all_metrics])

    all_rewards = [result['reward'] for result in all_question_results] if all_question_results else [0]

    # Calculate total actions for each module
    module1_total = (global_action_counts['module1_low'] +
                     global_action_counts['module1_mid'] +
                     global_action_counts['module1_high'])
    module2_total = (global_action_counts['module2_low'] +
                     global_action_counts['module2_mid'] +
                     global_action_counts['module2_high'])
    module3_total = (global_action_counts['module3_low'] +
                     global_action_counts['module3_mid'] +
                     global_action_counts['module3_high'])
    module5_total = (global_action_counts['module5_low'] +
                     global_action_counts['module5_mid'] +
                     global_action_counts['module5_high'])
    module4_total = (global_action_counts['module4_low'] +
                     global_action_counts['module4_mid'] +
                     global_action_counts['module4_high'])

    epoch_log = {
        # === PPO Metrics ===
        'ppo/policy_loss': avg_metrics.get('policy_loss', 0),
        'ppo/value_loss': avg_metrics.get('value_loss', 0),
        'ppo/entropy': avg_metrics.get('entropy', 0),
        'ppo/approx_kl': avg_metrics.get('approx_kl', 0),
        'ppo/clip_fraction': avg_metrics.get('clip_fraction', 0),
        'ppo/grad_norm': avg_metrics.get('grad_norm', 0),
        'ppo/explained_variance': avg_metrics.get('explained_variance', 0),
        'ppo/avg_cost': avg_metrics.get('avg_cost', 0),
        'ppo/num_updates': avg_metrics.get('num_updates', 0),

        # === Training statistics ===
        f'train/{args.llm_judge}_f1_avg_reward': avg_reward,  # Raw F1 average
        'train/avg_cost': np.mean([r['total_cost'] for r in all_question_results]) if all_question_results else 0.0,  # Cost efficiency average
        'train/f1_aligned_avg_reward': np.mean([r.get('aligned_reward', r['reward']) for r in all_question_results]) if all_question_results else 0.0,  # Aligned F1 average
        'train/cost_aligned_avg': np.mean([r.get('aligned_cost', r['total_cost']) for r in all_question_results]) if all_question_results else 0.0,  # Aligned cost average
        'train/rl_train_reward': np.mean([getattr(args, 'reward_weight', 1.0) * r.get('aligned_reward', r['reward']) + getattr(args, 'cost_weight', 0.0) * r.get('aligned_cost', r['total_cost']) for r in all_question_results]) if all_question_results else 0.0,  # RL reward using aligned values
        'train/f1_max_reward': max(all_rewards) if all_rewards else 0.0,
        'train/f1_min_reward': min(all_rewards) if all_rewards else 0.0,
        'train/f1_std_reward': float(np.std(all_rewards)) if all_rewards else 0.0,
        'train/num_questions': num_questions,
        'train/epoch': epoch,

        # === Module statistics ===
        'modules/module1/low_ratio': global_action_counts['module1_low'] / max(module1_total, 1),
        'modules/module1/mid_ratio': global_action_counts['module1_mid'] / max(module1_total, 1),
        'modules/module1/high_ratio': global_action_counts['module1_high'] / max(module1_total, 1),

        # Module 2
        'modules/module2/low_ratio': global_action_counts['module2_low'] / max(module2_total, 1),
        'modules/module2/mid_ratio': global_action_counts['module2_mid'] / max(module2_total, 1),
        'modules/module2/high_ratio': global_action_counts['module2_high'] / max(module2_total, 1),

        # Module 3
        'modules/module3/low_ratio': global_action_counts['module3_low'] / max(module3_total, 1),
        'modules/module3/mid_ratio': global_action_counts['module3_mid'] / max(module3_total, 1),
        'modules/module3/high_ratio': global_action_counts['module3_high'] / max(module3_total, 1),

        # Module 5
        'modules/module5/low_ratio': global_action_counts['module5_low'] / max(module5_total, 1),
        'modules/module5/mid_ratio': global_action_counts['module5_mid'] / max(module5_total, 1),
        'modules/module5/high_ratio': global_action_counts['module5_high'] / max(module5_total, 1),

        # Module 4
        'modules/module4/low_ratio': global_action_counts['module4_low'] / max(module4_total, 1),
        'modules/module4/mid_ratio': global_action_counts['module4_mid'] / max(module4_total, 1),
        'modules/module4/high_ratio': global_action_counts['module4_high'] / max(module4_total, 1),
    }

    # Use global_batch_idx as step, ensure epoch logs display at correct position
    wandb.log(epoch_log, step=global_batch_idx)
    if epoch < 3:
        os.makedirs("./train", exist_ok=True)
        
        failed_questions_file = f"./train/failed_questions_epoch_{epoch}.json"
        report = global_question_tracker.save_failed_questions_report(failed_questions_file, min_failures=2)
        print(f"\nFailed Questions Report (Epoch {epoch}):")
        print(f"  Total failed questions (>=2 consecutive failures): {report['total_failed_questions']}")

    return {'avg_reward': avg_reward, **avg_metrics}, global_batch_idx



def load_hotpotqa_train_data(train_file: str, val_file: str = None, shuffle: bool = True, seed: int = 42) -> List[Dict]:
    """
    Load HotpotQA training dataset, convert each sample to Locomo format

    Args:
        train_file: Training data file path (hotpotqa_7k.json)
        val_file: Validation data file path (optional, if not provided will split from training set)
        shuffle: Whether to shuffle order
        seed: Random seed

    Returns:
        Training sample list (converted to Locomo-compatible format)
    """
    # Load training data
    all_train_samples = json.load(open(train_file))

    # If validation file not specified, split from training set
    if val_file is None:
        # Use last 20% as validation set
        split_idx = int(len(all_train_samples) * 0.8)
        train_samples = all_train_samples[:split_idx]
        val_samples = all_train_samples[split_idx:]
        print(f"Split training data: {len(train_samples)} train, {len(val_samples)} val")
    else:
        train_samples = all_train_samples
        val_samples = json.load(open(val_file))

    # Merge training and validation data for training
    combined_samples = train_samples + val_samples

    if shuffle:
        random.seed(seed)
        random.shuffle(combined_samples)

    # Convert format to Locomo-compatible
    converted_samples = []
    for idx, item in enumerate(combined_samples):
        # Parse documents
        documents = parse_documents_from_context(item["context"])

        # Create Locomo-format sample
        sample = {
            'sample_id': f'hotpotqa_{item["index"]}',
            'conversation': {
                'session_1': [],  # Temporarily empty, will fill later
                'session_1_date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'qa': [{
                'question': item['input'],
                'answer': ', '.join(item['answers']) if isinstance(item['answers'], list) else str(item['answers']),
                'category': 1,  # HotpotQA uniformly category 1 (F1 score)
                'id': f'hotpotqa_{item["index"]}'
            }],
            'documents': documents,  # Save original document list
            'num_docs': item['num_docs']
        }

        converted_samples.append(sample)

    print(f"Loaded {len(converted_samples)} HotpotQA samples")
    print(f"Total questions: {len(converted_samples)}")
    print(f"All questions are Category 1 (F1 score)")

    return converted_samples


def load_hotpotqa_test_data(test_file: str) -> List[Dict]:
    """
    Load HotpotQA test dataset

    Args:
        test_file: Test data file path (eval_200.json)

    Returns:
        Test sample list (converted to Locomo-compatible format)
    """
    test_samples = json.load(open(test_file))

    converted_samples = []
    for item in test_samples:
        # Parse documents
        documents = parse_documents_from_context(item["context"])

        # Create Locomo-format sample
        sample = {
            'sample_id': f'hotpotqa_{item["index"]}',
            'conversation': {
                'session_1': [],  # Temporarily empty
                'session_1_date_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'qa': [{
                'question': item['input'],
                'answer': ', '.join(item['answers']) if isinstance(item['answers'], list) else str(item['answers']),
                'category': 1,  # HotpotQA uniformly category 1
                'id': f'hotpotqa_{item["index"]}'
            }],
            'documents': documents,  # Save original document list
            'num_docs': item['num_docs']
        }

        converted_samples.append(sample)

    print(f"Loaded {len(converted_samples)} HotpotQA test samples")

    return converted_samples


def parse_documents_from_context(context: str) -> List[str]:
    """
    Parse document list from HotpotQA context string

    Args:
        context: String in format "Document 1:
...

Document 2:
..."

    Returns:
        Document content list
    """
    # Split documents
    pattern = r'Document \d+:'
    parts = re.split(pattern, context)
    documents = []
    for part in parts[1:]:  # Skip first empty part
        doc = part.strip()
        if doc:
            documents.append(doc)
    return documents



def main(args):
    from tests.test_utils_hotpotqa import test_on_test_set

    # Initialize scale alignment history
    reset_scale_alignment_history()
    print("Initialized reward-cost scale alignment history")

    # Load modules based on cost-performance strategy
    print(f"Loading modules with cost strategy: {args.cost_strategy}")
    (CostLevel, ModuleOutput, MemoryModuleOutputs,
     Module1_Filter, Module2_EntityRelation, Module3_TemporalRelation, Module5_TopicRelation, Module4_Summary,
     ModularPipelineExecutor) = load_modules(args.cost_strategy)

    print("******************  Training Model %s with Pipeline + PPO ***************" % args.model)
    print(f"Cost Strategy: {args.cost_strategy}")

    import os
    wandb_run_name = os.environ.get("WANDB_RUN_NAME")
    if not wandb_run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1]
        wandb_run_name = f"ppo_{model_short}_{args.retriever}_{args.cost_strategy}_{timestamp}"
    
    print(f"Wandb run name: {wandb_run_name}")

    wandb.init(
        project="hotpotqa-training-expriment",
        name=wandb_run_name,
        config={
            "model": args.model,
            "retriever": args.retriever,
            "lr": 3e-4,
            "clip_epsilon": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "ppo_epochs": 4,
            "num_epochs": args.num_epochs
        }
    )

    train_file = args.data_file or os.environ.get('DATA_FILE', './data/hotpotqa/hotpotqa_7k.json')
    all_samples = load_hotpotqa_train_data(train_file)
    # Randomly select 1000 samples for training
    import random
    random.seed(42)  # For reproducibility
    samples = random.sample(all_samples)
    print(f"Randomly selected {len(samples)} samples for training from {len(all_samples)} total samples")
    context_tokenizer, context_encoder = init_context_model(args.retriever)
    query_tokenizer, query_encoder = init_query_model(args.retriever)
    data_tokenizer, data_encoder = init_data_embedding_model(
        model_name='all-mpnet-base-v2'
    )

    dataset_name = extract_dataset_name(train_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(
        "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/lx/memory/res_code/res_model",
        dataset_name,
        args.cost_strategy,
        timestamp
    )
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"Model save directory: {model_save_dir}")

    # Use data_encoder to initialize module description embeddings from text
    # data_encoder should be a SentenceTransformer model with encode method
    actor_critic = ActorCriticNetwork(
        query_dim=768,
        memory_dim=768,
        desc_dim=768,  # dimension of module description embeddings
        hidden_dim=256,
        projection_dim=256,  # dimension after projection
        num_actions_per_module=3,
        desc_encoder=data_encoder  # Pass encoder to initialize description embeddings from text
    )
    actor_critic.to(DEVICE)

    ppo_trainer = PPOTrainer(
        actor_critic=actor_critic,
        lr=3e-4,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        reward_weight=getattr(args, 'reward_weight', 1.0),
        cost_weight=getattr(args, 'cost_weight', 0.0),
        max_grad_norm=0.5,
        ppo_epochs=4,
        device=DEVICE
    )

    # Fixed to 1 epoch for training
    num_epochs = 1
    print(f"Training with fixed {num_epochs} epoch(s)")
    best_reward = -float('inf')
    global_batch_idx = 0

    # Preprocess all samples' memories before training (only for rule_llm strategy)
    sample_memory_pools = None
    if args.cost_strategy == "rule_llm" and not args.restore_data:
        print(f"\nDataset: {dataset_name}")
        print(f"Cost strategy: {args.cost_strategy}")

        # Try to load cached preprocessed memories
        sample_memory_pools = load_preprocessed_memories(
            dataset_name=dataset_name,
            cost_strategy=args.cost_strategy
        )

        # If not cached, preprocess and save
        if sample_memory_pools is None:
            sample_memory_pools = preprocess_all_samples_memories_hotpotqa(
                samples=samples,
                args=args,
                context_tokenizer=context_tokenizer,
                context_encoder=context_encoder,
                data_tokenizer=data_tokenizer,
                data_encoder=data_encoder,
                Module2_EntityRelation=Module2_EntityRelation,
                Module3_TemporalRelation=Module3_TemporalRelation,
                Module5_TopicRelation=Module5_TopicRelation,
                max_workers=32,  # Use 32 threads for parallel preprocessing
                max_tokens_per_chunk=args.chunk_max_tokens  # 256 tokens per memory chunk
            )

            # Save preprocessed memories for future use
            save_preprocessed_memories(
                sample_memory_pools=sample_memory_pools,
                dataset_name=dataset_name,
                cost_strategy=args.cost_strategy
            )
        else:
            print(f"[Cache] Using cached preprocessed memories - skipping preprocessing!")
            print(f"[Cache] This will save significant time!\n")

    # Preprocess test samples' memories (only for rule_llm strategy)
    test_memory_pools = None
    if args.cost_strategy == "rule_llm" and not args.restore_data:
        test_file = args.test_data_file or os.environ.get('TEST_DATA_FILE', './data/hotpotqa/eval_200.json')
        test_samples = load_hotpotqa_test_data(test_file)

        # Try to load cached preprocessed test memories
        test_memory_pools = load_preprocessed_memories(
            dataset_name=dataset_name,
            cost_strategy=args.cost_strategy,
            split='test'
        )

        # If not cached, preprocess and save
        if test_memory_pools is None:
            test_memory_pools = preprocess_all_samples_memories_hotpotqa(
                samples=test_samples,
                args=args,
                context_tokenizer=context_tokenizer,
                context_encoder=context_encoder,
                data_tokenizer=data_tokenizer,
                data_encoder=data_encoder,
                Module2_EntityRelation=Module2_EntityRelation,
                Module3_TemporalRelation=Module3_TemporalRelation,
                Module5_TopicRelation=Module5_TopicRelation,
                max_workers=32,
                max_tokens_per_chunk=args.chunk_max_tokens
            )

            # Save preprocessed test memories
            save_preprocessed_memories(
                sample_memory_pools=test_memory_pools,
                dataset_name=dataset_name,
                cost_strategy=args.cost_strategy,
                split='test'
            )
        else:
            print(f"[Cache] Using cached preprocessed test memories - skipping preprocessing!\n")

    for epoch in range(num_epochs):
        print(f"\n========== Epoch {epoch + 1}/{num_epochs} ==========")
        
        epoch_samples = samples.copy()  # Shallow copy is faster than deepcopy
        random.seed(42 + epoch)  # Different seed for each epoch
        random.shuffle(epoch_samples)
        print(f"Shuffled {len(epoch_samples)} samples for epoch {epoch + 1}")

        metrics, global_batch_idx = train_epoch(
            epoch_samples, actor_critic, ppo_trainer,
            context_tokenizer, context_encoder,
            query_tokenizer, query_encoder, args,
            Module1_Filter, Module2_EntityRelation, Module3_TemporalRelation, Module5_TopicRelation, Module4_Summary,
            ModularPipelineExecutor,
            data_tokenizer, data_encoder, epoch, global_batch_idx,
            sample_memory_pools=sample_memory_pools,  # Pass preprocessed memory pools
            model_save_dir=model_save_dir  # Pass model save directory for step-level checkpoints
        )

        avg_reward = metrics.get('avg_reward', 0)
        print(f"Epoch {epoch + 1} - Avg Reward: {avg_reward:.4f}")
        print(f"Metrics: {metrics}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_path = os.path.join(model_save_dir, f"best_model_{args.cost_strategy}.pt")
            torch.save({
                'epoch': epoch + 1,
                'num_actions_per_module': int(getattr(actor_critic, 'num_actions_per_module', 4)),
                'model_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': ppo_trainer.optimizer.state_dict(),
                'avg_reward': avg_reward,
                'cost_strategy': args.cost_strategy
            }, best_path)
            print(f"New best model saved to {best_path}")
            # Use global_batch_idx as step, consistent with epoch logs
            wandb.log({'best_reward': best_reward, 'best_epoch': epoch + 1}, step=global_batch_idx)

        # Save checkpoint every 5 epochs, or if it is the last epoch also save
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{epoch + 1}_{args.cost_strategy}.pt")
            torch.save({
                'epoch': epoch + 1,
                'num_actions_per_module': int(getattr(actor_critic, 'num_actions_per_module', 4)),
                'model_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': ppo_trainer.optimizer.state_dict(),
                'metrics': metrics,
                'cost_strategy': args.cost_strategy
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

    print("\n========== Training Completed ==========")
    print(f"Best Reward: {best_reward:.4f}")

    # Save last epoch model
    final_model_path = os.path.join(model_save_dir, f"final_model_{args.cost_strategy}.pt")
    torch.save({
        'epoch': num_epochs,
        'step': global_batch_idx,
        'num_actions_per_module': int(getattr(actor_critic, 'num_actions_per_module', 4)),
        'model_state_dict': actor_critic.state_dict(),
        'optimizer_state_dict': ppo_trainer.optimizer.state_dict(),
        'final_reward': metrics.get('avg_reward', 0) if 'metrics' in locals() else 0,
        'cost_strategy': args.cost_strategy
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    all_test_results = {}

    print("\n" + "="*80)
    print("Testing Best Step Model")
    print("="*80)
    # Reset statistics, ensure best step model test statistics are independent
    reset_llm_api_stats()
    # Prefer checkpoint_step_best, if not available then find all step checkpoints
    import glob
    best_step_model_path = os.path.join(model_save_dir, f"checkpoint_step_best_{args.cost_strategy}.pt")
    if not os.path.exists(best_step_model_path):
        # Fallback: find all step checkpoints, select latest
        best_step_model_files = glob.glob(os.path.join(model_save_dir, f"checkpoint_step_*_{args.cost_strategy}.pt"))
        # Exclude best checkpoint, only select those with step numbers
        best_step_model_files = [f for f in best_step_model_files if 'checkpoint_step_best' not in f]
        if best_step_model_files:
            # Sort by step number, select latest
            def extract_step_number(filepath):
                match = re.search(r'checkpoint_step_(\d+)_', filepath)
                return int(match.group(1)) if match else 0
            best_step_model_files.sort(key=extract_step_number, reverse=True)
            best_step_model_path = best_step_model_files[0]
        else:
            best_step_model_path = None

    if best_step_model_path and os.path.exists(best_step_model_path):
        test_results_best_step = test_on_test_set(
            args,
            context_tokenizer, context_encoder,
            query_tokenizer, query_encoder,
            data_tokenizer, data_encoder,
            model_path=best_step_model_path,
            Module1_Filter=Module1_Filter,
            Module2_EntityRelation=Module2_EntityRelation,
            Module3_TemporalRelation=Module3_TemporalRelation,
            Module5_TopicRelation=Module5_TopicRelation,
            Module4_Summary=Module4_Summary,
            ModularPipelineExecutor=ModularPipelineExecutor,
            test_memory_pools=test_memory_pools
        )
        all_test_results['best_step'] = test_results_best_step

        # Save best step model test LLM API statistics
        llm_stats = get_llm_api_stats()
        llm_stats['timestamp'] = datetime.now().isoformat()
        llm_stats['test_phase'] = 'best_step'
        llm_stats['model_path'] = best_step_model_path
        llm_stats['model_save_dir'] = model_save_dir
        llm_stats['avg_f1'] = test_results_best_step['avg_f1']
        llm_stats['avg_llm_judge'] = test_results_best_step.get('avg_llm_judge', 0.0)
        llm_stats['category_performance_f1'] = test_results_best_step['category_performance_f1']
        llm_stats['category_performance_llm_judge'] = test_results_best_step['category_performance_llm_judge']
        stats_path = os.path.join(model_save_dir, 'llm_api_stats_best_step.json')
        print(llm_stats)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(llm_stats, f, indent=2, ensure_ascii=False)
        print(f"\nBest Step Model Test - LLM API stats saved to {stats_path}")
        print(f"  Total API calls: {llm_stats['total_api_calls']}")
        print(f"  Total input tokens: {llm_stats['total_input_tokens']:,}")
        print(f"  Total output tokens: {llm_stats['total_output_tokens']:,}")
        print(f"  Total tokens: {llm_stats['total_tokens']:,}")
        print(f"  Total cost: ${llm_stats['total_cost_usd']:.4f}")

        wandb.log({
            'test_best_step/avg_f1': test_results_best_step['avg_f1'],
            'test_best_step/avg_llm_judge': test_results_best_step.get('avg_llm_judge', 0.0),
            'test_best_step/num_questions': test_results_best_step['num_questions'],
            **{f'test_best_step/{k}': v for k, v in test_results_best_step['action_counts'].items()}
        }, step=num_epochs + 1)
    else:
        print(f"Warning: Best step model file not found in {model_save_dir}")

    print("\n" + "="*80)
    print("Testing Last Epoch Model")
    print("="*80)
    # Reset statistics, ensure last epoch test statistics are independent
    reset_llm_api_stats()       
    last_epoch_model_path = os.path.join(model_save_dir, f"checkpoint_epoch_{num_epochs}_{args.cost_strategy}.pt")
    if os.path.exists(last_epoch_model_path):
        test_results_last = test_on_test_set(
            args,
            context_tokenizer, context_encoder,
            query_tokenizer, query_encoder,
            data_tokenizer, data_encoder,
            model_path=last_epoch_model_path,
            Module1_Filter=Module1_Filter,
            Module2_EntityRelation=Module2_EntityRelation,
            Module3_TemporalRelation=Module3_TemporalRelation,
            Module5_TopicRelation=Module5_TopicRelation,
            Module4_Summary=Module4_Summary,
            ModularPipelineExecutor=ModularPipelineExecutor,
            test_memory_pools=test_memory_pools
        )
        all_test_results['last_epoch'] = test_results_last
        
        # Save last epoch test LLM API statistics
        llm_stats = get_llm_api_stats()
        llm_stats['timestamp'] = datetime.now().isoformat()
        llm_stats['test_phase'] = 'last_epoch'
        llm_stats['model_path'] = last_epoch_model_path
        llm_stats['model_save_dir'] = model_save_dir
        llm_stats['avg_f1'] = test_results_last['avg_f1']
        llm_stats['avg_llm_judge'] = test_results_last.get('avg_llm_judge', 0.0)
        llm_stats['category_performance_f1'] = test_results_last['category_performance_f1']
        llm_stats['category_performance_llm_judge'] = test_results_last['category_performance_llm_judge']
        stats_path = os.path.join(model_save_dir, 'llm_api_stats_last_epoch.json')
        print(llm_stats)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(llm_stats, f, indent=2, ensure_ascii=False)
        print(f"\nLast Epoch Test - LLM API stats saved to {stats_path}")
        print(f"  Total API calls: {llm_stats['total_api_calls']}")
        print(f"  Total input tokens: {llm_stats['total_input_tokens']:,}")
        print(f"  Total output tokens: {llm_stats['total_output_tokens']:,}")
        print(f"  Total tokens: {llm_stats['total_tokens']:,}")
        print(f"  Total cost: ${llm_stats['total_cost_usd']:.4f}")
        
        wandb.log({
            'test_last/avg_f1': test_results_last['avg_f1'],
            'test_last/avg_llm_judge': test_results_last.get('avg_llm_judge', 0.0),
            'test_last/num_questions': test_results_last['num_questions'],
            **{f'test_last/{k}': v for k, v in test_results_last['action_counts'].items()}
        }, step=num_epochs + 1)
    else:
        print(f"Warning: Last epoch model file not found at {last_epoch_model_path}")
    
    if 'best_step' in all_test_results and 'last_epoch' in all_test_results:
        print("\n" + "="*80)
        print("Model Comparison Summary")
        print("="*80)
        best_step_f1 = all_test_results['best_step']['avg_f1']
        last_f1 = all_test_results['last_epoch']['avg_f1']

        best_step_llm = all_test_results['best_step'].get('avg_llm_judge', 0.0)
        last_llm = all_test_results['last_epoch'].get('avg_llm_judge', 0.0)

        print(f"Best Step Model   - F1: {best_step_f1:.4f}, LLM-Judge: {best_step_llm:.4f}")
        print(f"Last Epoch Model  - F1: {last_f1:.4f}, LLM-Judge: {last_llm:.4f}")
        print()

        # Display tokens and cost statistics (if available)
        # Here displays last test statistics, need to separately load each model statistics file
        print("Token and Cost Statistics (from saved files):")
        print("Note: Statistics shown below are from the last test run")
        print()

        # Calculate difference (Best Step vs Last Epoch)
        diff_step_vs_last_f1 = best_step_f1 - last_f1
        diff_step_vs_last_llm = best_step_llm - last_llm

        print(f"Difference (Best Step vs Last Epoch):")
        print(f"  F1: {diff_step_vs_last_f1:+.4f}, LLM-Judge: {diff_step_vs_last_llm:+.4f}")
        print("="*80)

        wandb.log({
            'test_comparison/best_step_f1': best_step_f1,
            'test_comparison/last_epoch_f1': last_f1,
            'test_comparison/diff_step_vs_last_f1': diff_step_vs_last_f1,
            'test_comparison/best_step_llm_judge': best_step_llm,
            'test_comparison/last_epoch_llm_judge': last_llm,
            'test_comparison/diff_step_vs_last_llm': diff_step_vs_last_llm
        }, step=num_epochs + 1)

    wandb.finish()

if __name__ == '__main__':
    args = get_locomo_args()
    print(args)
    main(args=args)
