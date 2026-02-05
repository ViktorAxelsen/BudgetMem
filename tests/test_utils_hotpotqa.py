import os
import sys
import json
import pickle
import copy
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.utils.llm_utils import get_llm_response, get_llm_response_via_api
from src.utils.rag_utils import get_embeddings_with_model, get_data_embeddings
from src.utils.eval_utils import f1_score, f1_max, parse_judge_response
from src.prompts.prompt_pool import (
    QA_PROMPT, LLM_JUDGE_GENERAL_PROMPT,HOTPOTQA_ANSWER_PROMPT
)
from src.trainer.ppo_trainer import ActorCriticNetwork

def _infer_num_actions_per_module_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    default: int = 4,
) -> int:
    """Infer action dimension from a saved ActorCriticNetwork state_dict.
    
    Supports both:
    - New unified architecture: action_head.2.bias/weight
    - Old multi-head architecture: module1_head.2.bias/weight, etc.
    """
    # First, try new unified architecture (action_head)
    preferred_keys_new = [
        "action_head.2.bias",  # Last layer bias: shape = [num_actions_per_module]
        "action_head.2.weight",  # Last layer weight: shape = [num_actions_per_module, hidden_dim//2]
    ]
    for key in preferred_keys_new:
        tensor = state_dict.get(key)
        if tensor is not None:
            try:
                # For bias: shape is [num_actions]
                # For weight: shape is [num_actions, hidden_dim//2]
                if len(tensor.shape) >= 1:
                    return int(tensor.shape[0])
            except Exception:
                continue
    
    # Fallback: try old multi-head architecture (for backward compatibility)
    preferred_keys_old = [
        "module1_head.2.bias",
        "module1_head.2.weight",
        "module2_head.2.bias",
        "module2_head.2.weight",
        "module3_head.2.bias",
        "module3_head.2.weight",
        "module4_head.2.bias",
        "module4_head.2.weight",
    ]
    for key in preferred_keys_old:
        tensor = state_dict.get(key)
        if tensor is not None:
            try:
                return int(tensor.shape[0])
            except Exception:
                continue

    # Final fallback: search for action_head or module*_head patterns
    for key, tensor in state_dict.items():
        if not hasattr(tensor, "shape") or len(tensor.shape) == 0:
            continue
        if key.startswith("action_head.") and len(tensor.shape) >= 1:
            first_dim = int(tensor.shape[0])
            if 2 <= first_dim <= 16:
                return first_dim
        if key.startswith("module") and "head" in key and len(tensor.shape) >= 1:
            first_dim = int(tensor.shape[0])
            if 2 <= first_dim <= 16:
                return first_dim

    return int(default)


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
    judge_args.max_new_tokens = args.max_new_tokens
    judge_args.model = "openai/gpt-oss-120b"
    judge_args.temperature = 0.0
    judge_args.api = True

    # Call LLM judge
    try:
        response, _, _ = get_llm_response_via_api(
            prompt=judge_prompt,
            MAX_TOKENS=judge_args.max_new_tokens,
            LLM_MODEL=judge_args.model,
            TAU=judge_args.temperature,
            base_url=judge_args.api_base,
            api_key=judge_args.api_key,
        )

        if response is None:
            print(f"[LLM Judge] API returned None, returning 0.0")
            return 0.0

        # Parse judge response
        reward = parse_judge_response(response)
        return reward
    except Exception as e:
        print(f"[LLM Judge] Error: {e}, returning 0.0")
        return 0.0


def test_on_test_set(
    args,
    context_tokenizer,
    context_encoder,
    query_tokenizer,
    query_encoder,
    data_tokenizer=None,
    data_encoder=None,
    model_path: str = "./checkpoints/best_model.pt",
    Module1_Filter=None,
    Module2_EntityRelation=None,
    Module3_TemporalRelation=None,
    Module5_TopicRelation=None,
    Module4_Summary=None,
    ModularPipelineExecutor=None,
    test_memory_pools=None
):
    """
    Evaluate using trained model on test set.

    Testing strategy:
    - Execute pipeline for each question (retrieval, filter, Entity relations, Temporal relations, Summary)
    - Use pipeline output information to build context and answer questions
    - Evaluate answer quality using F1-score or LLM Judge (controlled by args.llm_judge)
    - Do not write summary memory (consistent with training code)

    Saved memory file format:
    - Path: ./data/{dataset_prefix}_memory_{sample_id}.pkl
    - Format: {
        'embeddings': np.ndarray,  # [N, D]
        'date_time': List[str],
        'dia_id': List[str],       # Format: "S{session_id}:original" (only contains original session chunks)
        'context': List[str]
      }
    - Can be loaded in eval_locomo.py using --rag_mode memory

    Args:
        args: Configuration arguments
        context_tokenizer: Context tokenizer
        context_encoder: Context encoder
        query_tokenizer: Query tokenizer
        query_encoder: Query encoder
        data_tokenizer: Data embedding tokenizer (optional)
        data_encoder: Data embedding encoder (optional)
        model_path: Trained model path
        
        test_memory_pools: Preprocessed test set memory pools dictionary (optional, if provided will be used to avoid repeated construction)

    Returns:
        test_results: Dictionary containing all test results
    """
    # Lazy import to avoid circular imports
    from train.train_hotpotqa import (
        DEVICE,
        QuestionTrackingRecord,
        global_question_tracker,
        construct_global_memory_hotpotqa,
        load_hotpotqa_test_data
    )

    print("\n" + "="*80)
    print("Testing on HotpotQA Test Set")
    print("="*80)

    import os
    test_file = getattr(args, 'test_data_file', None) or os.environ.get('TEST_DATA_FILE', './data/hotpotqa/eval_200.json')
    test_samples = load_hotpotqa_test_data(test_file)
    print(f"Loaded {len(test_samples)} HotpotQA test samples")
    print(f"All questions are Category 1 (F1 score)")
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    checkpoint_cost_strategy = checkpoint.get('cost_strategy', None)
    if checkpoint_cost_strategy is not None:
        if checkpoint_cost_strategy != getattr(args, 'cost_strategy', None):
            print(f"Warning: Checkpoint cost_strategy ({checkpoint_cost_strategy}) differs from args.cost_strategy ({getattr(args, 'cost_strategy', None)})")
            print(f"Using checkpoint cost_strategy: {checkpoint_cost_strategy}")
        args.cost_strategy = checkpoint_cost_strategy
    else:
        # Use args.cost_strategy or default
        if not hasattr(args, 'cost_strategy') or args.cost_strategy is None:
            args.cost_strategy = 'rule_llm'  # Default fallback
            print(f"Warning: No cost_strategy found in checkpoint, using default: {args.cost_strategy}")
    
    # Load modules with correct cost_strategy (after determining from checkpoint)
    if Module1_Filter is None or Module5_TopicRelation is None or Module2_EntityRelation is None or Module3_TemporalRelation is None or Module4_Summary is None or ModularPipelineExecutor is None:
        from train.train_hotpotqa import load_modules
        (_, _, _, Module1_Filter, Module2_EntityRelation,
         Module3_TemporalRelation, Module5_TopicRelation,
         Module4_Summary, ModularPipelineExecutor) = load_modules(
            args.cost_strategy)
    
    num_actions_per_module = checkpoint.get('num_actions_per_module', None)
    if num_actions_per_module is None:
        num_actions_per_module = _infer_num_actions_per_module_from_state_dict(state_dict, default=4)
    else:
        num_actions_per_module = int(num_actions_per_module)

    actor_critic = ActorCriticNetwork(
        query_dim=768,
        memory_dim=768,
        desc_dim=768,  # dimension of module description embeddings
        hidden_dim=256,
        projection_dim=256,  # dimension after projection
        num_actions_per_module=num_actions_per_module,
        desc_encoder=data_encoder  # Use same encoder as training to initialize description embeddings
    )
    actor_critic.load_state_dict(state_dict)
    actor_critic.to(DEVICE)
    actor_critic.eval()
    print(f"Loaded model from {model_path} (num_actions_per_module={num_actions_per_module})")

    # Create pipeline (consistent with training code)
    # General Modules (existing)
    module1 = Module1_Filter(
        encoder=data_encoder,
        args=args,
        top_k=args.module_top_k
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

    all_results = []
    total_reward_f1 = 0.0
    total_reward_llm_judge = 0.0
    num_questions = 0
    action_counts = {
        'module1_low': 0, 'module1_mid': 0, 'module1_high': 0,
        'module2_low': 0, 'module2_mid': 0, 'module2_high': 0,
        'module3_low': 0, 'module3_mid': 0, 'module3_high': 0,
        'module4_low': 0, 'module4_mid': 0, 'module4_high': 0,
        'module5_low': 0, 'module5_mid': 0, 'module5_high': 0,
    }
    # Statistics by category (F1 and LLM Judge separately, HotpotQA only has category 1)
    category_stats_f1 = {
        1: {'rewards': [], 'count': 0, 'total_reward': 0.0},
    }
    category_stats_llm_judge = {
        1: {'rewards': [], 'count': 0, 'total_reward': 0.0},
    }

    # Define function to process single sample
    def process_single_sample(sample_idx, data):
        """
        Process single test sample complete pipeline

        Args:
            sample_idx: Sample index
            data: Sample data

        Returns: Dictionary containing all necessary information
        """
        # Step 1: Get global memory pool (use preprocessed test_memory_pools if available)
        sample_id = data['sample_id']
        if test_memory_pools and sample_id in test_memory_pools:
            # Use preprocessed memory pool
            global_memory = test_memory_pools[sample_id]
            print(f"Using preprocessed global memory with {len(global_memory)} chunks for sample {sample_id}")
        else:
            # Fallback: construct global memory pool (split into chunks by args.chunk_max_tokens)
            global_memory = construct_global_memory_hotpotqa(
                data, args.retriever, context_tokenizer, context_encoder, args,
                data_tokenizer, data_encoder,
                max_tokens_per_chunk=args.chunk_max_tokens
            )
            print(f"Constructed global memory with {len(global_memory)} chunks for sample {sample_id}")

        # Step 2: Prepare questions
        out_data = {'sample_id': data['sample_id'], 'qa': copy.deepcopy(data['qa'])}
        questions = []
        for qa in data['qa']:
            questions.append(qa['question'])

        # Step 3: Batch compute question embeddings
        if data_encoder is not None:
            all_query_embs = get_data_embeddings(data_encoder, questions)
        else:
            all_query_embs = get_embeddings_with_model(args.retriever, questions, query_tokenizer, query_encoder)

        # Step 4: Process single question (HotpotQA each sample has only one question)
        i = 0  # Question index
        qa = data['qa'][i]
        question = questions[i]
        query_emb_np = all_query_embs[i]

        if query_emb_np.ndim > 1:
            query_emb_np = query_emb_np.flatten()

        query_emb = torch.from_numpy(query_emb_np).float()

        # 1: Use question as query to retrieve relevant sessions
        effective_top_k = max(args.top_k, 10)
        retrieved_memories = global_memory.retrieve(query_emb_np, top_k=effective_top_k)

        # 2: Execute pipeline（Use deterministic=True to ensure reproducibility）
        pipeline_result = pipeline.execute(
            query=question,
            query_emb=query_emb,
            initial_memories=retrieved_memories,
            deterministic=True  # Use deterministic mode during testing
        )

        filtered_memories = pipeline_result['filtered_memories']
        entity_relations = pipeline_result['entity_relations']
        temporal_relations = pipeline_result['temporal_relations']
        topic_relations = pipeline_result.get('topic_relations', [])
        summary = pipeline_result['summary']
        actions = pipeline_result['actions']  # (m1, m2, m3, m5, m4)
        total_cost_q = pipeline_result['total_cost']

        # 3: Build context and prompt (consistent with training logic)
        context_parts = []

        if summary:
            context_parts.append("\n<Summary>")
            context_parts.append(summary)
            context_parts.append("</Summary>")

        query_context = '\n'.join(context_parts)
        # 4: Build prompt
        input_prompt = HOTPOTQA_ANSWER_PROMPT.format(
                    context=query_context, question=question
                )

        # 5: Get LLM answer
        disable_threading = getattr(args, 'parallel_questions', 1) > 1
        task_args = [(i, input_prompt, args)]
        ret = get_llm_response(args=args, task_args=task_args, disable_internal_threading=disable_threading)

        # 6: Calculate both F1 and LLM Judge evaluation methods simultaneously
        answer = str(out_data['qa'][i]['answer'])

        prediction = ""
        reward_f1 = 0.0
        reward_llm_judge = 0.0
        if len(ret) > 0:
            idx, response, _, success = ret[0]
            if success:
                prediction = response.strip()
                # Parse answer from <answer> tags if present
                if "<answer>" in prediction and "</answer>" in prediction:
                    start = prediction.find("<answer>") + len("<answer>")
                    end = prediction.find("</answer>", start)
                    if end > start:
                        prediction = prediction[start:end].strip()
                out_data['qa'][i]['prediction'] = prediction

                # Calculate F1 score (HotpotQA all category 1, use f1_max)
                reward_f1 = f1_score(prediction, answer)

                # Calculate LLM Judge score
                try:
                    judge_question = qa['question']
                    reward_llm_judge = get_llm_judge_reward(
                        question=judge_question,
                        ground_truth=answer,
                        prediction=prediction,
                        args=args
                    )
                except Exception as e:
                    print(f"[Warning] LLM Judge failed for Q{i+1}: {e}, using 0.0")
                    reward_llm_judge = 0.0

        print(f"Sample {sample_idx+1}: F1={reward_f1:.4f}, LLM-Judge={reward_llm_judge:.4f}")

        # Record to question tracker (for statistics on consecutively failed questions)
        retrieved_memory_contents = []
        for mem in retrieved_memories:
            if hasattr(mem, 'get_enriched_content'):
                retrieved_memory_contents.append(mem.get_enriched_content())
            else:
                content = mem.content
                if hasattr(mem, 'date_time') and mem.date_time:
                    content = f"[Date: {mem.date_time}] {content}"
                retrieved_memory_contents.append(content)

        filtered_memory_contents = []
        for mem in filtered_memories:
            if hasattr(mem, 'get_enriched_content'):
                filtered_memory_contents.append(mem.get_enriched_content())
            else:
                content = mem.content
                if hasattr(mem, 'date_time') and mem.date_time:
                    content = f"[Date: {mem.date_time}] {content}"
                filtered_memory_contents.append(content)

        # Create tracking record (epoch set to -1 during testing, use F1 as main reward)
        tracking_record = QuestionTrackingRecord(
            sample_id=data['sample_id'],
            question_index=i,
            question=question,
            ground_truth=answer,
            prediction=prediction,
            reward=reward_f1,  # Use F1 as main reward
            epoch=-1,
            retrieved_memories=retrieved_memory_contents,
            filtered_memories=filtered_memory_contents,
            entity_relations=entity_relations,
            temporal_relations=temporal_relations,
            topic_relations=topic_relations,
            summary=summary,
            final_prompt=input_prompt,
            actions=actions
        )
        global_question_tracker.add_record(tracking_record)

        # Return sample results
        m1_action, m2_action, m3_action, m5_action, m4_action = actions
        sample_result = {
            'sample_id': data['sample_id'],
            'question_index': i,
            'question': question,
            'category': qa['category'],
            'ground_truth': out_data['qa'][i]['answer'],
            'prediction': prediction,
            'f1_score': reward_f1,
            'llm_judge_score': reward_llm_judge,
            'reward': reward_f1,  # Maintain backward compatibility, use F1 as main reward
            'actions': {
                'module1': ['low', 'mid', 'high'][m1_action],
                'module2': ['low', 'mid', 'high'][m2_action],
                'module3': ['low', 'mid', 'high'][m3_action],
                'module4': ['low', 'mid', 'high'][m4_action],
                'module5': ['low', 'mid', 'high'][m5_action]
            },
            'retrieved_sessions_count': len(retrieved_memories),
            'filtered_memories_count': len(filtered_memories),
            'qa_data': out_data['qa']
        }

        return sample_result

    # Process all test samples in parallel
    total_samples = len(test_samples)
    parallel_questions = getattr(args, 'parallel_questions', 1)

    if parallel_questions > 1:
        print(f"Using parallel processing with {parallel_questions} workers")
        sample_batch_size = parallel_questions
        num_batches = (total_samples + sample_batch_size - 1) // sample_batch_size

        sample_results = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * sample_batch_size
            end_idx = min(start_idx + sample_batch_size, total_samples)
            batch_samples = list(range(start_idx, end_idx))

            # Process current batch samples in parallel
            batch_results = []
            if len(batch_samples) > 1:
                with ThreadPoolExecutor(max_workers=parallel_questions) as executor:
                    future_to_sample = {}
                    for sample_idx in batch_samples:
                        data = test_samples[sample_idx]
                        future = executor.submit(process_single_sample, sample_idx, data)
                        future_to_sample[future] = sample_idx

                    batch_results_dict = {}
                    for future in as_completed(future_to_sample):
                        try:
                            result = future.result()
                            batch_results_dict[result['sample_id']] = result
                        except Exception as e:
                            sample_idx = future_to_sample[future]
                            print(f"Error processing sample {sample_idx}: {e}")
                            import traceback
                            traceback.print_exc()

                    batch_results = [batch_results_dict[result['sample_id']] for result in batch_results_dict.values()]
            else:
                # Single sample batch, process directly
                for sample_idx in batch_samples:
                    data = test_samples[sample_idx]
                    result = process_single_sample(sample_idx, data)
                    batch_results.append(result)

            sample_results.extend(batch_results)
    else:
        # Sequential processing (when parallel_questions <= 1)
        print("Using sequential processing")
        sample_results = []
        for sample_idx in range(total_samples):
            data = test_samples[sample_idx]
            result = process_single_sample(sample_idx, data)
            sample_results.append(result)

    # Process each test sample results
    for result in sample_results:
        # Accumulate statistics
        total_reward_f1 += result['f1_score']
        total_reward_llm_judge += result['llm_judge_score']
        num_questions += 1

        # Statistics by category
        category = result['category']
        if category in category_stats_f1:
            category_stats_f1[category]['rewards'].append(result['f1_score'])
            category_stats_f1[category]['count'] += 1
            category_stats_f1[category]['total_reward'] += result['f1_score']
        if category in category_stats_llm_judge:
            category_stats_llm_judge[category]['rewards'].append(result['llm_judge_score'])
            category_stats_llm_judge[category]['count'] += 1
            category_stats_llm_judge[category]['total_reward'] += result['llm_judge_score']

        # Accumulate action counts
        actions = result['actions']
        action_counts[f'module1_{actions["module1"]}'] += 1
        action_counts[f'module2_{actions["module2"]}'] += 1
        action_counts[f'module3_{actions["module3"]}'] += 1
        action_counts[f'module4_{actions["module4"]}'] += 1
        action_counts[f'module5_{actions["module5"]}'] += 1

        # Print sample-level statistics
        print(f"{'='*60}")
        print(f"Sample {result['sample_id']} Results:")
        print(f"{'='*60}")
        print(f"  F1: {result['f1_score']:.4f}")
        print(f"  LLM-Judge: {result['llm_judge_score']:.4f}")
        print(f"{'='*60}")

        all_results.append({
            'sample_id': result['sample_id'],
            'avg_f1': result['f1_score'],
            'avg_llm_judge': result['llm_judge_score'],
            'questions': [result],
            'qa_data': result['qa_data'],
            'saved_memory_path': None  # Do not save memory file during parallel processing
            
        })

    # Calculate overall statistics (F1 and LLM Judge)
    avg_test_reward_f1 = total_reward_f1 / max(num_questions, 1)
    avg_test_reward_llm_judge = total_reward_llm_judge / max(num_questions, 1)
    category_performance_f1 = {}
    category_performance_llm_judge = {}
    for cat in [1]:
        # F1 statistics (HotpotQA only has category 1, use f1_score)
        stats_f1 = category_stats_f1[cat]
        if stats_f1['count'] > 0:
            avg_reward = stats_f1['total_reward'] / stats_f1['count']
            category_performance_f1[cat] = {
                'avg_reward': float(avg_reward),
                'total_reward': float(stats_f1['total_reward']),
                'count': stats_f1['count'],
                'min_reward': float(min(stats_f1['rewards'])) if stats_f1['rewards'] else 0.0,
                'max_reward': float(max(stats_f1['rewards'])) if stats_f1['rewards'] else 0.0,
                'std_reward': float(np.std(stats_f1['rewards'])) if len(stats_f1['rewards']) > 1 else 0.0
            }
        else:
            category_performance_f1[cat] = {
                'avg_reward': 0.0, 'total_reward': 0.0, 'count': 0,
                'min_reward': 0.0, 'max_reward': 0.0, 'std_reward': 0.0
            }
        
        # LLM Judge statistics (HotpotQA only has category 1, use llm_judge)
        stats_llm = category_stats_llm_judge[cat]
        if stats_llm['count'] > 0:
            avg_reward = stats_llm['total_reward'] / stats_llm['count']
            category_performance_llm_judge[cat] = {
                'avg_reward': float(avg_reward),
                'total_reward': float(stats_llm['total_reward']),
                'count': stats_llm['count'],
                'min_reward': float(min(stats_llm['rewards'])) if stats_llm['rewards'] else 0.0,
                'max_reward': float(max(stats_llm['rewards'])) if stats_llm['rewards'] else 0.0,
                'std_reward': float(np.std(stats_llm['rewards'])) if len(stats_llm['rewards']) > 1 else 0.0
            }
        else:
            category_performance_llm_judge[cat] = {
                'avg_reward': 0.0, 'total_reward': 0.0, 'count': 0,
                'min_reward': 0.0, 'max_reward': 0.0, 'std_reward': 0.0
            }
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)
    print(f"Testing Strategy:")
    print(f"  - Execute pipeline for each question")
    print(f"  - Evaluate with BOTH F1-score and LLM Judge")
    print(f"  - Do NOT add summary memories to memory bank (consistent with training)")
    print(f"\nOverall Results:")
    print(f"  Total Questions: {num_questions}")
    print(f"  Average F1 Score: {avg_test_reward_f1:.4f}")
    print(f"  Average LLM Judge Score: {avg_test_reward_llm_judge:.4f}")
    print(f"\nCategory Performance (F1 Score):")
    for cat in sorted(category_performance_f1.keys()):
        perf = category_performance_f1[cat]
        if perf['count'] > 0:
            print(f"  Category {cat}: {perf['count']} questions, "
                  f"Avg F1: {perf['avg_reward']:.4f} "
                  f"(Min: {perf['min_reward']:.4f}, Max: {perf['max_reward']:.4f}, "
                  f"Std: {perf['std_reward']:.4f})")
        else:
            print(f"  Category {cat}: 0 questions")
    print(f"\nCategory Performance (LLM Judge):")
    for cat in sorted(category_performance_llm_judge.keys()):
        perf = category_performance_llm_judge[cat]
        if perf['count'] > 0:
            print(f"  Category {cat}: {perf['count']} questions, "
                  f"Avg LLM-Judge: {perf['avg_reward']:.4f} "
                  f"(Min: {perf['min_reward']:.4f}, Max: {perf['max_reward']:.4f}, "
                  f"Std: {perf['std_reward']:.4f})")
        else:
            print(f"  Category {cat}: 0 questions")
    print(f"\nAction Distribution:")
    total_actions = sum(action_counts.values())
    for action, count in action_counts.items():
        ratio = count / max(total_actions, 1)
        print(f"  {action}: {count} ({ratio:.2%})")

    # Save test results
    os.makedirs("./test_results", exist_ok=True)
    
    return {
        'avg_f1': avg_test_reward_f1,
        'avg_llm_judge': avg_test_reward_llm_judge,
        'num_questions': num_questions,
        'action_counts': action_counts,
        'category_performance_f1': category_performance_f1,
        'category_performance_llm_judge': category_performance_llm_judge,
        'sample_results': all_results
    }
