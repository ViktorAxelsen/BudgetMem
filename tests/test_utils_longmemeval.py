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
from datetime import datetime
from src.utils.llm_utils import get_llm_response, get_llm_response_via_api
from src.utils.rag_utils import get_embeddings_with_model, get_data_embeddings
from src.utils.eval_utils import f1_score, f1_max, parse_judge_response, compute_bleu, compute_rouge_l    
from src.prompts.prompt_pool import (
    LONGMEMEVAL_ANSWER_PROMPT_COT, LONGMEMEVAL_ANSWER_PROMPT, LLM_JUDGE_GENERAL_PROMPT
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
            api_key=judge_args.api_key[0] if isinstance(judge_args.api_key, list) else judge_args.api_key
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
        Module1_Filter, Module2_EntityRelation, Module3_TemporalRelation, Module4_Summary, ModularPipelineExecutor:
            Module classes loaded based on cost strategy
        test_memory_pools: Preprocessed test set memory pools dictionary (optional, if provided will be used to avoid repeated construction)

    Returns:
        test_results: Dictionary containing all test results
    """
    # Lazy import to avoid circular imports
    from train.train_longmemeval import (
        DEVICE,
        QuestionTrackingRecord,
        global_question_tracker,
        construct_global_memory
    )

    print("\n" + "="*80)
    print("Testing on Test Set")
    print("="*80)

    # Load test data (use splits file, consistent with training code)
    all_samples = json.load(open(args.data_file))
    splits_file = os.path.join(
        os.path.dirname(args.data_file),
        'longmemeval_s_splits.json'
    )
    if os.path.exists(splits_file):
        splits = json.load(open(splits_file))
        test_indices = splits.get('test', [])
    
    test_samples = [all_samples[idx] for idx in test_indices if idx < len(all_samples)]
    print(f"Loaded {len(test_samples)} test samples (indices: {test_indices})")
    
    # Statistics for all categories (support 1-6, consistent with training code)
    total_qa_count = 0
    category_stats = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # Support all 6 categories
    
    for sample in test_samples:
        if 'qa' not in sample or len(sample['qa']) == 0:
            continue
        
        for qa in sample['qa']:
            category = qa.get('category')
            if category and category in category_stats:
                category_stats[category] += 1
                total_qa_count += 1
    
    # Print statistics
    print(f"Total QA pairs: {total_qa_count}")
    for cat in sorted(category_stats.keys()):
        count = category_stats[cat]
        if count > 0:
            print(f"  Category {cat}: {count} QA pairs ({count/total_qa_count*100:.1f}%)")
    print("Note: All categories (1-6) are included (consistent with training)")
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Read cost_strategy from checkpoint (if exists), ensure using correct modules
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
        from train.train_longmemeval import load_modules
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
    # print(f"Model epoch: {checkpoint.get('epoch', 'unknown')}, Avg reward: {checkpoint.get('avg_reward', 'unknown'):.4f}\n")

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
    total_reward_bleu = 0.0
    total_reward_rouge_l = 0.0
    num_questions = 0
    action_counts = {
        'module1_low': 0, 'module1_mid': 0, 'module1_high': 0,
        'module2_low': 0, 'module2_mid': 0, 'module2_high': 0,
        'module3_low': 0, 'module3_mid': 0, 'module3_high': 0,
        'module4_low': 0, 'module4_mid': 0, 'module4_high': 0,
        'module5_low': 0, 'module5_mid': 0, 'module5_high': 0,
    }
    # Statistics by category (F1 and LLM Judge separately)
    category_stats_f1 = {
        1: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        2: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        3: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        4: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        5: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        6: {'rewards': [], 'count': 0, 'total_reward': 0.0},
    }
    category_stats_llm_judge = {
        1: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        2: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        3: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        4: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        5: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        6: {'rewards': [], 'count': 0, 'total_reward': 0.0},
    }
    category_stats_bleu = {
        1: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        2: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        3: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        4: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        5: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        6: {'rewards': [], 'count': 0, 'total_reward': 0.0},
    }
    category_stats_rouge_l = {
        1: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        2: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        3: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        4: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        5: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        6: {'rewards': [], 'count': 0, 'total_reward': 0.0},
    }   

    for data in tqdm(test_samples, desc="Testing"):
        print(f"\n{'='*60}")
        print(f"Processing test sample: {data['sample_id']}")
        print(f"{'='*60}")

        # Step 1: Get global memory pool (use preprocessed test_memory_pools if available)
        sample_id = data['sample_id']
        if test_memory_pools and sample_id in test_memory_pools:
            # Use preprocessed memory pool
            global_memory = test_memory_pools[sample_id]
            print(f"Using preprocessed global memory with {len(global_memory)} chunks")
        else:
            # Fallback: construct global memory pool (split into chunks by max_tokens, finer granularity)
            global_memory = construct_global_memory(
                data, args.retriever, context_tokenizer, context_encoder, args,
                data_tokenizer, data_encoder,
                max_tokens=args.chunk_max_tokens
            )
            print(f"Constructed global memory with {len(global_memory)} chunks")

        # Step 2: Prepare questions
        out_data = {'sample_id': data['sample_id'], 'qa': copy.deepcopy(data['qa'])}
        questions = []
        for i, qa in enumerate(data['qa']):
            question = qa['question']
            questions.append(question)

        
        if data_encoder is not None:
            all_query_embs = get_data_embeddings(data_encoder, questions)
            print(f"Using data_encoder for question embeddings (consistent with global_memory)")
        else:
            all_query_embs = get_embeddings_with_model(args.retriever, questions, query_tokenizer, query_encoder)
            print(f"Using query_encoder for question embeddings (consistent with global_memory)")

        print(f"Processing {len(questions)} questions for sample {data['sample_id']}")

        print(f"\n{'='*60}")
        print("Executing pipeline and evaluating each question (F1 + LLM Judge)")
        print(f"{'='*60}")

        sample_rewards_f1 = []
        sample_rewards_llm_judge = []
        sample_rewards_bleu = []
        sample_rewards_rouge_l = []
        sample_results = []

        # Define function to process single question (consistent with training code)
        def process_single_question(i, qa, question, query_emb_np):
            """
            Process a single question complete pipeline (retrieval, pipeline, evaluation)

            Args:
                i: Question index
                qa: Question-answer dictionary
                question: Question text
                query_emb_np: Question embedding

            Returns: Dictionary containing all necessary information for subsequent PPO updates and memory writes
            """
            if query_emb_np.ndim > 1:
                query_emb_np = query_emb_np.flatten()

            query_emb = torch.from_numpy(query_emb_np).float()

            # Retrieve relevant memories
            effective_top_k = max(args.top_k, 10)
            retrieved_memories = global_memory.retrieve(query_emb_np, top_k=effective_top_k)

            # Execute pipeline（Use deterministic=True to ensure reproducibility）
            pipeline_result = pipeline.execute(
                query=question,
                query_emb=query_emb,
                initial_memories=retrieved_memories,
                deterministic=True  # Use deterministic mode during testing
            )

            filtered_memories = pipeline_result['filtered_memories']
            entity_relations = pipeline_result['entity_relations']
            temporal_relations = pipeline_result['temporal_relations']
            topic_relations = pipeline_result['topic_relations']
            summary = pipeline_result['summary']
            actions = pipeline_result['actions']  # (m1, m2, m3, m5, m4) actions
            total_cost_q = pipeline_result['total_cost']

            # Build context and prompt (consistent with training logic)
            context_parts = []
            
            if summary:
                context_parts.append("\n<Summary>")
                context_parts.append(summary)
                context_parts.append("</Summary>")

            query_context = '\n'.join(context_parts)
            # Build prompt
            input_prompt = (
                LONGMEMEVAL_ANSWER_PROMPT.format(query_context, datetime.now().strftime("%Y-%m-%d"), question) if not args.cot else LONGMEMEVAL_ANSWER_PROMPT_COT.format(query_context, datetime.now().strftime("%Y-%m-%d"), question)
            )

            # Get LLM answer
            disable_threading = getattr(args, 'parallel_questions', 1) > 1
            task_args = [(i, input_prompt, args)]
            ret = get_llm_response(args=args, task_args=task_args, disable_internal_threading=disable_threading)

            # Calculate both F1 and LLM Judge evaluation methods simultaneously
            answer = str(out_data['qa'][i]['answer'])
        

            prediction = ""
            reward_f1 = 0.0
            reward_llm_judge = 0.0
            reward_bleu = 0.0
            reward_rouge_l = 0.0

            if len(ret) > 0:
                idx, response, _, success = ret[0]
                if success:
                    prediction = response.strip()
                    out_data['qa'][i]['prediction'] = prediction

                    reward_f1 = f1_score(prediction, answer)
                    reward_bleu = compute_bleu(prediction, answer)
                    reward_rouge_l = compute_rouge_l(prediction, answer)
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

            # Extract retrieved memory content
            retrieved_memory_contents = []
            for mem in retrieved_memories:
                if hasattr(mem, 'get_enriched_content'):
                    retrieved_memory_contents.append(mem.get_enriched_content())
                else:
                    content = mem.content
                    if hasattr(mem, 'date_time') and mem.date_time:
                        content = f"[Date: {mem.date_time}] {content}"
                    retrieved_memory_contents.append(content)

            # Extract filtered memory content
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

            return {
                'question_index': i,
                'question': question,
                'qa': qa,
                'prediction': prediction,
                'reward_f1': reward_f1,
                'reward_llm_judge': reward_llm_judge,
                'reward_bleu': reward_bleu,
                'reward_rouge_l': reward_rouge_l,
                'retrieved_memories': retrieved_memories,
                'filtered_memories': filtered_memories,
                'entity_relations': entity_relations,
                'temporal_relations': temporal_relations,
                'topic_relations': topic_relations,
                'summary': summary,
                'actions': actions,
                'total_cost': total_cost_q,
                'pipeline_result': pipeline_result,
                'final_prompt': input_prompt,
                'ground_truth': answer
            }

        # Process questions in batches (supports parallel)
        memory_batch_size = args.parallel_questions if args.parallel_questions > 1 else 1
        total_questions = len(questions)
        num_batches = (total_questions + memory_batch_size - 1) // memory_batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * memory_batch_size
            end_idx = min(start_idx + memory_batch_size, total_questions)
            batch_questions = list(range(start_idx, end_idx))

            # Process current batch questions
            batch_results = []

            if args.parallel_questions > 1 and len(batch_questions) > 1:
                with ThreadPoolExecutor(max_workers=args.parallel_questions) as executor:
                    future_to_question = {}
                    for i in batch_questions:
                        qa = data['qa'][i]
                        question = questions[i]
                        query_emb_np = all_query_embs[i]
                        future = executor.submit(process_single_question, i, qa, question, query_emb_np)
                        future_to_question[future] = i

                    batch_results_dict = {}
                    for future in as_completed(future_to_question):
                        try:
                            result = future.result()
                            batch_results_dict[result['question_index']] = result
                        except Exception as e:
                            print(f"Error processing question {future_to_question[future]}: {e}")
                            import traceback
                            traceback.print_exc()

                    batch_results = [batch_results_dict[i] for i in sorted(batch_results_dict.keys())]
            else:
                for i in batch_questions:
                    qa = data['qa'][i]
                    question = questions[i]
                    query_emb_np = all_query_embs[i]
                    result = process_single_question(i, qa, question, query_emb_np)
                    batch_results.append(result)

            # Process batch results
            for result in batch_results:
                i = result['question_index']
                qa = result['qa']
                reward_f1 = result['reward_f1']
                reward_llm_judge = result['reward_llm_judge']
                reward_bleu = result['reward_bleu']
                reward_rouge_l = result['reward_rouge_l']
                actions = result['actions']
                retrieved_memories = result['retrieved_memories']
                filtered_memories = result['filtered_memories']

                print(f"Q{i+1}: F1={reward_f1:.4f}, LLM-Judge={reward_llm_judge:.4f}, Actions: {actions}")

                # Statistics separately for F1 and LLM Judge
                sample_rewards_f1.append(reward_f1)
                sample_rewards_llm_judge.append(reward_llm_judge)
                sample_rewards_bleu.append(reward_bleu) 
                sample_rewards_rouge_l.append(reward_rouge_l)
                total_reward_f1 += reward_f1
                total_reward_llm_judge += reward_llm_judge
                total_reward_bleu += reward_bleu
                total_reward_rouge_l += reward_rouge_l
                num_questions += 1

                # Statistics by category（Statistics separately for F1 and LLM Judge）
                category = qa['category']
                if category in category_stats_f1:
                    category_stats_f1[category]['rewards'].append(reward_f1)
                    category_stats_f1[category]['count'] += 1
                    category_stats_f1[category]['total_reward'] += reward_f1
                if category in category_stats_llm_judge:
                    category_stats_llm_judge[category]['rewards'].append(reward_llm_judge)
                    category_stats_llm_judge[category]['count'] += 1
                    category_stats_llm_judge[category]['total_reward'] += reward_llm_judge
                if category in category_stats_bleu:
                    category_stats_bleu[category]['rewards'].append(reward_bleu)
                    category_stats_bleu[category]['count'] += 1
                    category_stats_bleu[category]['total_reward'] += reward_bleu
                if category in category_stats_rouge_l:
                    category_stats_rouge_l[category]['rewards'].append(reward_rouge_l)
                    category_stats_rouge_l[category]['count'] += 1
                    category_stats_rouge_l[category]['total_reward'] += reward_rouge_l

                # Record detailed results (includes both F1 and LLM Judge scores)
                m1_action, m2_action, m3_action, m5_action, m4_action = actions
                sample_results.append({
                    'question_index': i,
                    'question': result['question'],
                    'category': qa['category'],
                    'ground_truth': result['ground_truth'],
                    'prediction': result['prediction'],
                    'f1_score': reward_f1,
                    'llm_judge_score': reward_llm_judge,
                    'bleu_score': reward_bleu,
                    'rouge_l_score': reward_rouge_l,
                    'reward': reward_f1,  # Maintain backward compatibility, use F1 as main reward  
                    'actions': {
                        'module1': ['low', 'mid', 'high'][m1_action],
                        'module2': ['low', 'mid', 'high'][m2_action],
                        'module3': ['low', 'mid', 'high'][m3_action],
                        'module4': ['low', 'mid', 'high'][m4_action],
                        'module5': ['low', 'mid', 'high'][m5_action]
                    },
                    'retrieved_sessions_count': len(retrieved_memories),
                    'filtered_memories_count': len(filtered_memories)
                })

                # Accumulate action counts
                action_counts[f'module1_{["low", "mid", "high"][m1_action]}'] += 1
                action_counts[f'module2_{["low", "mid", "high"][m2_action]}'] += 1
                action_counts[f'module3_{["low", "mid", "high"][m3_action]}'] += 1
                action_counts[f'module4_{["low", "mid", "high"][m4_action]}'] += 1
                action_counts[f'module5_{["low", "mid", "high"][m5_action]}'] += 1
        # Note: since summary memory is no longer written, global memory maintains initial state (only contains original session chunks)
        print(f"\n{'='*60}")
        print(f"Final memory state: {len(global_memory)} memories (original session chunks only)")
        print(f"{'='*60}\n")

        # ========== Save global memory to local (for eval_locomo.py) ==========
        print(f"\n{'='*60}")
        print("Saving global memory to local file (original session chunks only)...")
        print(f"{'='*60}")

        # Extract all memories from global_memory and build database format
        contexts = []
        date_times = []
        embeddings_list = []
        context_ids = []

        for mem in global_memory.memories:
            contexts.append(mem.content)
            date_times.append(mem.date_time)
            embeddings_list.append(mem.embedding)
            # Build context_id identifier
            if mem.is_original:
                context_id = f"S{mem.session_id}:original"
            else:
                context_id = f"Q{mem.session_id}:processed"
            context_ids.append(context_id)

        # Build database dictionary (compatible with eval_locomo.py)
        database = {
            'embeddings': np.stack(embeddings_list),
            'date_time': date_times,
            'dia_id': context_ids,
            'context': contexts
        }

        # Save as pkl file
        dataset_prefix = os.path.splitext(os.path.split(args.data_file)[-1])[0]
        os.makedirs("./data", exist_ok=True)
        save_path = os.path.join("./data", f'{dataset_prefix}_memory_{data["sample_id"]}.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(database, f)

        print(f"Saved global memory to {save_path}")
        print(f"  Total memories: {len(global_memory)}")
        print(f"  Original session chunks: {sum(1 for m in global_memory.memories if m.is_original)}")
        print(f"  Note: Summary memories are NOT written (consistent with training)")
        print(f"{'='*60}\n")

        # Print sample-level statistics (F1 and LLM Judge)
        avg_sample_f1 = np.mean(sample_rewards_f1) if sample_rewards_f1 else 0.0
        avg_sample_llm_judge = np.mean(sample_rewards_llm_judge) if sample_rewards_llm_judge else 0.0
        max_f1 = max(sample_rewards_f1) if sample_rewards_f1 else 0.0
        min_f1 = min(sample_rewards_f1) if sample_rewards_f1 else 0.0
        max_llm = max(sample_rewards_llm_judge) if sample_rewards_llm_judge else 0.0
        min_llm = min(sample_rewards_llm_judge) if sample_rewards_llm_judge else 0.0
        avg_sample_bleu = np.mean(sample_rewards_bleu) if sample_rewards_bleu else 0.0
        max_bleu = max(sample_rewards_bleu) if sample_rewards_bleu else 0.0
        min_bleu = min(sample_rewards_bleu) if sample_rewards_bleu else 0.0
        avg_sample_rouge_l = np.mean(sample_rewards_rouge_l) if sample_rewards_rouge_l else 0.0
        max_rouge_l = max(sample_rewards_rouge_l) if sample_rewards_rouge_l else 0.0
        min_rouge_l = min(sample_rewards_rouge_l) if sample_rewards_rouge_l else 0.0
        print(f"{'='*60}")
        print(f"Sample {data['sample_id']} Results:")
        print(f"{'='*60}")
        print(f"  Total questions: {len(sample_rewards_f1)}")
        print(f"  Avg F1: {avg_sample_f1:.4f} (Max: {max_f1:.4f}, Min: {min_f1:.4f})")
        print(f"  Avg LLM-Judge: {avg_sample_llm_judge:.4f} (Max: {max_llm:.4f}, Min: {min_llm:.4f})")
        print(f"  Avg BLEU: {avg_sample_bleu:.4f} (Max: {max_bleu:.4f}, Min: {min_bleu:.4f})")
        print(f"  Avg ROUGE-L: {avg_sample_rouge_l:.4f} (Max: {max_rouge_l:.4f}, Min: {min_rouge_l:.4f})")
        print(f"{'='*60}")

        all_results.append({
            'sample_id': data['sample_id'],
            'avg_f1': avg_sample_f1,
            'avg_llm_judge': avg_sample_llm_judge,
            'avg_bleu': avg_sample_bleu,
            'avg_rouge_l': avg_sample_rouge_l,
            'questions': sample_results,
            'qa_data': out_data['qa'],
            'saved_memory_path': save_path
        })

    # Calculate overall statistics (F1 and LLM Judge)
    avg_test_reward_f1 = total_reward_f1 / max(num_questions, 1)
    avg_test_reward_llm_judge = total_reward_llm_judge / max(num_questions, 1)
    avg_test_reward_bleu = total_reward_bleu / max(num_questions, 1)
    avg_test_reward_rouge_l = total_reward_rouge_l / max(num_questions, 1)
    # Calculate average performance for each category (F1 and LLM Judge calculated separately)
    category_performance_f1 = {}
    category_performance_llm_judge = {}
    category_performance_bleu = {}
    category_performance_rouge_l = {}
    for cat in [1, 2, 3, 4, 5, 6]:
        # F1 statistics
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
        
        # LLM Judge statistics
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

        # BLEU statistics
        stats_bleu = category_stats_bleu[cat]
        if stats_bleu['count'] > 0:
            avg_reward = stats_bleu['total_reward'] / stats_bleu['count']
            category_performance_bleu[cat] = {
                'avg_reward': float(avg_reward),
                'total_reward': float(stats_bleu['total_reward']),
                'count': stats_bleu['count'],
                'min_reward': float(min(stats_bleu['rewards'])) if stats_bleu['rewards'] else 0.0,
                'max_reward': float(max(stats_bleu['rewards'])) if stats_bleu['rewards'] else 0.0,
                'std_reward': float(np.std(stats_bleu['rewards'])) if len(stats_bleu['rewards']) > 1 else 0.0
            }
        else:
            category_performance_bleu[cat] = {
                'avg_reward': 0.0, 'total_reward': 0.0, 'count': 0,
                'min_reward': 0.0, 'max_reward': 0.0, 'std_reward': 0.0
            }

        # ROUGE-L statistics
        stats_rouge_l = category_stats_rouge_l[cat]
        if stats_rouge_l['count'] > 0:
            avg_reward = stats_rouge_l['total_reward'] / stats_rouge_l['count']
            category_performance_rouge_l[cat] = {
                'avg_reward': float(avg_reward),
                'total_reward': float(stats_rouge_l['total_reward']),
                'count': stats_rouge_l['count'],
                'min_reward': float(min(stats_rouge_l['rewards'])) if stats_rouge_l['rewards'] else 0.0,
                'max_reward': float(max(stats_rouge_l['rewards'])) if stats_rouge_l['rewards'] else 0.0,
                'std_reward': float(np.std(stats_rouge_l['rewards'])) if len(stats_rouge_l['rewards']) > 1 else 0.0
            }
        else:
            category_performance_rouge_l[cat] = {
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
    print(f"  Average BLEU Score: {avg_test_reward_bleu:.4f}")
    print(f"  Average ROUGE-L Score: {avg_test_reward_rouge_l:.4f}")
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
    print(f"\nCategory Performance (BLEU):")
    for cat in sorted(category_performance_bleu.keys()):
        perf = category_performance_bleu[cat]
        if perf['count'] > 0:
            print(f"  Category {cat}: {perf['count']} questions, "
                  f"Avg BLEU: {perf['avg_reward']:.4f} "
                  f"(Min: {perf['min_reward']:.4f}, Max: {perf['max_reward']:.4f}, "
                  f"Std: {perf['std_reward']:.4f})")
        else:
            print(f"  Category {cat}: 0 questions")
    print(f"\nCategory Performance (ROUGE-L):")
    for cat in sorted(category_performance_rouge_l.keys()):
        perf = category_performance_rouge_l[cat]
        if perf['count'] > 0:
            print(f"  Category {cat}: {perf['count']} questions, "
                  f"Avg ROUGE-L: {perf['avg_reward']:.4f} "
                  f"(Min: {perf['min_reward']:.4f}, Max: {perf['max_reward']:.4f}, "
                  f"Std: {perf['std_reward']:.4f})")
        else:
            print(f"  Category {cat}: 0 questions")
    print(f"\nAction Distribution:")
    total_actions = sum(action_counts.values())
    for action, count in action_counts.items():
        ratio = count / max(total_actions, 1)
        print(f"  {action}: {count} ({ratio:.2%})")

    # Print saved memory file information
    print(f"\nSaved Memory Files (for eval_locomo.py):")
    for result in all_results:
        print(f"  {result['sample_id']}: {result['saved_memory_path']}")

    # Save test results
    os.makedirs("./test_results", exist_ok=True)
    
    return {
        'avg_f1': avg_test_reward_f1,
        'avg_llm_judge': avg_test_reward_llm_judge,
        'avg_bleu': avg_test_reward_bleu,
        'avg_rouge_l': avg_test_reward_rouge_l,
        'num_questions': num_questions,
        'action_counts': action_counts,
        'category_performance_f1': category_performance_f1,
        'category_performance_llm_judge': category_performance_llm_judge,
        'category_performance_bleu': category_performance_bleu,
        'category_performance_rouge_l': category_performance_rouge_l,
        'sample_results': all_results
    }
