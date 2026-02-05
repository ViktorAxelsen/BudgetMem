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
    QA_PROMPT, LLM_JUDGE_GENERAL_PROMPT
)
from src.trainer.ppo_trainer import ActorCriticNetwork



def _infer_num_actions_per_module_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    default: int = 4,
) -> int:
    """Infer action dimension from a saved ActorCriticNetwork state_dict."""
    preferred_keys_new = [
        "action_head.2.bias",
        "action_head.2.weight",
    ]
    for key in preferred_keys_new:
        tensor = state_dict.get(key)
        if tensor is not None:
            try:
                if len(tensor.shape) >= 1:
                    return int(tensor.shape[0])
            except Exception:
                continue
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
            print("[LLM Judge] API returned None, returning 0.0")
            return 0.0

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
    """
    from train.train_locomo import (
        DEVICE,
        QuestionTrackingRecord,
        global_question_tracker,
        construct_global_memory
    )

    print("\n" + "="*80)
    print("Testing on Test Set")
    print("="*80)

    all_samples = json.load(open(args.data_file))
    test_index = [8, 9]
    test_samples = [all_samples[idx] for idx in test_index if idx < len(all_samples)]
    print(f"Loaded {len(test_samples)} test samples (indices: {test_index})")
    
    total_qa_count = 0
    filtered_category_5_count = 0
    category_stats = {1: 0, 2: 0, 3: 0, 4: 0}  # Only statistics for categories 1-4
    
    for sample in test_samples:
        if 'qa' not in sample or len(sample['qa']) == 0:
            continue
        
        filtered_qa = []
        for qa in sample['qa']:
            category = qa.get('category')
            if category == 5:
                filtered_category_5_count += 1
                continue
            if category and category in category_stats:
                filtered_qa.append(qa)
                category_stats[category] += 1
        
        sample['qa'] = filtered_qa
        total_qa_count += len(filtered_qa)
    
    print(f"Total QA pairs (after filtering): {total_qa_count}")
    print(f"Filtered out Category 5: {filtered_category_5_count} QA pairs")
    for cat in sorted(category_stats.keys()):
        count = category_stats[cat]
        if count > 0:
            print(f"  Category {cat}: {count} QA pairs ({count/total_qa_count*100:.1f}%)")
    print("Note: Category 5 (Adversarial) has been filtered out")
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    checkpoint_cost_strategy = checkpoint.get('cost_strategy', None)
    if checkpoint_cost_strategy is not None:
        if checkpoint_cost_strategy != getattr(args, 'cost_strategy', None):
            print(f"Warning: Checkpoint cost_strategy ({checkpoint_cost_strategy}) differs from args.cost_strategy ({getattr(args, 'cost_strategy', None)})")
            print(f"Using checkpoint cost_strategy: {checkpoint_cost_strategy}")
        args.cost_strategy = checkpoint_cost_strategy
    else:
        if not hasattr(args, 'cost_strategy') or args.cost_strategy is None:
            args.cost_strategy = 'rule_llm'  # Default fallback
            print(f"Warning: No cost_strategy found in checkpoint, using default: {args.cost_strategy}")
    
    if Module1_Filter is None or Module5_TopicRelation is None or Module2_EntityRelation is None or Module3_TemporalRelation is None or Module4_Summary is None or ModularPipelineExecutor is None:
        print(f"Loading modules with cost strategy: {args.cost_strategy}")
        from train.train_locomo import load_modules
        (_, _, _, Module1_Filter, Module2_EntityRelation,
         Module3_TemporalRelation, Module5_TopicRelation,
         Module4_Summary, ModularPipelineExecutor) = load_modules(
            args.cost_strategy)
    print(f"Loaded modules with cost strategy: {args.cost_strategy}")
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
        desc_encoder=data_encoder
    )
    actor_critic.load_state_dict(state_dict)
    actor_critic.to(DEVICE)
    actor_critic.eval()
    print(f"Loaded model from {model_path} (num_actions_per_module={num_actions_per_module})")

    module1 = Module1_Filter(
        encoder=data_encoder,
        args=args,
        top_k=getattr(args, 'module_topk', 5)
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
        'module5_low': 0, 'module5_mid': 0, 'module5_high': 0,
        'module4_low': 0, 'module4_mid': 0, 'module4_high': 0,
    }
    category_stats_f1 = {
        1: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        2: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        3: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        4: {'rewards': [], 'count': 0, 'total_reward': 0.0},
    }
    category_stats_llm_judge = {
        1: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        2: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        3: {'rewards': [], 'count': 0, 'total_reward': 0.0},
        4: {'rewards': [], 'count': 0, 'total_reward': 0.0},
    }
    for data in tqdm(test_samples, desc="Testing"):
        print(f"\n{'='*60}")
        print(f"Processing test sample: {data['sample_id']}")
        print(f"{'='*60}")

        sample_id = data['sample_id']
        if test_memory_pools and sample_id in test_memory_pools:
            global_memory = test_memory_pools[sample_id]
            print(f"Using preprocessed global memory with {len(global_memory)} chunks")
        else:
            global_memory = construct_global_memory(
                data, args.retriever, context_tokenizer, context_encoder, args,
                data_tokenizer, data_encoder,
                max_tokens=getattr(args, 'chunk_max_tokens', 256)
            )
            print(f"Constructed global memory with {len(global_memory)} chunks")

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
        sample_results = []

        def process_single_question(i, qa, question, query_emb_np, answer):
            if query_emb_np.ndim > 1:
                query_emb_np = query_emb_np.flatten()

            query_emb = torch.from_numpy(query_emb_np).float()

            effective_top_k = max(args.top_k, 10)
            retrieved_memories = global_memory.retrieve(query_emb_np, top_k=effective_top_k)

            pipeline_result = pipeline.execute(
                query=question,
                query_emb=query_emb,
                initial_memories=retrieved_memories,
                deterministic=True
            )

            filtered_memories = pipeline_result['filtered_memories']
            entity_relations = pipeline_result['entity_relations']
            temporal_relations = pipeline_result['temporal_relations']
            topic_relations = pipeline_result['topic_relations']
            summary = pipeline_result['summary']
            actions = pipeline_result['actions']  # (m1, m2, m3, m5, m4)

            context_parts = []
            
            if summary:
                context_parts.append("\n<Summary>")
                context_parts.append(summary)
                context_parts.append("</Summary>")

            query_context = '\n'.join(context_parts)
            input_prompt = query_context + '\n\n' + (
                QA_PROMPT.format(question)
            )

            disable_threading = getattr(args, 'parallel_questions', 1) > 1
            task_args = [(i, input_prompt, args)]
            ret = get_llm_response(args=args, task_args=task_args, disable_internal_threading=disable_threading)
            answer_str = str(answer)
            if qa['category'] == 3:
                answer_str = answer_str.split(';')[0].strip()

            prediction = ""
            reward_f1 = 0.0
            reward_llm_judge = 0.0

            if len(ret) > 0:
                idx, response, _, success = ret[0]
                if success:
                    prediction = response.strip()
                    out_data['qa'][i]['prediction'] = prediction

                    if qa['category'] == 1:
                        reward_f1 = f1_max(prediction, answer_str)
                    elif qa['category'] in [2, 3, 4]:
                        reward_f1 = f1_score(prediction, answer_str)
                    try:
                        judge_question = qa['question']
                        reward_llm_judge = get_llm_judge_reward(
                            question=judge_question,
                            ground_truth=answer_str,
                            prediction=prediction,
                            args=args
                        )
                    except Exception as e:
                        print(f"[Warning] LLM Judge failed for Q{i+1}: {e}, using 0.0")
                        reward_llm_judge = 0.0

            print(f"Q{i+1}: F1={reward_f1:.4f}, LLM-Judge={reward_llm_judge:.4f}, Actions: {actions}")

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

            tracking_record = QuestionTrackingRecord(
                sample_id=data['sample_id'],
                question_index=i,
                question=question,
                ground_truth=answer_str,
                prediction=prediction,
                reward=reward_f1,
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

            m1_action, m2_action, m3_action, m5_action, m4_action = actions
            question_result = {
                'question_index': i,
                'question': question,
                'category': qa['category'],
                'ground_truth': out_data['qa'][i]['answer'],
                'prediction': prediction,
                'f1_score': reward_f1,
                'llm_judge_score': reward_llm_judge,
                'reward': reward_f1,
                'actions': {
                    'module1': ['low', 'mid', 'high'][m1_action],
                    'module2': ['low', 'mid', 'high'][m2_action],
                    'module3': ['low', 'mid', 'high'][m3_action],
                    'module4': ['low', 'mid', 'high'][m4_action],
                    'module5': ['low', 'mid', 'high'][m5_action]
                },
                'retrieved_sessions_count': len(retrieved_memories),
                'filtered_memories_count': len(filtered_memories)
            }

            return question_result

        total_questions = len(data['qa'])
        parallel_questions = getattr(args, 'parallel_questions', 1)

        if parallel_questions > 1:
            print(f"Using parallel processing with {parallel_questions} workers")
            print("Using sequential processing")
            memory_batch_size = parallel_questions
            num_batches = (total_questions + memory_batch_size - 1) // memory_batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * memory_batch_size
                end_idx = min(start_idx + memory_batch_size, total_questions)
                batch_questions = list(range(start_idx, end_idx))

                batch_results = []
                if len(batch_questions) > 1:
                    with ThreadPoolExecutor(max_workers=parallel_questions) as executor:
                        future_to_question = {}
                        for i in batch_questions:
                            qa = data['qa'][i]
                            question = questions[i]
                            query_emb_np = all_query_embs[i]
                            answer = out_data['qa'][i]['answer']
                            future = executor.submit(process_single_question, i, qa, question, query_emb_np, answer)
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
                        answer = out_data['qa'][i]['answer']
                        result = process_single_question(i, qa, question, query_emb_np, answer)
                        batch_results.append(result)

                for result in batch_results:
                    sample_results.append(result)
                    sample_rewards_f1.append(result['f1_score'])
                    sample_rewards_llm_judge.append(result['llm_judge_score'])

                    total_reward_f1 += result['f1_score']
                    total_reward_llm_judge += result['llm_judge_score']
                    num_questions += 1

                    category = result['category']
                    if category in category_stats_f1:
                        category_stats_f1[category]['rewards'].append(result['f1_score'])
                        category_stats_f1[category]['count'] += 1
                        category_stats_f1[category]['total_reward'] += result['f1_score']
                    if category in category_stats_llm_judge:
                        category_stats_llm_judge[category]['rewards'].append(result['llm_judge_score'])
                        category_stats_llm_judge[category]['count'] += 1
                        category_stats_llm_judge[category]['total_reward'] += result['llm_judge_score']

                    actions = result['actions']
                    action_counts[f'module1_{actions["module1"]}'] += 1
                    action_counts[f'module2_{actions["module2"]}'] += 1
                    action_counts[f'module3_{actions["module3"]}'] += 1
                    action_counts[f'module4_{actions["module4"]}'] += 1
                    action_counts[f'module5_{actions["module5"]}'] += 1
        else:
            print("Using sequential processing")
            for i in range(total_questions):
                qa = data['qa'][i]
                question = questions[i]
                query_emb_np = all_query_embs[i]
                answer = out_data['qa'][i]['answer']
                result = process_single_question(i, qa, question, query_emb_np, answer)

                sample_results.append(result)
                sample_rewards_f1.append(result['f1_score'])
                sample_rewards_llm_judge.append(result['llm_judge_score'])

                total_reward_f1 += result['f1_score']
                total_reward_llm_judge += result['llm_judge_score']
                num_questions += 1

                category = result['category']
                if category in category_stats_f1:
                    category_stats_f1[category]['rewards'].append(result['f1_score'])
                    category_stats_f1[category]['count'] += 1
                    category_stats_f1[category]['total_reward'] += result['f1_score']
                if category in category_stats_llm_judge:
                    category_stats_llm_judge[category]['rewards'].append(result['llm_judge_score'])
                    category_stats_llm_judge[category]['count'] += 1
                    category_stats_llm_judge[category]['total_reward'] += result['llm_judge_score']

                actions = result['actions']
                action_counts[f'module1_{actions["module1"]}'] += 1
                action_counts[f'module2_{actions["module2"]}'] += 1
                action_counts[f'module3_{actions["module3"]}'] += 1
                action_counts[f'module4_{actions["module4"]}'] += 1
                action_counts[f'module5_{actions["module5"]}'] += 1
    
        print(f"\n{'='*60}")
        print(f"Final memory state: {len(global_memory)} memories (original session chunks only)")
        print(f"{'='*60}\n")

        print(f"\n{'='*60}")
        print("Saving global memory to local file (original session chunks only)...")
        print(f"{'='*60}")

        contexts = []
        date_times = []
        embeddings_list = []
        context_ids = []

        for mem in global_memory.memories:
            contexts.append(mem.content)
            date_times.append(mem.date_time)
            embeddings_list.append(mem.embedding)
            if mem.is_original:
                context_id = f"S{mem.session_id}:original"
            else:
                context_id = f"Q{mem.session_id}:processed"
            context_ids.append(context_id)

        database = {
            'embeddings': np.stack(embeddings_list),
            'date_time': date_times,
            'dia_id': context_ids,
            'context': contexts
        }

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

        avg_sample_f1 = np.mean(sample_rewards_f1) if sample_rewards_f1 else 0.0
        avg_sample_llm_judge = np.mean(sample_rewards_llm_judge) if sample_rewards_llm_judge else 0.0
        max_f1 = max(sample_rewards_f1) if sample_rewards_f1 else 0.0
        min_f1 = min(sample_rewards_f1) if sample_rewards_f1 else 0.0
        max_llm = max(sample_rewards_llm_judge) if sample_rewards_llm_judge else 0.0
        min_llm = min(sample_rewards_llm_judge) if sample_rewards_llm_judge else 0.0
        print(f"{'='*60}")
        print(f"Sample {data['sample_id']} Results:")
        print(f"{'='*60}")
        print(f"  Total questions: {len(sample_rewards_f1)}")
        print(f"  Avg F1: {avg_sample_f1:.4f} (Max: {max_f1:.4f}, Min: {min_f1:.4f})")
        print(f"  Avg LLM-Judge: {avg_sample_llm_judge:.4f} (Max: {max_llm:.4f}, Min: {min_llm:.4f})")
        print(f"{'='*60}")

        all_results.append({
            'sample_id': data['sample_id'],
            'avg_f1': avg_sample_f1,
            'avg_llm_judge': avg_sample_llm_judge,
            'questions': sample_results,
            'qa_data': out_data['qa'],
            'saved_memory_path': save_path
        })

    avg_test_reward_f1 = total_reward_f1 / max(num_questions, 1)
    avg_test_reward_llm_judge = total_reward_llm_judge / max(num_questions, 1)
    category_performance_f1 = {}
    category_performance_llm_judge = {}
    for cat in [1, 2, 3, 4]:
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
    print("Testing Strategy:")
    print("  - Execute pipeline for each question")
    print("  - Evaluate with BOTH F1-score and LLM Judge")
    print("  - Do NOT add summary memories to memory bank (consistent with training)")
    print("\nOverall Results:")
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

    print(f"\nSaved Memory Files (for eval_locomo.py):")
    for result in all_results:
        print(f"  {result['sample_id']}: {result['saved_memory_path']}")

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
