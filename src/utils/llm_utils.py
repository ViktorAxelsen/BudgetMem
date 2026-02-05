import time
import openai
import threading
import tiktoken
import logging
import copy
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from multiprocessing.dummy import Pool as ThreadPool

# Disable HTTP request logging from openai library
logging.getLogger("openai._client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Disable httpx INFO level logging (HTTP library used by OpenAI client)
logging.getLogger("httpx").setLevel(logging.WARNING)
# Disable OpenAI client INFO level logging (including retry information)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


MAX_CONTEXT_LENGTH = {
    # ----- OpenAI GPT -----
    'gpt-4-turbo': 128000,
    'gpt-4': 4096,
    'gpt-4-32k': 320000,
    'gpt-3.5-turbo-16k': 16000,
    'gpt-3.5-turbo-12k': 12000,
    'gpt-3.5-turbo-8k': 8000,
    'gpt-3.5-turbo-4k': 4000,
    'gpt-3.5-turbo': 4096,
    'gpt-4o': 128000,
    'gpt-4o-2024-08-06': 128000,
    "gpt-4o-mini-2024-07-18": 128000,

    # ----- LLaMA / Meta -----
    "meta-llama/llama-3-70b-chat-hf": 8192,
    "meta-llama/llama-3.2-3b-instruct": 8192,
    "meta/llama-3.1-70b-instruct": 128000,
    "meta/llama-3.3-70b-instruct": 128000,

    # ----- Gemma -----
    "google/gemma-7b-it": 8192,

    # ----- Qwen -----
    "qwen/qwen1.5-72b-chat": 32768,
    "qwen/qwen2.5-3b-instruct": 32768,
    "qwen/qwen2.5-7b-instruct": 32768,
    "qwen/qwen2.5-0.5b-instruct": 32768,
    "qwen/qwen3-next-80b-a3b-instruct": 262144,
    "qwen/qwen3-4b-instruct-2507": 262144,

    # ----- Mixtral / Mistral -----
    "mistralai/mixtral-8x7B-instruct-v0.1": 32768,
    "nousresearch/nous-hermes-2-mixtral-8x7b-dpo": 32768,
    "mistralai/mixtral-8x22b-instruct-v0.1": 65536,
}




# ---------------- Global variables: multi-key round-robin + client cache ----------------

_client_cache = {}        # {api_key: openai.OpenAI(...)}
_key_index = 0            # round-robin index for next key to use
_client_lock = threading.Lock()

# ---------------- Global variables: vLLM instance cache ----------------
_vllm_cache = {}          # {model_name: LLM(...)}
_vllm_lock = threading.Lock()
_llm_api_stats = {
    'total_input_tokens': 0,
    'total_output_tokens': 0,
    'total_tokens': 0,
    'total_api_calls': 0,
    'total_cost_usd': 0.0,
    'models': {}
}
_llm_api_stats_lock = threading.Lock()


def _get_or_create_vllm(model_name, tensor_parallel_size=1):
    """
    Get or create vLLM instance (global cache, avoid reloading model)
    Thread-safe.
    """
    global _vllm_cache

    cache_key = f"{model_name}_tp{tensor_parallel_size}"

    # First check if exists (lock-free fast path)
    llm = _vllm_cache.get(cache_key)
    if llm is not None:
        return llm

    # Need to create new instance, acquire lock
    with _vllm_lock:
        # Double-check to avoid multiple threads creating simultaneously
        llm = _vllm_cache.get(cache_key)
        if llm is not None:
            return llm

        # Create new vLLM instance
        print(f"[vLLM] Creating new LLM instance for {model_name} (tensor_parallel_size={tensor_parallel_size})")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=tensor_parallel_size
        )
        _vllm_cache[cache_key] = llm
        print(f"[vLLM] LLM instance created and cached: {cache_key}")

    return llm


def _get_client_round_robin(
    api_keys,
    base_url="",
    max_retries=2,
    timeout=120
):
    """
    Select a key from api_keys in round-robin fashion, return corresponding client.
    Thread-safe.
    """
    global _key_index, _client_cache

    # Support passing single string
    if isinstance(api_keys, str):
        api_keys = [api_keys]

    if not api_keys:
        raise ValueError("API KEY EMPTY")

    # Select key (round-robin inside lock to ensure thread safety)
    with _client_lock:
        key = api_keys[_key_index % len(api_keys)]
        _key_index += 1

        # Check if client for this key has been created
        client = _client_cache.get(key)
        if client is None:
            client = openai.OpenAI(
                base_url=base_url,
                api_key=key,
                max_retries=max_retries,
                timeout=timeout
            )
            _client_cache[key] = client

    return client


def get_llm_response_via_api(prompt,
                             LLM_MODEL="",
                             base_url="",
                             api_key="",
                             MAX_TOKENS=100,
                             TAU=1.0,
                             TOP_P=1.0,
                             SEED=42,
                             MAX_TRIALS=3,
                             TIME_GAP=5,
                             response_format=None):
    '''
    res = get_llm_response_via_api(prompt='hello')  # Default: TAU Sampling (TAU=1.0)
    res = get_llm_response_via_api(prompt='hello', TAU=0)  # Greedy Decoding
    res = get_llm_response_via_api(prompt='hello', TAU=0.5, N=2, SEED=None)  # Return Multiple Responses w/ TAU Sampling
    '''
    if not api_key:
        raise ValueError("API KEY EMPTY")

    completion = None
    trials_left = MAX_TRIALS
    # Increase timeout, large model requests may need longer time
    timeout_value = 120  # 120s 

    while trials_left:
        trials_left -= 1
        client = _get_client_round_robin(
            api_keys=api_key,
            base_url=base_url,
            max_retries=2,
            timeout=timeout_value
        )
        try:
            # Build API call parameters
            api_params = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TAU,
                "top_p": TOP_P,
                "seed": SEED,
                "max_tokens": MAX_TOKENS,
            }

            # If response_format is specified, add to parameters
            if response_format is not None:
                api_params["response_format"] = response_format

            completion = client.chat.completions.create(**api_params)
            break
        except Exception as e:
            print(e)
            if "request timed out" in str(e).strip().lower():
                # Modified: timeout should not directly break, should continue trying next key
                # _get_client_round_robin will round-robin to next key
                if trials_left > 0:
                    print(f"Request timed out, trying next key... "
                          f"(trials left: {trials_left})")
                    time.sleep(TIME_GAP)
                    continue  # Continue next iteration, try next key
                else:
                    # All attempts failed
                    break
            print("Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception("Reach MAX_TRIALS={}".format(MAX_TRIALS))

    contents = completion.choices
    meta_info = completion.usage
    completion_tokens = meta_info.completion_tokens
    prompt_tokens = meta_info.prompt_tokens
    total_tokens = meta_info.total_tokens

    # Only count on success (completion is not None)
    if completion is not None:
        from llm_pricing import calculate_cost
        cost = calculate_cost(LLM_MODEL, prompt_tokens, completion_tokens)

        with _llm_api_stats_lock:
            _llm_api_stats['total_input_tokens'] += prompt_tokens
            _llm_api_stats['total_output_tokens'] += completion_tokens
            _llm_api_stats['total_tokens'] += total_tokens
            _llm_api_stats['total_api_calls'] += 1
            _llm_api_stats['total_cost_usd'] += cost

            if LLM_MODEL not in _llm_api_stats['models']:
                _llm_api_stats['models'][LLM_MODEL] = {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'api_calls': 0,
                    'cost_usd': 0.0
                }
            model_stats = _llm_api_stats['models'][LLM_MODEL]
            model_stats['input_tokens'] += prompt_tokens
            model_stats['output_tokens'] += completion_tokens
            model_stats['total_tokens'] += total_tokens
            model_stats['api_calls'] += 1
            model_stats['cost_usd'] += cost

    if len(contents) == 1:
        return contents[0].message.content, prompt_tokens, completion_tokens
    else:
        return [c.message.content for c in contents], prompt_tokens, completion_tokens



def get_tokenizer(model_name):
    """
    Determine whether to use tiktoken (OpenAI GPT series) based on model_name
    or HuggingFace AutoTokenizer (all other models).
    """

    # 1. Determine if it is OpenAI GPT model
    lower_name = model_name.lower()
    is_gpt = (
        "gpt" in lower_name or
        "openai" in lower_name or
        lower_name.startswith("o1") or
        lower_name.startswith("o3") or
        lower_name.startswith("gpt-")
    )

    if is_gpt:
        # ---- OpenAI tokenizer (tiktoken) ----
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            # fallback: if model has no specific config, use cl100k_base
            encoding = tiktoken.get_encoding("cl100k_base")

        return encoding

    else:
        # ---- Regular HuggingFace tokenizer ----
        if model_name == "meta/llama-3.3-70b-instruct":
            model_name = "meta-llama/Llama-3.1-70B-Instruct"
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return hf_tokenizer


def request_task(data):
    q_id, query_text, args = data
    try:
        # Check if response_format parameter exists
        response_format = getattr(args, 'response_format', None)

        answer, prompt_tokens, completion_tokens = get_llm_response_via_api(
            prompt=query_text,
            MAX_TOKENS=args.max_tokens,
            LLM_MODEL=args.model,
            TAU=args.temperature,
            base_url=args.api_base,
            api_key=args.api_key,
            response_format=response_format
        )
        # print("answer: ",answer)
        # print("prompt_tokens: ",prompt_tokens)
        # print("completion_tokens: ",completion_tokens)
        success = True
    except Exception as e:
        print(e)
        answer = "API Request Error"
        # print("API Request Error: ", e)
        prompt_tokens = 0
        completion_tokens = 0
        success = False

    return q_id, answer, (prompt_tokens, completion_tokens), success


def get_llm_response(args, task_args, disable_internal_threading=False):
    """
    q_id must be int, and is sorted by ascend order (q_id can be non-continuous)

    Args:
        args: Arguments object
        task_args: Task arguments list
        disable_internal_threading: Disable internal ThreadPool (set to True when outer layer is already parallel, avoid thread explosion)
    """
    ret = []
    if args.api:
        full_task_args = list(task_args)
        func_round = args.round
        while func_round > 0:
            func_round -= 1
            if len(ret) != 0:
                ret.sort(key=lambda x: x[0], reverse=False)
                task_args = [i for ind, i in enumerate(full_task_args) if not ret[ind][-1]]
                ret = [i for i in ret if i[-1]]

            # If internal threading disabled, process serially; otherwise use ThreadPool
            if disable_internal_threading:
                # Serial processing to avoid conflict with outer parallelism
                for task in task_args:
                    r = request_task(task)
                    ret.append(r)
            else:
                # Normal use ThreadPool for parallel processing
                with ThreadPool(args.batch_size) as p:
                    for r in p.imap_unordered(request_task, task_args):
                        ret.append(r)

            if sum([1 if not i[-1] else 0 for i in ret]) == 0:
                break
    else:
        # Use cached vLLM instance (avoid reloading model)
        tensor_parallel_size = getattr(args, 'tensor_parallel_size', 1)
        llm = _get_or_create_vllm(args.model, tensor_parallel_size)

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": query[1]}],
                tokenize=False,
                add_generation_prompt=True
            )
            for query in task_args
        ]

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.max_tokens
        )
        all_outputs = []
        output_token_counts = []
        for i in range(0, len(prompt_texts), args.batch_size):
            batch_prompts = prompt_texts[i:i + args.batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            for output in outputs:
                clean_text = output.outputs[0].text.strip()
                all_outputs.append(clean_text)
                output_tokens = len(tokenizer.encode(clean_text, add_special_tokens=False))
                output_token_counts.append(output_tokens)

        input_token_counts = [
            len(tokenizer.encode(prompt_text, add_special_tokens=False))
            for prompt_text in prompt_texts
        ]
        ret = []
        for element in zip([i[0] for i in task_args], all_outputs, input_token_counts, output_token_counts, [True] * len(all_outputs)):
            ret.append((element[0], element[1], (element[2], element[3]), element[-1]))

    ret.sort(key=lambda x: x[0], reverse=False)
    # print("ret: ", ret)
    return ret


def get_llm_api_stats():
    """
    Get global LLM API call statistics (thread-safe, returns deep copy)

    Returns:
        Statistics dictionary, format consistent with structure in add_locomo.py
    """
    with _llm_api_stats_lock:
        return copy.deepcopy(_llm_api_stats)


def reset_llm_api_stats():
    """Reset global LLM API call statistics"""
    global _llm_api_stats
    with _llm_api_stats_lock:
        _llm_api_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'total_api_calls': 0,
            'total_cost_usd': 0.0,
            'models': {}
        }


if __name__ == '__main__':
    pass