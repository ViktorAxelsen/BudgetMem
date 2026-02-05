import argparse
import os


def get_locomo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, default="./data/locomo10.json")
    parser.add_argument('--test-data-file', type=str, default=None,
                        help="Path to test data file (default: None)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        choices=["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
                                 "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
                                 "meta/llama-3.1-70b-instruct", "qwen/qwen2.5-7b-instruct",
                                 "meta/llama-3.3-70b-instruct", "qwen/qwen3-next-80b-a3b-instruct"])
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument('--api', action='store_true')
    parser.add_argument('--llm-judge', action='store_true')
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--module-topk', type=int, default=5,
                        help="Top-k for module input selection (default: 5)")
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--api-base', type=str, default="")
    parser.add_argument('--api-key', type=str, nargs='+',
                        default=os.environ.get('NVIDIA_API_KEYS', '').split(',') if os.environ.get('NVIDIA_API_KEYS') else None,
                        help="NVIDIA API keys for LLM inference. Can also be set via NVIDIA_API_KEYS env var (comma-separated)")
    parser.add_argument('--parallel-questions', type=int, default=32,
                        help="Number of questions to process in parallel (1=sequential, >1=parallel)")
    parser.add_argument('--num-epochs', type=int, default=7,
                        help="Number of training epochs")
    parser.add_argument('--reward-weight', type=float, default=1,
                        help="Weight for reward scaling (default: 1.0)")
    parser.add_argument('--cost-weight', type=float, default=0.5,
                        help="Weight for cost penalty (default: 0.0, set to 0 to disable cost penalty)")
    parser.add_argument('--chunk-max-tokens', type=int, default=256,
                        help="Maximum tokens per memory chunk when constructing global memory (default: 512, optimized for Longformer's extended context)")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to the trained model checkpoint (default: None, can also use MODEL_PATH env var)")
    parser.add_argument('--enable-specific-modules', action='store_true', default=True,
                        help="Enable Specific Modules (PreferenceModule, DecisionRationaleModule, ProjectTaskModule) in the pipeline (default: False)")
    parser.add_argument('--restore-data', default=False, help="prepare data for training")
    # Cost-Performance Balance Strategy
    parser.add_argument('--cost-strategy', type=str, default='rule_llm',
                        choices=['rule_llm', 'prompt_tier', 'model_tier'],
                        help="Cost-performance balance strategy: "
                             "rule_llm (rules+embedding+LLM), "
                             "prompt_tier (simple/CoT/advanced prompts), "
                             "model_tier (small/medium/large models)")

    # Model size configuration for model_tier strategy
    parser.add_argument('--small-model', type=str, default='meta/llama-3.2-3b-instruct',
                        help="Small model for LOW cost tier (model_tier strategy)")
    parser.add_argument('--medium-model', type=str, default='meta/llama-3.1-8b-instruct',
                        help="Medium model for MID cost tier (model_tier strategy)")
    parser.add_argument('--large-model', type=str, default='meta/llama-3.3-70b-instruct',
                        help="Large model for HIGH cost tier (model_tier strategy)"),
    parser.add_argument('--test-last-model', type=str, default='./test_model/last_epoch.pt',
                        help="Test last model for testing (default: ./test_model/last_epoch.pt)"),
    parser.add_argument('--test-best-model', type=str, default='./test_model/best_model.pt',
                        help="Test best model for testing (default: ./test_model/best_model.pt)"),

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    pass

