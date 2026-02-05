# Adaptive Long-term Memory Dialogue System

A reinforcement learning framework for dynamically balancing cost and performance in long-term memory dialogue systems. The system uses PPO (Proximal Policy Optimization) to learn optimal strategies for selecting computational resources across multiple processing modules.

## Key Features

- **Dynamic Cost-Performance Balancing**: Automatically learns to allocate computational resources based on query complexity
- **Modular Architecture**: 5 specialized processing modules (Filter, Entity Relation, Temporal Relation, Topic Relation, Summary)
- **Multiple Cost Strategies**: Support for three different cost-performance balancing approaches:
  - `rule_llm`: Rule-based + Embedding + LLM hybrid strategy
  - `prompt_tier`: Prompt complexity tiering (Direct/CoT/ReAct)
  - `model_tier`: Model size tiering (Small/Medium/Large)
- **Multi-Dataset Support**: Compatible with HotpotQA, LongMemEval, and LoCoMo datasets
- **Comprehensive Evaluation**: Built-in metrics including F1, BLEU, ROUGE-L, and LLM-as-Judge

## Architecture

```
Query
  |
  v
[Module 1: Filter] -----> Select top-k relevant memories
  |
  v
[Module 2: Entity Relation] -----> Extract entity relationships
[Module 3: Temporal Relation] -----> Extract temporal information
[Module 5: Topic Relation] -----> Analyze topic transitions
  |
  v
[Module 4: Summary] -----> Generate structured knowledge summary
  |
  v
Final Answer
```

Each module can independently select its cost level (LOW/MID/HIGH/NOOP) based on the learned policy, enabling fine-grained control over computational resources.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ GPU memory recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adaptive-memory-dialogue.git
cd adaptive-memory-dialogue
```

2. Create virtual environment:
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate

# Or using conda
conda create -n memory python=3.10
conda activate memory
```

3. Install dependencies:
```bash
pip install -r requirements.txt

# For GPU support with FAISS
pip install faiss-gpu

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NVIDIA_API_KEYS` | Comma-separated NVIDIA API keys | Yes (for API mode) |
| `HF_TOKEN` | Hugging Face access token | Yes (for model downloads) |
| `WANDB_API_KEY` | Weights & Biases API key | Optional |
| `CUDA_VISIBLE_DEVICES` | GPU device IDs | Optional |

### Command Line Arguments

Key arguments for training:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | LLM model name | `meta/llama-3.3-70b-instruct` |
| `--cost-strategy` | Cost balancing strategy | `rule_llm` |
| `--reward-weight` | Weight for reward (F1 score) | `1.0` |
| `--cost-weight` | Weight for cost penalty | `0.0` |
| `--top_k` | Number of memories to retrieve | `10` |
| `--module-topk` | Top-k for module input selection | `9` |
| `--retriever` | Retriever type | `contriever` |

## Usage

### Training

```bash
# Train on LoCoMo dataset
bash scripts/train_locomo.sh

# Train on HotpotQA dataset
bash scripts/train_hotpotqa.sh

# Train on LongMemEval dataset
bash scripts/train_longmemeval.sh
```

### Custom Training

```bash
python train/train_locomo.py \
    --model "meta/llama-3.3-70b-instruct" \
    --api \
    --retriever "contriever" \
    --cost-strategy "prompt_tier" \
    --reward-weight 1.0 \
    --cost-weight 0.5 \
    --llm-judge
```

### Evaluation

```bash
python tests/test_utils.py \
    --model-path ./checkpoints/best_model.pt \
    --data-file ./data/test.json
```

## Project Structure

```
.
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration and argument parsing
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── ppo_trainer.py        # Actor-Critic network and PPO training
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── rule_llm.py           # Rule + LLM hybrid strategy
│   │   ├── prompt_tier.py        # Prompt complexity tiering strategy
│   │   └── model_tier.py         # Model size tiering strategy
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── llm_utils.py          # LLM inference utilities
│   │   ├── llm_pricing.py        # Cost calculation and normalization
│   │   ├── rag_utils.py          # RAG retrieval utilities
│   │   └── eval_utils.py         # Evaluation metrics
│   └── prompts/
│       ├── __init__.py
│       └── prompt_pool.py        # Prompt templates for all modules
├── train/                        # Training scripts
│   ├── train_locomo.py           # LoCoMo dataset training
│   ├── train_hotpotqa.py         # HotpotQA dataset training
│   └── train_longmemeval.py      # LongMemEval dataset training
├── tests/                        # Test utilities
│   ├── test_utils.py
│   ├── test_utils_hotpotqa.py
│   └── test_utils_longmemeval.py
├── scripts/                      # Shell scripts for training
│   ├── train_locomo.sh
│   ├── train_hotpotqa.sh
│   └── train_longmemeval.sh
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Cost Strategies

### 1. Rule + LLM (`rule_llm`)

Combines rule-based methods with LLM inference:
- **LOW**: Simple rule matching and keyword extraction
- **MID**: Embedding similarity + rule-based filtering
- **HIGH**: Full LLM inference with complex reasoning

### 2. Prompt Tier (`prompt_tier`)

Uses the same model with different prompt complexities:
- **LOW**: Direct, simple prompts
- **MID**: Chain-of-Thought (CoT) prompts
- **HIGH**: Plan-Act-Reflect multi-step reasoning

### 3. Model Tier (`model_tier`)

Uses different model sizes:
- **LOW**: Small model (e.g., Llama-3.2-3B)
- **MID**: Medium model (e.g., Llama-3.1-8B)
- **HIGH**: Large model (e.g., Llama-3.3-70B)

## Supported Datasets

| Dataset | Description | Task Type |
|---------|-------------|-----------|
| LoCoMo | Long-context conversational memory | Multi-turn QA |
| HotpotQA | Multi-hop question answering | Reading comprehension |
| LongMemEval | Long-term memory evaluation | Memory retrieval |

## Evaluation Metrics

- **F1 Score**: Token-level F1 with stemming
- **BLEU**: Sentence-level BLEU score
- **ROUGE-L**: Longest common subsequence metric
- **LLM Judge**: GPT-based answer quality assessment
- **Cost**: Normalized API call cost

## Citation

If you find this work useful, please cite:

```bibtex
@article{adaptive-memory-dialogue,
  title={Adaptive Long-term Memory Dialogue System with Dynamic Cost-Performance Balancing},
  author={},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [vLLM](https://github.com/vllm-project/vllm)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://github.com/langchain-ai/langchain)
