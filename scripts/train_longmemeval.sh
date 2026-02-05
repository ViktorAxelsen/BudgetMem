#!/bin/bash

# =============================================================================
# Environment Variables Configuration
# =============================================================================
# Copy .env.example to .env and set your API keys there, or export them directly

# Hugging Face Token - set via environment variable or .env file
# export HF_TOKEN=your_huggingface_token_here
# export HUGGINGFACE_TOKEN=$HF_TOKEN
# export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Wandb configuration
# export WANDB_API_KEY=your_wandb_api_key_here
export WANDB_DISABLE=true  # Set to false to enable wandb logging

# CUDA device configuration
export CUDA_VISIBLE_DEVICES=0

# WANDB project configuration
WANDB_PROJECT="longmemeval-training"
WANDB_ENTITY=""  # If using a team, set team name here, otherwise leave empty for personal account

# Training parameters (for generating unique run name)
MODEL_NAME="meta/llama-3.3-70b-instruct"
RETRIEVER_NAME="contriever"
COST_STRATEGY="rule_llm"  # Options: rule_llm, prompt_tier, model_tier
REWARD_WEIGHT=1
COST_WEIGHT=0
tag='llm'
DATA_FILE="../data/longmemeval/longmemeval_s.json"
chunk_max_tokens=256

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

WANDB_RUN_NAME="ppo_${MODEL_NAME##*/}_${RETRIEVER_NAME}_${COST_STRATEGY}_rw${REWARD_WEIGHT}_cw${COST_WEIGHT}_${tag}_cmt${chunk_max_tokens}_${TIMESTAMP}"
WANDB_RUN_NAME=$(echo "$WANDB_RUN_NAME" | sed 's/\//_/g' | sed 's/:/_/g')

export WANDB_RUN_NAME

mkdir -p ./logs_longmemeval

LOG_FILE="./logs_longmemeval/training_${COST_STRATEGY}_${tag}_cmt${chunk_max_tokens}_cw${COST_WEIGHT}_${TIMESTAMP}.log"


# Write training information to log file
echo "=== Training Information ===" > $LOG_FILE
echo "Start Time: $(date)" >> $LOG_FILE
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" >> $LOG_FILE
echo "WANDB_PROJECT: $WANDB_PROJECT" >> $LOG_FILE
echo "WANDB_RUN_NAME: $WANDB_RUN_NAME" >> $LOG_FILE
echo "MODEL_NAME: $MODEL_NAME" >> $LOG_FILE
echo "RETRIEVER_NAME: $RETRIEVER_NAME" >> $LOG_FILE
echo "COST_STRATEGY: $COST_STRATEGY" >> $LOG_FILE
echo "REWARD_WEIGHT: $REWARD_WEIGHT" >> $LOG_FILE
echo "COST_WEIGHT: $COST_WEIGHT" >> $LOG_FILE
echo "API_BASE: $API_BASE" >> $LOG_FILE
echo "API_KEY: [HIDDEN_FOR_SECURITY]" >> $LOG_FILE
echo "===========================" >> $LOG_FILE
echo "" >> $LOG_FILE

python -u train/train_longmemeval.py \
    --data-file ${DATA_FILE} \
    --model ${MODEL_NAME} \
    --api \
    --retriever ${RETRIEVER_NAME} \
    --llm-judge \
    --cost-strategy ${COST_STRATEGY} \  
    --reward-weight ${REWARD_WEIGHT} \
    --cost-weight ${COST_WEIGHT} \
    --chunk-max-tokens ${chunk_max_tokens} \
    2>&1 | tee -a $LOG_FILE

# Display log file location
echo "" >> $LOG_FILE
echo "=== Training Completed ===" >> $LOG_FILE
echo "Completion Time: $(date)" >> $LOG_FILE
echo "Training completed. Log file: $LOG_FILE"