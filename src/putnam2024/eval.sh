#!/bin/bash

# Dataset path
DATASET="putnam_test.json"

# GPUs
export CUDA_VISIBLE_DEVICES="6,7"

# Base command
BASE_CMD="inspect eval eval.py --log-dir logs --log-format=json -T dataset_path=${DATASET}"

# Run evaluations for each model
echo "Starting evaluations..."

# Claude 3.5 Sonnet
echo "Running Claude 3.5 Sonnet evaluation..."
$BASE_CMD --model anthropic/claude-3-5-sonnet-20241022

# GPT-4o (Original)
echo "Running GPT-4o evaluation..."
$BASE_CMD --model openai/gpt-4o-2024-11-20

# o1-preview
echo "Running o1-preview evaluation..."
$BASE_CMD --model openai/o1-preview-2024-09-12

# o1 mini
echo "Running o1-mini evaluation..."
$BASE_CMD --model openai/o1-mini-2024-09-12

# Gemini 1.5 Pro
echo "Running Gemini 1.5 Pro evaluation..."
$BASE_CMD --model google/gemini-1.5-pro-002

# Grok
echo "Running Grok evaluation..."
$BASE_CMD --model grok/grok-beta

# Llama 70B Instruct
echo "Running Llama 70B Instruct evaluation..."
$BASE_CMD --model vllm/meta-llama/Llama-3.1-70B-Instruct -M tensor_parallel_size=2

# Llama 7B
echo "Running Llama 7B evaluation..."
$BASE_CMD --model vllm/meta-llama/Llama-3.1-8B-Instruct -M tensor_parallel_size=2

# Qwen 7B Math Instruct
echo "Running QwQ-32B-Preview evaluation..."
$BASE_CMD --model vllm/Qwen/QwQ-32B-Preview -M tensor_parallel_size=2

# Qwen 2.5 Math Instruct
echo "Running Qwen 2.5 Math evaluation..."
$BASE_CMD --model vllm/Qwen/Qwen2.5-Math-72B-Instruct -M tensor_parallel_size=2

# Numina 72B
echo "Running Numina 72B evaluation..."
$BASE_CMD --model vllm/AI-MO/NuminaMath-72B-CoT -M tensor_parallel_size=2

echo "Evaluation complete!"