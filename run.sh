#!/usr/bin/env bash
# Minimal reproduction of the VibeThinker-3B reasoning claims (arXiv 2606.16140).
# Serves the released 3B model with vLLM and evaluates Pass@1/Pass@n/Cons@n on
# the four shipped benchmarks: AIME24, AIME25, HMMT25, GPQA.
# Hardware: 1x H100 (80GB). The 3B model is ~6GB in bf16.
set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

echo "==> installing deps"
pip install -q "vllm==0.10.1" "transformers>=4.54.0" "math_verify[antlr4_13_2]" \
    pandas pyarrow hf_transfer 2>&1 | tail -5 || \
  pip install -q "vllm==0.10.1" "transformers>=4.54.0" math_verify pandas pyarrow hf_transfer 2>&1 | tail -5

echo "==> running 4-benchmark sweep (AIME24/AIME25/HMMT25/GPQA)"
python3 vibethinker_eval.py
