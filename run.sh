#!/usr/bin/env bash
# Same-harness scale comparison: swap to the 1.5B model and rerun the
# identical AIME25 protocol. The repo's official eval_model.sh actually
# defaults to WeiboAI/VibeThinker-1.5B, so this isolates the model-size
# effect (paper claims 74.4 Pass@1 for 1.5B vs 91.4 for 3B) under one
# consistent eval. Hardware: 1x H100 (80GB); 1.5B is ~3GB in bf16.
set -euo pipefail

export MODEL_PATH="WeiboAI/VibeThinker-1.5B"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

echo "==> installing deps"
pip install -q "vllm==0.10.1" "transformers>=4.54.0" "math_verify[antlr4_13_2]" \
    pandas pyarrow hf_transfer 2>&1 | tail -5 || \
  pip install -q "vllm==0.10.1" "transformers>=4.54.0" math_verify pandas pyarrow hf_transfer 2>&1 | tail -5

echo "==> running AIME25 evaluation"
python3 vibethinker_eval.py
