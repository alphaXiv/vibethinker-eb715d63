#!/usr/bin/env bash
# Minimal reproduction of the VibeThinker-3B AIME25 claim (arXiv 2606.16140).
# Serves the released 3B model with vLLM and evaluates Pass@1 on AIME25.
# Hardware: 1x H100 (80GB). The 3B model is ~6GB in bf16.
set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# Ablation: cap reasoning budget at 16384 tokens (vs the default 40960) to
# measure how much of VibeThinker-3B's AIME25 Pass@1 depends on long CoT.
# vibethinker_eval.py keys both SamplingParams.max_tokens and max_model_len
# off this env var, so this is the single knob being changed.
export MAX_TOKENS=16384

echo "==> installing deps"
pip install -q "vllm==0.10.1" "transformers>=4.54.0" "math_verify[antlr4_13_2]" \
    pandas pyarrow hf_transfer 2>&1 | tail -5 || \
  pip install -q "vllm==0.10.1" "transformers>=4.54.0" math_verify pandas pyarrow hf_transfer 2>&1 | tail -5

echo "==> running AIME25 evaluation"
python3 vibethinker_eval.py
