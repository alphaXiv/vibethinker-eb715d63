set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODEL_PATH="WeiboAI/VibeThinker-1.5B"
DATA_PATH="/path/to/eval/data"
DATATYPES=("aime" "aime25" "hmmt25" "gpqa")
OUTPUT_DIR="./output"  # Add default output directory

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    if [ "${DATA_TYPE}" == "gpqa" ]; then
        N_SAMPLES=16
    else
        N_SAMPLES=64
    fi
    echo "Processing ${DATA_TYPE} with ${N_SAMPLES} samples"
    python3 -m verl.trainer.main_evaluation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=${DATA_PATH}/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.json \
        data.n_samples=${N_SAMPLES} \
        data.batch_size=2048 \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=40960 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.95 \
        rollout.tensor_model_parallel_size=1 
done