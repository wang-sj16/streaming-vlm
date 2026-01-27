#!/bin/bash
#SBATCH --job-name=streamingvlm-sft-s2
#SBATCH --partition=ml.p5en.48xlarge-ultra-low
#SBATCH --account=low-pri
#SBATCH --nodes=8                      # 节点数量 (可修改)
#SBATCH --ntasks-per-node=1            # 每节点1个任务
#SBATCH --gpus-per-node=8              # 每节点8个GPU
#SBATCH --time=1-00:00:00              # 最大运行时间 (24 hours / 1 day)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive                    # 独占节点

########################################################
# Stage 2: High Quality Annealing Data
# 
# 等效batch size计算:
#   单节点: 8 GPUs × 1 batch × 64 accumulation = 512
#   多节点: (N nodes × 8 GPUs) × 1 batch × (64/N) accumulation = 512
########################################################

# ============== 用户配置区 ==============
NUM_NODES=${SLURM_NNODES:-4}
GPUS_PER_NODE=8
MASTER_PORT=${MASTER_PORT:-29500}

# 视频处理参数
export VIDEO_MIN_PIXELS=78400
export FPS_MAX_FRAMES=480
export VIDEO_MAX_PIXELS=19267584

# StreamingVLM 特有参数
text_sink=512
TEXT_SLIDING_WINDOW=512

# 数据集路径
export DATASET_PATH=/fsx/home/shijie.wang/code/streaming-vlm/data/Inf-Stream-Train

# ============== 训练超参数 ==============
epoch_num=1
learning_rate=1e-5
model_name="checkpoints/StreamingVLM_SFT_stage_1_e1_lr1e-5_ps512_pw512_n8/checkpoint-1380"  # Stage 1 的checkpoint路径
OPTIMIZER="adamw_torch"  # Options: adamw_torch, adamw_torch_fused

# 计算梯度累积步数
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
EFFECTIVE_BATCH_SIZE=512
PER_DEVICE_BATCH_SIZE=1
gradient_accumulation_steps=$((EFFECTIVE_BATCH_SIZE / WORLD_SIZE / PER_DEVICE_BATCH_SIZE))

echo "=============================================="
echo "Stage 2 Multi-Node Training Configuration:"
echo "  NUM_NODES: ${NUM_NODES}"
echo "  GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "  WORLD_SIZE: ${WORLD_SIZE}"
echo "  PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
echo "  GRADIENT_ACCUMULATION_STEPS: ${gradient_accumulation_steps}"
echo "  EFFECTIVE_BATCH_SIZE: $((WORLD_SIZE * PER_DEVICE_BATCH_SIZE * gradient_accumulation_steps))"
echo "  OPTIMIZER: ${OPTIMIZER}"
echo "=============================================="

# ============== WANDB 配置 ==============
# WANDB_API_KEY=${WANDB_API_KEY:-your-wandb-api-key}
# WANDB_ENTITY=${WANDB_ENTITY:-your-wandb-entity}
WANDB_PROJECT_NAME=StreamingVLM_SFT_stage_2
export WANDB_PROJECT=$WANDB_PROJECT_NAME

# Create RUN_NAME (add optimizer suffix only if not using default adamw_torch)
if [ "${OPTIMIZER}" = "adamw_torch" ]; then
    export RUN_NAME="${WANDB_PROJECT_NAME}_e${epoch_num}_lr${learning_rate}_ps${text_sink}_pw${TEXT_SLIDING_WINDOW}_n${NUM_NODES}"
else
    OPTIM_SHORT=${OPTIMIZER#adamw_}
    export RUN_NAME="${WANDB_PROJECT_NAME}_e${epoch_num}_lr${learning_rate}_ps${text_sink}_pw${TEXT_SLIDING_WINDOW}_n${NUM_NODES}_${OPTIM_SHORT}"
fi
export OUTPUT_DIR="./checkpoints"

# ============== 数据集配置 (Stage 2: 高质量退火数据) ==============
TRAIN_DATASET_NAMES=(
    "train_all_fg_with_seeks.jsonl"
)
VALID_DATASET_NAMES=(
    "valid_all_fg_with_seeks.jsonl"
)

TRAIN_FILES=("${TRAIN_DATASET_NAMES[@]/#/$DATASET_PATH/}")
VALID_FILES=("${VALID_DATASET_NAMES[@]/#/$DATASET_PATH/}")

# ============== 分布式环境设置 ==============
if [ -n "$SLURM_JOB_ID" ]; then
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=${MASTER_PORT}
    export WORLD_SIZE=${WORLD_SIZE}
    export RANK=${SLURM_PROCID}
    export LOCAL_RANK=${SLURM_LOCALID}
    
    echo "SLURM Mode:"
    echo "  MASTER_ADDR: ${MASTER_ADDR}"
    echo "  MASTER_PORT: ${MASTER_PORT}"
else
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
    export MASTER_PORT=${MASTER_PORT:-29500}
    echo "Manual Mode - Please set MASTER_ADDR, RANK, etc."
fi

# ============== 环境变量 ==============
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_DEBUG=INFO

# AWS EFA settings for p5 instances
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export NCCL_PROTO=simple

# Disable IB, use EFA instead
export NCCL_IB_DISABLE=1
# Let NCCL auto-detect the network interface (comment out if eth0 doesn't exist)
# export NCCL_SOCKET_IFNAME=eth0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 创建日志目录和checkpoint目录
mkdir -p logs
mkdir -p "${OUTPUT_DIR}/${RUN_NAME}"

# 创建符号链接，将SLURM日志关联到checkpoint目录
if [ -n "$SLURM_JOB_ID" ]; then
    LOG_OUT="$(realpath logs)/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
    LOG_ERR="$(realpath logs)/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    LINK_OUT="${OUTPUT_DIR}/${RUN_NAME}/train_${SLURM_JOB_ID}.out"
    LINK_ERR="${OUTPUT_DIR}/${RUN_NAME}/train_${SLURM_JOB_ID}.err"
    
    # 删除旧链接（如果存在）并创建新链接
    rm -f "${LINK_OUT}" "${LINK_ERR}"
    ln -s "${LOG_OUT}" "${LINK_OUT}"
    ln -s "${LOG_ERR}" "${LINK_ERR}"
    echo "Log symlinks created:"
    echo "  ${LINK_OUT} -> ${LOG_OUT}"
    echo "  ${LINK_ERR} -> ${LOG_ERR}"
fi

# ============== 启动训练 ==============
if [ -n "$SLURM_JOB_ID" ]; then
    srun --jobid=$SLURM_JOB_ID \
        bash -c "
        torchrun \
            --nnodes=${NUM_NODES} \
            --nproc_per_node=${GPUS_PER_NODE} \
            --rdzv_id=${SLURM_JOB_ID} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            train.py \
            --deepspeed ./scripts/zero0.json \
            --overwrite_output_dir True \
            --output_dir \"${OUTPUT_DIR}/${RUN_NAME}\" \
            --run_name ${RUN_NAME} \
            --do_train True \
            --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \
            --learning_rate ${learning_rate} \
            --warmup_ratio 0.03 \
            --optim ${OPTIMIZER} \
            --lr_scheduler_type cosine \
            --num_train_epochs ${epoch_num} \
            --logging_steps 1 \
            --bf16 True \
            --tf32 True \
            --gradient_checkpointing True \
            --pretrained_model_name_or_path ${model_name} \
            --train_annotation_paths ${TRAIN_FILES[*]} \
            --dataloader_num_workers 8 \
            --use_liger_kernel True \
            --report_to wandb \
            --ignore_data_skip False \
            --save_strategy steps \
            --save_steps 20 \
            --save_total_limit 1 \
            --load_best_model_at_end False \
            --greater_is_better False \
            --prediction_loss_only true \
            --eval_steps 200 \
            --metric_for_best_model eval_loss \
            --eval_strategy steps \
            --per_device_eval_batch_size 1 \
            --eval_annotation_paths ${VALID_FILES[*]} \
            --text_sink ${text_sink} \
            --text_sliding_window ${TEXT_SLIDING_WINDOW}
        "
else
    torchrun \
        --nnodes=${NUM_NODES} \
        --nproc_per_node=${GPUS_PER_NODE} \
        --rdzv_id=streamingvlm_s2_${SLURM_JOB_ID:-$$} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        train.py \
        --deepspeed ./scripts/zero0.json \
        --overwrite_output_dir True \
        --output_dir "${OUTPUT_DIR}/${RUN_NAME}" \
        --run_name ${RUN_NAME} \
        --do_train True \
        --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --learning_rate ${learning_rate} \
        --warmup_ratio 0.03 \
        --optim ${OPTIMIZER} \
        --lr_scheduler_type cosine \
        --num_train_epochs ${epoch_num} \
        --logging_steps 1 \
        --bf16 True \
        --tf32 True \
        --gradient_checkpointing True \
        --pretrained_model_name_or_path ${model_name} \
        --train_annotation_paths "${TRAIN_FILES[@]}" \
        --dataloader_num_workers 8 \
        --use_liger_kernel True \
        --report_to wandb \
        --ignore_data_skip False \
        --save_strategy steps \
        --save_steps 20 \
        --save_total_limit 1 \
        --load_best_model_at_end False \
        --greater_is_better False \
        --prediction_loss_only true \
        --eval_steps 200 \
        --metric_for_best_model eval_loss \
        --eval_strategy steps \
        --per_device_eval_batch_size 1 \
        --eval_annotation_paths "${VALID_FILES[@]}" \
        --text_sink ${text_sink} \
        --text_sliding_window ${TEXT_SLIDING_WINDOW}
fi
