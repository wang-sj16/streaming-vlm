#!/bin/bash
########################################################
# 多节点训练启动脚本
# 
# 使用方法:
#   1. SLURM模式 (推荐):
#      sbatch scripts/sft_stage_1_multinode.sh
#
#   2. 手动模式 (在每个节点上分别运行):
#      # 节点0 (master):
#      MASTER_ADDR=node0_ip NUM_NODES=4 NODE_RANK=0 ./scripts/launch_multinode.sh
#      # 节点1:
#      MASTER_ADDR=node0_ip NUM_NODES=4 NODE_RANK=1 ./scripts/launch_multinode.sh
#      # ...以此类推
#
#   3. 使用pdsh批量启动:
#      pdsh -w node[0-3] "cd /path/to/streaming-vlm && MASTER_ADDR=node0 NUM_NODES=4 NODE_RANK=\$(echo \$HOSTNAME | grep -oP '\d+') ./scripts/launch_multinode.sh"
########################################################

set -e

# ============== 配置 ==============
NUM_NODES=${NUM_NODES:-4}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
STAGE=${STAGE:-1}  # 1 或 2

echo "=============================================="
echo "Launching multi-node training"
echo "  NUM_NODES: ${NUM_NODES}"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  STAGE: ${STAGE}"
echo "=============================================="

# 激活环境
source ~/.bashrc
conda activate streamingvlm-sft

# 切换到项目目录
cd "$(dirname "$0")/.."

# 导出环境变量供子脚本使用
export NUM_NODES
export NODE_RANK
export GPUS_PER_NODE
export MASTER_ADDR
export MASTER_PORT

# 选择Stage
if [ "$STAGE" == "1" ]; then
    bash scripts/sft_stage_1_multinode.sh
elif [ "$STAGE" == "2" ]; then
    bash scripts/sft_stage_2_multinode.sh
else
    echo "Invalid STAGE: ${STAGE}. Must be 1 or 2."
    exit 1
fi
