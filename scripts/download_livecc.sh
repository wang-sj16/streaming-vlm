#!/bin/bash

# 下载 Live-WhisperX-526K 数据集到指定目录
# 如果下载失败，自动重试

LOCAL_DIR="data/Inf-Stream-Train/Livecc_sft"
REPO_ID="chenjoya/Live-WhisperX-526K"
MAX_RETRIES=100  # 最大重试次数
RETRY_DELAY=3    # 重试间隔（秒）

# 创建目标目录
mkdir -p "$LOCAL_DIR"

attempt=1
while true; do
    echo "=========================================="
    echo "下载尝试 #$attempt"
    echo "=========================================="
    
    huggingface-cli download "$REPO_ID" \
        --repo-type dataset \
        --local-dir "$LOCAL_DIR"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "=========================================="
        echo "下载成功！"
        echo "=========================================="
        break
    else
        echo "=========================================="
        echo "下载失败 (退出码: $exit_code)"
        echo "等待 ${RETRY_DELAY} 秒后重试..."
        echo "=========================================="
        sleep $RETRY_DELAY
        ((attempt++))
        
        if [ $attempt -gt $MAX_RETRIES ]; then
            echo "已达到最大重试次数 ($MAX_RETRIES)，退出"
            exit 1
        fi
    fi
done

echo "数据集已下载到: $LOCAL_DIR"
