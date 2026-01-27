#!/bin/bash

# 解压 Livecc_sft/videos 目录下的所有 tar 文件（多线程版本）

VIDEO_DIR="data/Inf-Stream-Train/Livecc_sft/videos"
NUM_JOBS=${1:-8}  # 默认 8 个并行任务，可通过参数指定

cd /fsx/home/shijie.wang/code/streaming-vlm

if [ ! -d "$VIDEO_DIR" ]; then
    echo "目录不存在: $VIDEO_DIR"
    exit 1
fi

cd "$VIDEO_DIR"

total=$(ls -1 *.tar 2>/dev/null | wc -l)

echo "共有 $total 个 tar 文件需要解压"
echo "使用 $NUM_JOBS 个并行任务"
echo "=========================================="

# 定义解压函数
extract_and_delete() {
    f="$1"
    echo "正在解压: $f"
    tar -xf "$f"
    if [ $? -eq 0 ]; then
        echo "  ✓ 解压成功，删除: $f"
        rm "$f"
    else
        echo "  ✗ 解压失败: $f"
    fi
}
export -f extract_and_delete

# 使用 xargs 并行解压
ls -1 *.tar 2>/dev/null | xargs -P "$NUM_JOBS" -I {} bash -c 'extract_and_delete "$@"' _ {}

echo "=========================================="
echo "解压完成！"
