# data preprocessing
# python evaluation/ovobench/transfer_annotation_format.py --input .../ovobench/ovo_bench_new.json --output .../ovobench/ovo-bench-formatted.jsonl

python streaming_vlm/eval/ovobench/transfer_annotation_format.py --input data/ovobench/ovo_bench_new.json --output data/ovobench/ovo-bench-formatted.jsonl

# evaluation
conda activate streamingvlm-ovo
torchrun --standalone --nproc_per_node=8 streaming_vlm/eval/ovobench/distributed_evaluate_ovobench.py --benchmark_dir data/ovobench --model_path models/StreamingVLM