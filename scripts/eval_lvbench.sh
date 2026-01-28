export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
cd streaming_vlm/eval/VLMEvalKit/
torchrun --nproc-per-node=4 run.py --data LVBench_1fps --model StreamingVLM --verbose --reuse
torchrun --nproc-per-node=4 run.py --data LVBench_1fps --model Qwen2.5-VL-7B-Instruct-ForVideo --verbose --reuse