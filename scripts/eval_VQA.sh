export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
# cd streaming_vlm/eval/VLMEvalKit/
# torchrun --nproc-per-node=8 streaming_vlm/eval/VLMEvalKit/run.py --data Video-MME_1fps --model StreamingVLM --verbose --reuse
# torchrun --nproc-per-node=8 streaming_vlm/eval/VLMEvalKit/run.py --data MVBench_64frame --model StreamingVLM --verbose --reuse
torchrun --nproc-per-node=8 streaming_vlm/eval/VLMEvalKit/run.py --data LongVideoBench_1fps --model StreamingVLM --verbose --reuse