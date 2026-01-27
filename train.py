from types import MethodType
# patch incorrect liger kernel
# import liger_kernel.transformers.model.qwen2_5_vl as qwen2_5_vl
# from streaming_vlm.utils.patch_liger_kernel import lce_forward
# qwen2_5_vl.lce_forward = lce_forward

# Suppress various non-critical warnings
import logging as pylogging
import warnings

# Suppress "Unused or unrecognized kwargs: fps, return_tensors" from image_utils
pylogging.getLogger("transformers.image_utils").setLevel(pylogging.ERROR)

# Suppress "use_cache=True is incompatible with gradient checkpointing" 
pylogging.getLogger("transformers.modeling_utils").setLevel(pylogging.ERROR)

# Suppress "None of the inputs have requires_grad=True" from torch checkpoint (frozen visual encoder)
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from streaming_vlm.utils.patch_trainer import compute_loss_logging_labels

from dataclasses import asdict
import transformers
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging, TrainerCallback

from streaming_vlm.inference.qwen2_5.pos_emb import get_rope_index
import os
import torch
from models import ModelArguments
from streaming_vlm.data.lmm_dataset import DataArguments, LMMDataset, EvalDataArguments
from transformers import set_seed
import wandb

logger = logging.get_logger(__name__)

WANDB_RUN_ID_FILE = "wandb_run_id.txt"

def setup_wandb_resume(output_dir: str, resume_ckpt: str):
    """
    Setup WandB to resume from a previous run if resuming training.
    - If resuming and wandb_run_id.txt exists, set WANDB_RUN_ID and WANDB_RESUME
    - Returns the run_id if resuming, None otherwise
    """
    wandb_id_path = os.path.join(output_dir, WANDB_RUN_ID_FILE)
    
    if resume_ckpt and os.path.isfile(wandb_id_path):
        with open(wandb_id_path, 'r') as f:
            run_id = f.read().strip()
        if run_id:
            os.environ['WANDB_RUN_ID'] = run_id
            os.environ['WANDB_RESUME'] = 'allow'
            print(f"[wandb] Resuming WandB run: {run_id}")
            return run_id
    
    print(f"[wandb] Starting new WandB run")
    return None

def save_wandb_run_id(output_dir: str):
    """
    Save the current WandB run ID to a file in output_dir.
    Only saves on rank 0, and only if the file doesn't already exist.
    """
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
    if rank != 0:
        return
    
    if wandb.run is not None:
        run_id = wandb.run.id
        wandb_id_path = os.path.join(output_dir, WANDB_RUN_ID_FILE)
        
        # Only write if file doesn't exist (don't overwrite existing run ID)
        if os.path.isfile(wandb_id_path):
            print(f"[wandb] WandB run ID file already exists: {wandb_id_path}, skipping write")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        with open(wandb_id_path, 'w') as f:
            f.write(run_id)
        print(f"[wandb] Saved WandB run ID to {wandb_id_path}: {run_id}")

class WandBRunIdCallback(TrainerCallback):
    """Callback to save WandB run ID when training starts."""
    
    def on_train_begin(self, args, state, control, **kwargs):
        save_wandb_run_id(args.output_dir)
        return control

def find_resume_checkpoint(output_dir: str):
    """
    Directly search for checkpoints in output_dir (exact match, no timestamp suffix).
    Find the latest checkpoint containing trainer_state.json.
    Return None if not found.
    """
    output_dir = os.path.abspath(output_dir)
    print(f"[resume] Checking directory {output_dir}")
    
    if not os.path.isdir(output_dir):
        print(f"[resume] Directory does not exist: {output_dir}")
        return None

    # Find checkpoint-*, sort by step number in descending order
    ckpts = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-", 1)[1])
            except:
                step = -1
            ckpts.append((step, os.path.join(output_dir, name)))
    ckpts.sort(key=lambda x: x[0], reverse=True)

    for _, cp in ckpts:
        if os.path.isfile(os.path.join(cp, "trainer_state.json")):
            print(f"[resume] Resuming from {cp}")
            return cp
    
    print(f"[resume] No checkpoint found in {output_dir}")
    return None

if __name__ == "__main__":
    training_args, model_args, data_args, eval_data_args = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments, EvalDataArguments)).parse_args_into_dataclasses()

    # Simple resume strategy: find the latest checkpoint in output_dir
    resume_ckpt = find_resume_checkpoint(training_args.output_dir)
    
    # Setup WandB to resume from previous run if resuming training
    setup_wandb_resume(training_args.output_dir, resume_ckpt)

    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path, trust_remote_code=True)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
            model_args.pretrained_model_name_or_path, 
            torch_dtype="auto", attn_implementation='flash_attention_2'
        )
    model.get_rope_index = MethodType(get_rope_index, model)
    for m in ["visual", "vision_tower"]:
        try:
            getattr(model, m).requires_grad_(False)
            print(f"Freezing module {m}")
        except:
            print(f"Module {m} not found in model")

    # Always load processor from official model to ensure video_processor is properly initialized
    # Loading from checkpoint may have incomplete processor configs (missing video_preprocessor.json)
    if 'Qwen2_5_VL' in model.config.architectures[0]:
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', padding_side='right')
    elif 'Qwen2VL' in model.config.architectures[0]:
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', padding_side='right')
    else:
        processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, padding_side='right', trust_remote_code=True)


    train_dataset = LMMDataset(**asdict(data_args), **asdict(training_args), **asdict(model_args), processor=processor)
    eval_dataset = LMMDataset(**asdict(data_args), **asdict(eval_data_args), **asdict(training_args), **asdict(model_args), processor=processor
        )
    # Add after model is built but before Trainer
    if hasattr(model, "llm_model_embed_tokens"):
        print("delattr llm_model_embed_tokens")
        delattr(model, "llm_model_embed_tokens")

    # Make same-name access a property that returns the actual weights (still shared, no extra memory usage)
    setattr(type(model), "llm_model_embed_tokens", property(lambda self: self.llm.model.embed_tokens))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.data_collator,
        processing_class=processor,
        callbacks=[WandBRunIdCallback()]
    )
    trainer.compute_loss = MethodType(compute_loss_logging_labels, trainer)
    # Pass specific path or False depending on whether resuming training
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else False)
