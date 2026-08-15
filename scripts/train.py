"""
Single-GPU training entrypoint for the asset-alignment VLA.

Trains `AssetVLA` with gradient accumulation, periodic validation, and safetensors
checkpointing, driven by the config and data pipeline in `src/asset_vla/training/`.

Usage:
  python scripts/train.py <config_name> --exp-name <run_name>
  Example:
  python scripts/train.py assets_aligner --exp-name my_run
  python scripts/train.py assets_aligner --exp-name my_run --resume  # Resume from latest checkpoint

Checkpoints are written to `<checkpoint_base_dir>/<config_name>/<exp_name>/<step>/`.
"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm
import wandb

import asset_vla.model.vla
import asset_vla.shared.normalize as _normalize
import asset_vla.training.config as _config
import asset_vla.training.data_loader as _data

import asset_vla.transforms as transforms
from asset_vla.training import checkpoints as _checkpoints
from asset_vla.geometry import geodesic_degrees



def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(config: _config.TrainConfig, split="train"):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False, split=split)
    return data_loader, data_loader.data_config()


def save_checkpoint(model, optimizer, global_step, config, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        safetensors.torch.save_model(model, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            safetensors.torch.load_model(model, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB"
    )

def count_trainable_parameters(model):
      return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def relative_rmse(pred, target):
    gt_norm = torch.sqrt(torch.mean(target ** 2, dim=(1, 2))) + 1e-8
    batched_rmse = torch.sqrt(torch.mean((pred - target) ** 2, dim=(1, 2))) / gt_norm
    return batched_rmse.mean()


@torch.no_grad()
def get_metrics(gt_actions, pred_actions, transform_gt, transform_pred):
    # calculate relative translation 

    gt_translations, gt_rotations =  gt_actions[..., :3], gt_actions[..., 3:]
    pred_translations, pred_rotations =  pred_actions[..., :3], pred_actions[..., 3:]

    gt_translations_dict = {"translations": gt_translations.detach().cpu()}
    pred_translations_dict = {"translations": pred_translations.detach().cpu()}

    gt_translations_unnormalized = transform_gt(gt_translations_dict)["translations"]
    pred_translations_unnormalized = transform_pred(pred_translations_dict)["translations"]

    translations_metric = relative_rmse(gt_translations_unnormalized, pred_translations_unnormalized)
    rotations_metric = geodesic_degrees(gt_rotations, pred_rotations)

    return translations_metric.mean(), rotations_metric.mean()

@torch.no_grad()
def run_validation(model, val_loader, transform_gt, transform_pred, device):
    val_step = 0
    pbar = (
        tqdm.tqdm(desc="Validation")
    )

    was_training = model.training
    model.eval()

    losses_rot = []
    losses_trans = []
    losses = []

    metric_trans = []
    metric_rot = []

    for observation, translations, rotations in val_loader:
        # Check if we've reached the target number of steps
        observation = jax.tree.map(lambda x: x.to(device), observation)

        actions = torch.cat([translations, rotations], dim=-1)
        gt_actions = actions.to(device, dtype=torch.float32)
        pred_actions = model.sample_actions(device, observation)

        # Forward / loss
        val_losses = model(observation, gt_actions)
        val_rot_loss = val_losses["rot_mse"].mean()
        val_trans_loss = val_losses["trans_mse"].mean()
        val_loss = val_rot_loss + val_trans_loss

        # Metrics
        trans_metric, rot_metric = get_metrics(gt_actions, pred_actions, transform_gt, transform_pred)
        
        losses.append(val_loss)
        losses_rot.append(val_rot_loss)
        losses_trans.append(val_trans_loss)
        metric_trans.append(trans_metric)
        metric_rot.append(rot_metric)
    
        pbar.set_postfix(
            loss_trans=f"{val_trans_loss.item():.4f}",
            loss_rot=f"{val_rot_loss.item():.4f}",
            loss=f"{val_loss.item():.4f}",
            trans_metric=f"{trans_metric.item():.4f}",
            rot_metric=f"{rot_metric.item():.4f}"
        )

        pbar.update(1)
        val_step += 1

    mean_loss = torch.stack(losses).mean()
    mean_loss_rot = torch.stack(losses_rot).mean()
    mean_loss_trans = torch.stack(losses_trans).mean()
    mean_metric_trans = torch.stack(metric_trans).mean()
    mean_metric_rot = torch.stack(metric_rot).mean()

    if was_training:
        model.train()

    return mean_loss.item(), mean_loss_rot.item(), mean_loss_trans.item(), mean_metric_trans.item(), mean_metric_rot.item()


def train_loop(config: _config.TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    set_seed(config.seed)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    effective_batch_size = config.batch_size_train * config.accumulation_steps
    logging.info(
        f"Using gradient accumulation: {config.accumulation_steps}, effective batch_size: {effective_batch_size}"
    )

    loader, data_config = build_datasets(config, split="train")
    norm_stats_train = _checkpoints.load_norm_stats(config.assets_dirs, data_config.asset_id, "train")
    val_loader, data_config = build_datasets(config, split="validation")
    norm_stats_val = _checkpoints.load_norm_stats(config.assets_dirs, data_config.asset_id, "validation")

    unnormalize_train = transforms.Unnormalize(norm_stats_train, use_quantiles=data_config.use_quantile_norm, strict=False)
    unnormalize_val = transforms.Unnormalize(norm_stats_val, use_quantiles=data_config.use_quantile_norm, strict=False) 

    # Build model
    model_cfg = config.model
    # Update dtype to match pytorch_training_precision
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = asset_vla.model.vla.AssetVLA(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # ==========================================
    # PARAMETER SPLITTING & FREEZING LOGIC
    # ==========================================
    
    # 1. Trans Projectors & 2. Rot Projectors
    trans_proj_params = list(model.trans_branch.action_in_proj.parameters()) + list(model.trans_branch.action_out_proj.parameters())
    rot_proj_params = list(model.rot_branch.action_in_proj.parameters()) + list(model.rot_branch.action_out_proj.parameters())

    # 3. VLM Backbone (Needs a tiny LR)
    vlm_params = list(model.qwen_with_expert.mibot.vlm.parameters())

    # 4. Rotation DiT Backbone
    rot_params = [p for p in model.rot_branch.dit.parameters()] + rot_proj_params

    # 5. Translation DiT Backbone
    trans_params = [p for p in model.trans_branch.dit.parameters()] + trans_proj_params


    trainable = count_trainable_parameters(model)
    total = count_all_parameters(model)
    logging.info(f"Model trainable params: {trainable:,} / Model total params: {total:,} ({100 * trainable / total:.4f}%)")

    # ==========================================
    # OPTIMIZER SETUP
    # ==========================================
    accumulation_steps = config.accumulation_steps
    warmup_steps = config.lr_schedule.warmup_steps
    vlm_warmup_steps = config.lr_schedule.vlm_warmup_steps
    vlm_lr_multiplier = config.lr_schedule.vlm_lr_multiplier
    decay_steps = config.lr_schedule.decay_steps
    peak_lr = config.lr_schedule.peak_lr
    end_lr = config.lr_schedule.decay_lr

    optim = torch.optim.AdamW(
        [
            {"params": trans_params},     # Group 0: Trans 
            {"params": rot_params},       # Group 1: Rot 
            {"params": vlm_params},       # Group 2: VLM
        ],
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


    def compute_lr(step: int, group_warmup_steps: int, group_decay_steps: int, group_peak_lr: float, group_end_lr: float):
        if step < group_warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = group_peak_lr / (group_warmup_steps + 1)
            return init_lr + (group_peak_lr - init_lr) * step / max(1, group_warmup_steps)
        
        # Cosine decay
        progress = min(1.0, (step - group_warmup_steps) / max(1, group_decay_steps - group_warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return group_end_lr + (group_peak_lr - group_end_lr) * cos
    
    def get_lrs(step: int):
        # 1. DiTs (Trans & Rot) get standard warmup and full peak_lr starting at step 0
        dit_lr = compute_lr(
            step=step, 
            group_warmup_steps=warmup_steps, 
            group_decay_steps=decay_steps,
            group_peak_lr=peak_lr, 
            group_end_lr=end_lr
        )
        
        # 2. VLM gets completely frozen during the DiT warmup
        if step < warmup_steps:
            vlm_lr = 0.0
        else:
            # VLM starts its own linear warmup AFTER the DiT warmup phase
            vlm_active_step = step - warmup_steps
            vlm_active_decay_steps = decay_steps - warmup_steps
            
            vlm_lr = compute_lr(
                step=vlm_active_step, 
                group_warmup_steps=vlm_warmup_steps, 
                group_decay_steps=vlm_active_decay_steps,
                group_peak_lr=peak_lr * vlm_lr_multiplier, 
                group_end_lr=end_lr * vlm_lr_multiplier
            )
            
        return dit_lr, vlm_lr
    

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    # ---------------------------------------------------------
    # CORRECT FREEZE INITIALIZATION (Handles Checkpoint Resume)
    # ---------------------------------------------------------
    current_opt_step = global_step // accumulation_steps
    if current_opt_step < warmup_steps:
        logging.info("Initializing VLM as FROZEN (Waiting for DiT warmup)")
        model.qwen_with_expert.mibot.vlm.requires_grad_(False)
    else:
        logging.info("Initializing VLM as UNFROZEN (Past DiT warmup phase)")
        model.qwen_with_expert.mibot.vlm.requires_grad_(True)

    # LOGGING PARTS 
    vlm = model.qwen_with_expert.mibot.vlm
    trainable = count_trainable_parameters(vlm)
    total = count_all_parameters(vlm)

    logging.info(
        f"VLM — Trainable params: {trainable:,} | "
        f"Total params: {total:,} | "
        f"Trainable %: {100 * trainable / total:.4f}%"
    )

    dit_trans = model.trans_branch.dit
    trainable = count_trainable_parameters(dit_trans)
    total = count_all_parameters(dit_trans)

    logging.info(
        f"dit_trans — Trainable params: {trainable:,} | "
        f"Total params: {total:,} | "
        f"Trainable %: {100 * trainable / total:.4f}%"
    )

    dit_rot = model.rot_branch.dit
    trainable = count_trainable_parameters(dit_rot)
    total = count_all_parameters(dit_rot)

    logging.info(
        f"dit_rot — Trainable params: {trainable:,} | "
        f"Total params: {total:,} | "
        f"Trainable %: {100 * trainable / total:.4f}%"
    )

    trainable = count_trainable_parameters(model)
    total = count_all_parameters(model)

    logging.info(
        f"Modal trainable params: {trainable:,} / Modal total params: {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
    logging.info(
        f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
    )
    logging.info(
        f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
    )
    logging.info("EMA is not supported for PyTorch training")
    logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training")

    running_loss_rot = 0
    running_loss_trans = 0

    while global_step < config.num_train_steps:
        for observation, translations, rotations in loader:
            if global_step >= config.num_train_steps:
                break
            
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = torch.cat([translations, rotations], dim=-1).to(device, dtype=torch.float32)
        
            # Forward pass
            losses = model(observation, actions)

            loss_rot = losses["rot_mse"].mean()
            loss_trans = losses["trans_mse"].mean()
            loss = loss_trans + loss_rot

            running_loss_rot += loss_rot.item() / accumulation_steps
            running_loss_trans += loss_trans.item() / accumulation_steps
            loss_scaled = loss / accumulation_steps

            # Backward pass
            loss_scaled.backward()
            
            # Log memory usage after backward pass
            if global_step < 5 and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            iter_inner = (global_step + 1) % accumulation_steps

            if iter_inner == 0:
                opt_step = (global_step + 1) // accumulation_steps
                logging.info(f"*** Opt Step {opt_step} (Batch {global_step+1})")

                if opt_step == warmup_steps:
                    logging.info("Unfreezing VLM and starting VLM warmup!")
                    model.qwen_with_expert.mibot.vlm.requires_grad_(True)

                # Apply LRs to the 4 groups
                backbone_lr, vlm_lr = get_lrs(opt_step)

                optim.param_groups[0]["lr"] = backbone_lr
                optim.param_groups[1]["lr"] = backbone_lr
                optim.param_groups[2]["lr"] = vlm_lr


                trans_dit_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.trans_branch.dit.parameters(), max_norm=float('inf')
                ).item()
                
                rot_dit_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.rot_branch.dit.parameters(), max_norm=float('inf')
                ).item()

                vlm_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.qwen_with_expert.mibot.vlm.parameters(), max_norm=float('inf')
                ).item()
            
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

                # Optimizer step
                optim.step()
                optim.zero_grad(set_to_none=True)

                # Clear gradients more aggressively
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad = None


                # Collect stats
                running_loss = running_loss_trans + running_loss_rot

                infos.append({
                    "loss": running_loss,
                    "loss_trans": running_loss_trans,
                    "loss_rot": running_loss_rot,
                    "lr_trans": optim.param_groups[0]["lr"],
                    "lr_rot": optim.param_groups[1]["lr"],
                    "lr_vlm": optim.param_groups[2]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "grad_norm_trans_dit": trans_dit_grad_norm,
                    "grad_norm_rot_dit": rot_dit_grad_norm,
                    "grad_norm_vlm": vlm_grad_norm,
                })

                running_loss = 0
                running_loss_trans = 0
                running_loss_rot = 0

                # LOGGING: Trigger based on opt_step instead of global_step
                if opt_step > 0 and (opt_step % config.log_interval == 0):
                    if len(infos) > 0:
                        elapsed = time.time() - start_time
                        avg_loss = sum(info["loss"] for info in infos) / len(infos)
                        avg_loss_trans = sum(info["loss_trans"] for info in infos) / len(infos)
                        avg_loss_rot = sum(info["loss_rot"] for info in infos) / len(infos)
                        
                        avg_lr_trans = sum(info["lr_trans"] for info in infos) / len(infos)
                        avg_lr_rot   = sum(info["lr_rot"] for info in infos) / len(infos)
                        avg_lr_vlm   = sum(info["lr_vlm"] for info in infos) / len(infos)
                  
                        model.eval()
                        pred_actions = model.sample_actions(device, observation)
                        metric_trans, metric_rot = get_metrics(actions, pred_actions, unnormalize_train, unnormalize_train)
                        model.train()

                        avg_metric_trans = metric_trans.item()
                        avg_metric_rot = metric_rot.item()

                        # Safely average gradient norms (ignoring Nones)
                        avg_grad_norm = None
                        valid_grad_norms = [info["grad_norm"] for info in infos if info.get("grad_norm") is not None]
                        if valid_grad_norms: 
                            avg_grad_norm = sum(valid_grad_norms) / len(valid_grad_norms)
                        
                        valid_t_grads = [i["grad_norm_trans_dit"] for i in infos if i.get("grad_norm_trans_dit") is not None]
                        avg_grad_trans_dit = sum(valid_t_grads) / len(valid_t_grads) if valid_t_grads else 0.0
                        
                        valid_r_grads = [i["grad_norm_rot_dit"] for i in infos if i.get("grad_norm_rot_dit") is not None]
                        avg_grad_rot_dit = sum(valid_r_grads) / len(valid_r_grads) if valid_r_grads else 0.0
                        
                        valid_vlm_grads = [i["grad_norm_vlm"] for i in infos if i.get("grad_norm_vlm") is not None]
                        avg_grad_vlm = sum(valid_vlm_grads) / len(valid_vlm_grads) if valid_vlm_grads else 0.0
                        
                        logging.info(
                            f"opt_step={opt_step} (batch={global_step+1}) loss={avg_loss:.4f} loss_trans={avg_loss_trans:.4f} "
                            f"loss_rot={avg_loss_rot:.4f} metric_trans={avg_metric_trans:.4f} metric_rot={avg_metric_rot:.4f} "
                            f"lr_trans={avg_lr_trans:.2e} lr_rot={avg_lr_rot:.2e} lr_vlm={avg_lr_vlm:.2e} "
                            f"grad_t={avg_grad_trans_dit:.3f} grad_r={avg_grad_rot_dit:.3f} avg_grad_vlm={avg_grad_vlm:.3f}"
                        )

                        if config.wandb_enabled:
                            log_payload = {
                                "loss": avg_loss,
                                "loss_trans": avg_loss_trans,
                                "loss_rot": avg_loss_rot,
                                "trans_metric": avg_metric_trans,
                                "rot_metric": avg_metric_rot,
                                "lr/trans_backbone": avg_lr_trans,
                                "lr/rot_backbone": avg_lr_rot,
                                "lr/vlm": avg_lr_vlm,
                                "grad_norm/trans_dit": avg_grad_trans_dit,
                                "grad_norm/rot_dit": avg_grad_rot_dit,
                                "grad_norm/vlm": avg_grad_vlm,
                                "global_step": global_step + 1,
                                "time_per_opt_step": elapsed / len(infos),
                            }
                            if avg_grad_norm is not None: 
                                log_payload["grad_norm"] = avg_grad_norm
                                
                            # Log using opt_step as the X-axis
                            wandb.log(log_payload, step=opt_step)

                    # Reset timer and stats
                    start_time = time.time()
                    infos = []
        
                # VALIDATION: Also trigger based on opt_step
                if opt_step > 0 and (opt_step % config.val_interval == 0):
                    val_loss, val_loss_rot, val_loss_trans, val_metric_trans, val_metric_rot = run_validation(
                        model, val_loader, unnormalize_val, unnormalize_train, device
                    )
                    if config.wandb_enabled:
                        wandb.log({
                            "val/loss": val_loss, 
                            "val/loss_trans": val_loss_trans, 
                            "val/loss_rot": val_loss_rot,
                            "val/metric_trans": val_metric_trans, 
                            "val/metric_rot": val_metric_rot,
                        }, step=opt_step)

            global_step += 1
            
            opt_step_current = global_step // accumulation_steps
            if (global_step % accumulation_steps == 0) and opt_step_current > 0 and (opt_step_current % config.save_interval == 0):
                save_checkpoint(model, optim, opt_step_current, config, data_config)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "L": f"{loss.item():.4f}",       
                    "Lt": f"{loss_trans.item():.4f}", 
                    "Lr": f"{loss_rot.item():.4f}",   
                    "opt_step": global_step // accumulation_steps
                })

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if config.wandb_enabled:
        wandb.finish()

def main():
    init_logging()
    config = _config.cli()
    
    train_loop(config)


if __name__ == "__main__":
    main()
