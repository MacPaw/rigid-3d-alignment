"""
Prediction dump for the asset-alignment VLA.

Loads a trained PyTorch checkpoint, runs it over one dataset split, and writes the
per-sample ground truth and predictions to disk. Metrics are deliberately not computed
here: the dump carries everything needed to score the run offline.

Translations are written in real units. The loader normalizes every split with the *train*
split's stats, because that is the space the model was trained in and the space its outputs
come back in, so unnormalizing with those same stats recovers true units for ground truth
and predictions alike. Only the train split's norm stats are ever read. Rotations are stored
as the raw 6D representation, which is never normalized.

Usage:
  python evaluate.py <config_name> [config overrides] [eval options]
  Example:
  python evaluate.py assets_aligner --exp-name eval --pytorch-weight-path /path/to/ckpt

The config name is a subcommand and must come first; tyro applies every following
option to it. The checkpoint directory must contain `model.safetensors`.

Eval options (parsed before the config CLI):
  --split         dataset split to run over (default: test)
  --max-samples   stop after roughly this many samples, rounded up to a whole batch.
                  Required with `--split train`, which shuffles and iterates forever.
  --out           output path; written as JSON Lines if it ends in `.jsonl`,
                  otherwise as a single JSON array.
"""

import argparse
import json
import logging
import os
import sys

import huggingface_hub
import jax
import safetensors.torch
import torch
import tqdm

import asset_vla.model.vla
import asset_vla.training.config as _config
import asset_vla.training.data_loader as _data
import asset_vla.transforms as transforms
from asset_vla.training import checkpoints as _checkpoints


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


def build_datasets(config: _config.TrainConfig, split="train"):
    # Always normalize with train stats: the model was trained on train-normalized inputs,
    # so feeding it a split's own statistics would shift its input distribution.
    # drop_last=False so the final partial batch is dumped rather than silently discarded.
    data_loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=False,
        split=split,
        norm_stats_split="train",
        drop_last=False,
    )
    return data_loader, data_loader.data_config()


def write_predictions(predictions, save_path):
    """Write as JSON Lines when the path ends in `.jsonl`, otherwise as one JSON array."""
    with open(save_path, "w") as f:
        if str(save_path).endswith(".jsonl"):
            for record in predictions:
                f.write(json.dumps(record) + "\n")
        else:
            json.dump(predictions, f, indent=4)


@torch.no_grad()
def run_test(model, test_loader, unnormalize, device, save_path, max_samples=None):
    pbar = tqdm.tqdm(desc="Predicting", total=max_samples)

    was_training = model.training
    model.eval()

    all_predictions = []

    for observation, translations, rotations in test_loader:
        observation = jax.tree.map(lambda x: x.to(device) if hasattr(x, "to") else x, observation)

        actions = torch.cat([translations, rotations], dim=-1)
        gt_actions = actions.to(device, dtype=torch.float32)
        pred_actions = model.sample_actions(device, observation)

        gt_trans_cpu = gt_actions[..., :3].detach().cpu()
        gt_rot_cpu = gt_actions[..., 3:].detach().cpu()
        pred_trans_cpu = pred_actions[..., :3].detach().cpu()
        pred_rot_cpu = pred_actions[..., 3:].detach().cpu()

        gt_trans_unnorm = unnormalize({"translations": gt_trans_cpu})["translations"]
        pred_trans_unnorm = unnormalize({"translations": pred_trans_cpu})["translations"]

        batch_size = gt_trans_unnorm.shape[0]
        for b in range(batch_size):
            raw_asset = observation.asset_name[b]
            raw_part = observation.part_idx[b]
            asset_id_str = str(raw_asset)
            part_id_int = raw_part.item() if hasattr(raw_part, "item") else int(raw_part)

            all_predictions.append({
                "asset_id": asset_id_str,
                "part_id": part_id_int,
                "gt_translations": gt_trans_unnorm[b].numpy().tolist(),
                "gt_rotations": gt_rot_cpu[b].numpy().tolist(),
                "pred_translations": pred_trans_unnorm[b].numpy().tolist(),
                "pred_rotations": pred_rot_cpu[b].numpy().tolist(),
            })

        pbar.update(batch_size)

        # The train split shuffles and iterates forever, so this is the only stopping condition.
        if max_samples is not None and len(all_predictions) >= max_samples:
            break

    write_predictions(all_predictions, save_path)

    print(f"\nSaved {len(all_predictions)} unnormalized action trajectories to {save_path}")

    if was_training:
        model.train()

    return all_predictions


def test_loop(config: _config.TrainConfig, split="test", max_samples=None, save_path="test_predictions.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if split == "train" and max_samples is None:
        raise ValueError("--max-samples is required with --split train: that split iterates forever.")

    test_loader, data_config = build_datasets(config, split=split)

    norm_stats_train = _checkpoints.load_norm_stats(config.assets_dirs, data_config.asset_id, "train")
    unnormalize = transforms.Unnormalize(norm_stats_train, use_quantiles=data_config.use_quantile_norm, strict=False)

    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    model = asset_vla.model.vla.AssetVLA(model_cfg).to(device)

    # An explicit --pytorch-weight-path wins; otherwise fall back to the released checkpoint
    # named by the config, so a clean checkout can be evaluated without any local weights.
    if config.pytorch_weight_path is not None:
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        logging.info(f"Loading weights from: {model_path}")
    elif config.checkpoint_repo_id is not None:
        logging.info(f"Downloading released checkpoint from {config.checkpoint_repo_id}")
        
        # Split the string to separate the repo_id from any subfolders
        parts = config.checkpoint_repo_id.strip("/").split("/")
        
        if len(parts) > 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            subfolder_path = "/".join(parts[2:])
            target_filename = f"{subfolder_path}/model.safetensors"
        else:
            repo_id = config.checkpoint_repo_id
            target_filename = "model.safetensors"

        model_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id, 
            filename=target_filename
        )
    else:
        raise ValueError(
            "No weights to evaluate: pass --pytorch-weight-path, or set `checkpoint_repo_id` "
            "on the config to point at a released checkpoint."
        )

    safetensors.torch.load_model(model, model_path)
    logging.info(f"Loaded weights from {model_path}")

    run_test(model, test_loader, unnormalize, device, save_path, max_samples)


def parse_eval_args():
    """Consume the eval-only flags, leaving the rest of argv for the tyro config CLI.

    `add_help=False` keeps `--help` flowing through to tyro, which documents the config.
    """
    # allow_abbrev=False stops argparse from claiming prefixes of config flags meant for tyro.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--out", default="test_predictions.json")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main():
    init_logging()
    eval_args = parse_eval_args()
    config = _config.cli()

    test_loop(config, split=eval_args.split, max_samples=eval_args.max_samples, save_path=eval_args.out)


if __name__ == "__main__":
    main()
