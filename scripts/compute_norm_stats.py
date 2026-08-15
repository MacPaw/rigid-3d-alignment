"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

from asset_vla.model.config import ModelConfig
import asset_vla.shared.normalize as normalize
import asset_vla.training.config as _config
import asset_vla.training.data_loader as _data_loader
import asset_vla.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: ModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    split = "train"
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, 
                                                model_config, split=split)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None:
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        split=split,
        num_workers=num_workers,
        shuffle=False,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    
    for split in ["train", "validation", "test"]:
        data_config = config.data.create(config.assets_dirs, config.model, split)
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size_train, 
            config.model, config.num_workers, max_frames, split
        )

        keys = ["state", "translations"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

        output_path = config.assets_dirs / data_config.repo_id / split
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
