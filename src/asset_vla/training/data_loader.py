from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import asset_vla.model.observation as _model
from asset_vla.model.config import ModelConfig
import asset_vla.training.config as _config
import asset_vla.transforms as _transforms
import torchvision.transforms as T
from datasets import load_dataset
import huggingface_hub
from pathlib import Path
import pandas as pd
from PIL import Image
import io

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)


class FakeDataset(Dataset):
    def __init__(self, model_config: ModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _filter_actions_len_9(example):
    """Module-level function so it can be pickled by DataLoader workers."""
    return len(example["action"]) == 9


def _load_index_map(auxiliary_repo_id: str, index_map_path: str) -> dict[tuple[str, int], int]:
    """Map (asset_name, part_idx) to a row index in the auxiliary rendering dataset.

    A local file wins when present; otherwise the CSV is pulled from the auxiliary Hugging
    Face repo and cached, so only the first run touches the network.
    """
    if not Path(index_map_path).exists():
        index_map_path = huggingface_hub.hf_hub_download(
            repo_id=auxiliary_repo_id, filename=index_map_path, repo_type="dataset"
        )
        logging.info(f"Downloaded index map from {auxiliary_repo_id}")

    df = pd.read_csv(index_map_path)
    return {
        (row.asset_name, row.part_idx): row.row_idx_rendering
        for row in df.itertuples(index=False)
    }


class LeRobotWrapper(torch.utils.data.IterableDataset):
    def __init__(
        self,
        repo_id,
        action_horizon,
        fps: float = 10.0,
        task: str = "",
        split: str = "train",
        data_dir: str = "data",
        shuffle: bool = False,
        seed: int = 42,
        buffer_size: int = 2_000,
        auxiliary_repo_id: str | None = None,
        index_map_path: str | None = None,
        gt_rendering_angles: Sequence[int] = (30, 90, 270),
        gt_rendering_planes: Sequence[str] = ("XY", "ZY"),
    ):
        super().__init__()

        self.H = action_horizon
        self.fps = fps
        self.task = task
        self.data_dir = data_dir
        self.split = split

        # Load the stream
        ds = load_dataset(repo_id, split=split, streaming=True)
        ds = ds.filter(_filter_actions_len_9)

        if split == "train":
            ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

        self.ds = ds

        # Optional auxiliary dataset of ground-truth renderings, used as reference views.
        # It is indexed by row, so `index_map_path` maps (asset_name, part_idx) -> row index.
        # Note this dataset is loaded eagerly rather than streamed, so random access is cheap.
        self.gt_rendering_angles = list(gt_rendering_angles)
        self.gt_rendering_planes = list(gt_rendering_planes)
        if auxiliary_repo_id is not None and index_map_path is not None:
            self.ds_auxiliary = load_dataset(auxiliary_repo_id, split="train")
            self.index_map = _load_index_map(auxiliary_repo_id, index_map_path)
        else:
            self.ds_auxiliary = None
            self.index_map = None

        self.image_transform = T.Compose([
            T.Resize((256, 256)),
            T.ConvertImageDtype(torch.float32),
        ])

    def pil_to_tensor(self, img) -> torch.Tensor:
        # 1. Catch the raw HF dictionary and decode the bytes into a PIL Image
        if isinstance(img, dict):
            if "bytes" in img and img["bytes"] is not None:
                img = Image.open(io.BytesIO(img["bytes"]))
            elif "path" in img and img["path"] is not None:
                img = Image.open(img["path"])
                
        # 2. Process normally
        if img.mode != "RGB":
            img = img.convert("RGB")
        t = T.functional.pil_to_tensor(img)
        return self.image_transform(t)

    def tif_to_tensor(self, img) -> torch.Tensor:
        # 1. Catch the raw HF dictionary for depth maps too
        if isinstance(img, dict):
            if "bytes" in img and img["bytes"] is not None:
                img = Image.open(io.BytesIO(img["bytes"]))
            elif "path" in img and img["path"] is not None:
                img = Image.open(img["path"])
                
        # 2. Process normally
        t = T.functional.pil_to_tensor(img)
        return t

    def __iter__(self):
        for idx, s in enumerate(self.ds):
            # ---- images ----
            right_image = self.pil_to_tensor(s["right_image"])
            back_image  = self.pil_to_tensor(s["back_image"])
            upper_image = self.pil_to_tensor(s["upper_image"])

            # ---- depths / point clouds ----
            right_depth = self.tif_to_tensor(s["right_depth"])
            back_depth  = self.tif_to_tensor(s["back_depth"])
            upper_depth = self.tif_to_tensor(s["upper_depth"])

            # ---- reference views: ground-truth renderings of the aligned pair ----
            rendered_images_tensor = None
            if self.ds_auxiliary is not None and self.index_map is not None:
                lookup_key = (s["asset_name"], s["part_idx"])
                if lookup_key in self.index_map:
                    render_s = self.ds_auxiliary[self.index_map[lookup_key]]
                    rendered_images = [
                        self.pil_to_tensor(render_s[f"image_{plane}_{degree}"])
                        for plane in self.gt_rendering_planes
                        for degree in self.gt_rendering_angles
                    ]
                    if rendered_images:
                        rendered_images_tensor = torch.stack(rendered_images)

            # ---- extrinsics ----
            camera_extrinsics = torch.tensor(s["camera_extrinsics"], dtype=torch.float32)

            # ---- actions ----
            a = torch.tensor(s["action"], dtype=torch.float32)
            actions = a.unsqueeze(0).repeat(self.H, 1)

            actions_is_pad = torch.ones(self.H, dtype=torch.bool)
            actions_is_pad[0] = False

            tgt_center = torch.tensor(s["tgt_center"], dtype=torch.float32)
            glb_path = Path(self.data_dir) / s["asset_name"] / f"part_{s['part_idx']}.glb"

            # ---- metadata ----
            sample = {
                "right_image": right_image,
                "back_image": back_image,
                "upper_image": upper_image,
                "right_depth": right_depth,
                "back_depth": back_depth,
                "upper_depth": upper_depth,
                "rendered_images": rendered_images_tensor,
                "camera_extrinsics": camera_extrinsics,
                "state": tgt_center,
                "glb_path": str(glb_path),
                "asset_name": s["asset_name"],
                "part_idx": torch.tensor(int(s["part_idx"]), dtype=torch.int64),
                "actions": actions,
                "actions_is_pad": actions_is_pad,
                "timestamp": torch.tensor(idx / self.fps, dtype=torch.float32),
                "frame_index": torch.tensor(idx, dtype=torch.int64),
                "episode_index": torch.tensor(0, dtype=torch.int64),
                "index": torch.tensor(idx, dtype=torch.int64),
                "task_index": torch.tensor(0, dtype=torch.int64),
            }

            yield sample


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: ModelConfig,
    split
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    task = "Align red source (src) to blue target (tgt)."
    dataset = LeRobotWrapper(
        data_config.repo_id,
        action_horizon,
        split=split,
        task=task,
        auxiliary_repo_id=data_config.auxiliary_repo_id,
        index_map_path=data_config.index_map_path,
    )

    instruction_parts = [
        "<|im_start|>user\nThe following observations are captured from multiple views.\n",
        "# Right View\n<|vision_start|><|image_pad|><|vision_end|>\n",
        "# Back View\n<|vision_start|><|image_pad|><|vision_end|>\n",
        "# Upper View\n<|vision_start|><|image_pad|><|vision_end|>\n",
    ]

    # One vision tag per reference view, in the same order the loader stacks them.
    # The count must match, or the tokenized prompt and the image batch disagree.
    if dataset.ds_auxiliary is not None:
        instruction_parts.append(
            "There are also ground truth views that represent aligned src and tgt from different viewpoints:\n"
        )
        for plane in dataset.gt_rendering_planes:
            for degree in dataset.gt_rendering_angles:
                instruction_parts.append(
                    f"# Render {plane} Plane {degree}°\n<|vision_start|><|image_pad|><|vision_end|>\n"
                )

    instruction_parts.append(
        f"Generate robot actions for the task:\n{task} /no_cot<|im_end|>\n"
        f"<|im_start|>assistant\n<cot></cot><|im_end|>\n"
    )

    tasks = {0: "".join(instruction_parts)}
    if data_config.prompt_from_task:
        dataset = IterableTransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                f"Normalization stats not found for {data_config.repo_id}, and they could not be "
                "downloaded from the dataset repo. Check network access and `huggingface-cli login` "
                "for private repos, or place them under the config's assets directory manually."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                f"Normalization stats not found for {data_config.repo_id}, and they could not be "
                "downloaded from the dataset repo. Check network access and `huggingface-cli login` "
                "for private repos, or place them under the config's assets directory manually."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    split="train",
    norm_stats_split: str | None = None,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    drop_last: bool = True,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        split: The dataset split to read.
        norm_stats_split: Which split's norm stats to normalize with. Defaults to `split`.
            Pass "train" to normalize every split the way the model was trained; the model
            expects train-normalized inputs, so this is what evaluation wants.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        drop_last: Whether to discard a final partial batch. Keep True for training, where
            uniform batch shapes matter; pass False when sweeping a split end to end, so the
            last few samples are not silently dropped.
    """
    # The split reaches DataConfig only to pick norm stats; the dataset split is passed
    # separately below, so the two can differ.
    data_config = config.data.create(config.assets_dirs, config.model, norm_stats_split or split)
    logging.info(f"data_config: {data_config}")

    batch_size = config.batch_size_train if split == "train" else config.batch_size_val
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        split=split,
        drop_last=drop_last,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: ModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    split: str = "train",
    drop_last: bool = True,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        drop_last: Whether to discard a final partial batch.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config, split)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Training is single-process; for JAX the batch is split across processes.
    local_batch_size = batch_size if framework == "pytorch" else batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        split=split,
        drop_last=drop_last,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        split: str,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        drop_last: bool = True,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
            drop_last: Whether to discard a final partial batch. False keeps the tail of a
                split, which matters when sweeping it end to end for evaluation.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._split = split

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=drop_last,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0

        # Outer loop controls whether we repeat the dataset (infinite dataloader)
        while True:
            for batch in self._data_loader:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                
                num_items += 1
                
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    # String metadata (e.g. asset_name) is passed through untouched.
                    yield jax.tree.map(lambda x: x if isinstance(x, str) else torch.as_tensor(x), batch)

            # If this is a validation/test split, do not loop infinitely. 
            if self._split != "train":
                break


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""

    def collate_leaf(*xs):
        # If elements are strs, keep them as a list (do not convert to numpy)
        if isinstance(xs[0], str):
            return list(xs)

        # Otherwise stack as numpy arrays
        return np.stack([np.asarray(x) for x in xs], axis=0)

    return jax.tree.map(collate_leaf, *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"



class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["translations"], batch["rotations"]
