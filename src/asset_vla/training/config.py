"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol

import etils.epath as epath
import huggingface_hub
from typing_extensions import override
import tyro

from asset_vla.model.config import ModelConfig
import asset_vla.model.tokenizer as _tokenizer
import asset_vla.policies.alignment_policy as alignment_policy
import asset_vla.shared.normalize as _normalize
import asset_vla.training.optimizer as _optimizer
import asset_vla.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Optional dataset of ground-truth renderings used as extra reference views.
    auxiliary_repo_id: str | None = None
    # CSV mapping (asset_name, part_idx) to a row index in the auxiliary dataset.
    index_map_path: str | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False


def _download_norm_stats(
    assets_dir: epath.Path, asset_id: str, split: str | None
) -> dict[str, _transforms.NormStats] | None:
    """Fetch norm stats from the dataset repo, mirroring the local assets layout.

    The dataset repo stores them under the same relative path used on disk, i.e.
    `assets/<config-name>/<asset_id>/<split>/norm_stats.json`.
    """
    parts = ["assets", pathlib.Path(str(assets_dir)).name, asset_id]
    if split is not None:
        parts.append(split)
    filename = "/".join([*parts, "norm_stats.json"])

    try:
        local_path = huggingface_hub.hf_hub_download(repo_id=asset_id, filename=filename, repo_type="dataset")
    except Exception:
        logging.warning(f"Could not download '{filename}' from the dataset repo '{asset_id}'.")
        return None

    logging.info(f"Downloaded norm stats from {asset_id}:{filename}")
    return _normalize.load(pathlib.Path(local_path).parent)


class GroupFactory(Protocol):
    def __call__(self, model_config: ModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates the transforms applied to data after normalization, just before the model."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: ModelConfig) -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(256, 256),
                _transforms.TokenizePrompt(
                    _tokenizer.XiaomiTokenizer(model_config.max_token_len),
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: ModelConfig, split=None) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: ModelConfig, split=None) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id, split),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None, split) -> dict[str, _transforms.NormStats] | None:
        """Load norm stats, downloading them from the dataset repo if they are not on disk."""
        if asset_id is None:
            return None
        try:
            if split is not None:
                data_assets_dir = str(assets_dir / asset_id / split)
            else:
                data_assets_dir = str(assets_dir / asset_id)

            norm_stats = _normalize.load(data_assets_dir)
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, fetching from Hugging Face.")

        return _download_norm_stats(assets_dir, asset_id, split)


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: ModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class AlignmentDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of my data pipeline.
    """

    # Optional dataset of ground-truth renderings used as extra reference views.
    auxiliary_repo_id: str | None = None
    # CSV mapping (asset_name, part_idx) to a row index in the auxiliary dataset.
    index_map_path: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: ModelConfig, split = None) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "right_image": "right_image",
                        "back_image": "back_image",
                        "upper_image": "upper_image",
                        "state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "camera_extrinsics": "camera_extrinsics",
                        "rendered_images": "rendered_images",
                        # Sample identity, carried through for evaluation only.
                        "asset_name": "asset_name",
                        "part_idx": "part_idx"
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[alignment_policy.AlignmentInputs()],
            outputs=[alignment_policy.AlignmentOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config, split),
            prompt_from_task=True,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            auxiliary_repo_id=self.auxiliary_repo_id,
            index_map_path=self.index_map_path,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "asset_vla"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see ModelConfig.
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    batch_size_train: int = 64
    batch_size_val: int = 128
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # How often (in steps) to validate model.
    val_interval: int = 100

    accumulation_steps: int = 16
    
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1
    
    # Repo id for storing new weights
    # Hugging Face repo holding the released checkpoint for this config. `predict.py` falls
    # back to it when --pytorch-weight-path is not given. Never used for training, so a
    # training run always starts from the Xiaomi pretrain unless told otherwise.
    checkpoint_repo_id: str | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
# Asset-base pairs with ground-truth rigid transformations, canonical multi-view renderings,
# depth, and Set-of-Marks visual prompts.
MAIN_DATASET = "macpaw-research/asset-alignment-pairs-905k"
# Renderings of the correctly assembled scene, used as reference views.
REFERENCE_DATASET = "macpaw-research/asset-alignment-reference-views"

_CONFIGS = [
    #
    # Feed-forward VLA, reference-free: the model sees only the three canonical views of the
    # unaligned pair. This is the headline setting of the paper.
    #
    TrainConfig(
        name="assets_aligner",
        model = ModelConfig(max_token_len=100),
        data = AlignmentDataConfig(
            repo_id = MAIN_DATASET,
        ),
        num_train_steps = 112_500,
        log_interval = 20,
        val_interval = 703,
        wandb_enabled = True,
        batch_size_train = 8,
        batch_size_val = 8,
        save_interval = 703,
        num_workers = 3,
        accumulation_steps = 32,
        checkpoint_repo_id = "macpaw-research/qwen3-dual-dit-aligner/10608/",
    ),
    #
    # Feed-forward VLA with reference views: the prompt additionally carries six renderings of
    # the correctly assembled scene, so the token budget is larger.
    #
    TrainConfig(
        name="assets_aligner_ref",
        model = ModelConfig(max_token_len=270),
        data = AlignmentDataConfig(
            repo_id = MAIN_DATASET,
            auxiliary_repo_id = REFERENCE_DATASET,
            index_map_path = "metadata/index_map.csv",
        ),
        num_train_steps = 112_500,
        log_interval = 20,
        val_interval = 703,
        wandb_enabled = True,
        batch_size_train = 8,
        batch_size_val = 8,
        save_interval = 703,
        num_workers = 3,
        accumulation_steps = 32,
        checkpoint_repo_id = "macpaw-research/qwen3-dual-dit-aligner-ref/10608/",
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
