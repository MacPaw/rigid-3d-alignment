"""Model configuration: the shapes the model expects, and how to build it from a checkpoint."""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import safetensors.torch

from asset_vla.model import observation as _observation
from asset_vla.model import vla
from asset_vla.shared import array_typing as at

logger = logging.getLogger("asset_vla")


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Configuration for the asset-alignment VLA."""

    # Weight precision used by the model.
    dtype: str = "bfloat16"
    # Action space dimension.
    action_dim: int = 32
    # Action sequence length.
    action_horizon: int = 1
    # Tokenized prompt maximum length. The prompt carries one vision block per input view,
    # so this must grow when reference views are enabled (270 covers 3 cameras + 6 renders).
    max_token_len: int = 270

    def load_pytorch(self, train_config, weight_path: str) -> vla.AssetVLA:
        """Instantiate the model and load weights from a safetensors checkpoint."""
        logger.info(f"train_config: {train_config}")
        model = vla.AssetVLA(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_observation.Observation, _observation.Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_observation.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _observation.Observation(
                images={key: image_spec for key in _observation.IMAGE_KEYS},
                image_masks={key: image_mask_spec for key in _observation.IMAGE_KEYS},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def fake_obs(self, batch_size: int = 1) -> _observation.Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> _observation.Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)
