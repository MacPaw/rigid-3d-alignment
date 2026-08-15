import logging

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
import copy

from asset_vla.model.qwen import QwenWithExpertModel, MLPProjector
import asset_vla.model.preprocessing as _preprocessing

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


class DiTBranch(nn.Module):
    """Encapsulates all modality-specific parameters for a single independent DiT."""
    def __init__(
        self, 
        dit, 
        action_in, 
        action_out, 
        sink,
        state_proj,
        extrinsics_proj,
        t_embedder,
        t_projector
    ):
        super().__init__()
        self.dit = dit
        self.action_in_proj = action_in
        self.action_out_proj = action_out
        self.sink = sink
        self.state_proj = state_proj
        self.extrinsics_proj = extrinsics_proj
        self.t_embedder = t_embedder
        self.t_projector = t_projector

    def gradient_checkpointing_enable(self):
        self.dit.gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        self.dit.gradient_checkpointing = False


class AssetVLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model_path = "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"
        self.qwen_with_expert = QwenWithExpertModel(
            model_path,
            precision=config.dtype,
        )
        expert_config = self.qwen_with_expert.config
        hidden_size = expert_config.dit_config.hidden_size


        # --- SHARED COMPONENTS ---
        self.rotary_emb = self.qwen_with_expert.mibot.rotary_emb

        # --- BRANCH 1: TRANSLATION ---
        trans_action_in = MLPProjector(dims=[3, 32, hidden_size, hidden_size])
        trans_action_out = MLPProjector(dims=[hidden_size, hidden_size, 32, 3])
        trans_state_proj = self.qwen_with_expert.mibot.state_projector
        trans_ext_proj = MLPProjector(dims=[3, hidden_size, hidden_size])
        
        trans_t_emb = self.qwen_with_expert.mibot.t_embedder
        trans_t_emb.dtype = torch.float32
        trans_t_proj = self.qwen_with_expert.mibot.t_projector
        
        self.trans_branch = DiTBranch(
            dit=self.qwen_with_expert.mibot.dit,
            action_in=trans_action_in,
            action_out=trans_action_out,
            sink=self.qwen_with_expert.mibot.sink,
            state_proj=trans_state_proj,
            extrinsics_proj=trans_ext_proj,
            t_embedder=trans_t_emb,
            t_projector=trans_t_proj
        )

        # --- BRANCH 2: ROTATION ---
        rot_action_in = MLPProjector(dims=[6, 32, hidden_size, hidden_size])
        rot_action_out = MLPProjector(dims=[hidden_size, hidden_size, 32, 6])
        rot_state_proj = copy.deepcopy(self.qwen_with_expert.mibot.state_projector)
        rot_ext_proj = MLPProjector(dims=[3, hidden_size, hidden_size])
        
        rot_t_emb = copy.deepcopy(self.qwen_with_expert.mibot.t_embedder)
        rot_t_emb.dtype = torch.float32
        rot_t_proj = copy.deepcopy(self.qwen_with_expert.mibot.t_projector)
        
        self.rot_branch = DiTBranch(
            dit=copy.deepcopy(self.qwen_with_expert.mibot.dit),
            action_in=rot_action_in,
            action_out=rot_action_out,
            sink=copy.deepcopy(self.qwen_with_expert.mibot.sink),
            state_proj=rot_state_proj,
            extrinsics_proj=rot_ext_proj,
            t_embedder=rot_t_emb,
            t_projector=rot_t_proj
        )
        
        # --- Delete unused original projectors ---
        del self.qwen_with_expert.mibot.action_projector
        del self.qwen_with_expert.mibot.action_output_layer

        torch.set_float32_matmul_precision("high")
        self.gradient_checkpointing_enabled = False
    

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.qwen_with_expert.mibot.vlm.model.language_model.gradient_checkpointing = True
        self.qwen_with_expert.mibot.vlm.model.visual.gradient_checkpointing = True
        self.trans_branch.gradient_checkpointing_enable()
        self.rot_branch.gradient_checkpointing_enable()

        logging.info("Enabled gradient checkpointing for AssetVLA model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.qwen_with_expert.mibot.vlm.model.language_model.gradient_checkpointing = False
        self.qwen_with_expert.mibot.vlm.model.visual.gradient_checkpointing = False
        self.trans_branch.gradient_checkpointing_disable()
        self.rot_branch.gradient_checkpointing_disable()

        logging.info("Disabled gradient checkpointing for AssetVLA model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)


    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            observation.camera_extrinsics
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )


    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)


    def split_language_tokens(self, lang_tokens, lang_masks):
        image_token = self.qwen_with_expert.qwen_processor.image_token
        image_token_id = self.qwen_with_expert.qwen_processor.tokenizer.convert_tokens_to_ids(image_token)

        image_positions = (lang_tokens[0] == image_token_id).nonzero(as_tuple=True)[0].cpu()

        lang_segments = torch.tensor_split(lang_tokens, image_positions, dim=1)
        mask_segments = torch.tensor_split(lang_masks, image_positions, dim=1)

        lang_segments = [
            lang_segments[0],
            *[seg[:, 1:] for seg in lang_segments[1:]]
        ]

        mask_segments = [
            mask_segments[0],
            *[seg[:, 1:] for seg in mask_segments[1:]]
        ]

        return lang_segments, mask_segments, image_token_id
    
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess images and language tokens to Qwen3VLForConditionalGeneration"""
        # Process images
        pixel_values_arr = []
        images_grids = []

        images = torch.stack(images, dim=0) # (K, B, C, H, W)
        images = torch.permute(images, (1, 0, 2, 3, 4))
        B, K = images.shape[:2]

        for b in range(B):
            imgs = images[b]
            pixel_values, image_grid_thw = self.qwen_with_expert.preprocess_images(imgs)
            pixel_values_arr.append(pixel_values)
            images_grids.append(image_grid_thw)

        # Process language tokens
        lang_segments, lang_masks, image_token_id = self.split_language_tokens(lang_tokens, lang_masks)
        
        # Fuse language and image embeddings in appropriate order
        input_ids = []
        attention_mask = []
        
        images_grids = torch.stack(images_grids, dim=0) # (B, K, W, H)
        merge_length = self.qwen_with_expert.qwen_processor.image_processor.merge_size**2
        # assume that per batch we have the same number of image tokens
        img_tokens_nb = images_grids[:, 0].prod(dim=-1)[0].item() // merge_length
        assert torch.all(images_grids[:] == images_grids[0])

        # assumption about prompt structure
        assert len(lang_segments) == K + 1
        
        for i in range(K):
            attention_mask.append(lang_masks[i])
            input_ids.append(lang_segments[i])

            image_tokens = torch.full(
                (B, img_tokens_nb),
                image_token_id,
                dtype=lang_tokens.dtype,
                device=lang_tokens.device
            )
            input_ids.append(image_tokens)

            image_mask = img_masks[i][:, None].expand(B, img_tokens_nb)
            attention_mask.append(image_mask)

        # last language segment
        input_ids.append(lang_segments[-1])
        attention_mask.append(lang_masks[-1])
        
        input_ids = torch.cat(input_ids, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1).to(torch.long)
        images_grids = images_grids.view(-1, 3) # (B*K, W, H)

        pixel_values = torch.stack(pixel_values_arr, dim=0) # B, T, H
        B, T, H = pixel_values.shape
        flattened_pixel_values = pixel_values.view(B * T, H)

        vlm_inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "pixel_values": flattened_pixel_values,
                      "image_grid_thw": images_grids}

        return self.qwen_with_expert.mibot.vlm(**vlm_inputs, use_cache=True)


    def embed_suffix(self, state, noisy_actions, timestep, camera_extrinsics, branch: DiTBranch):
        """Embed state, noisy_actions, timestep to prepare for DiT."""
        B = state.shape[0]
        device = state.device

        # ---- Camera translation embedding ----
        camera_translations = camera_extrinsics[:, :, :3, 3] 

        def extrinsics_proj_func(camera_translations):
            return branch.extrinsics_proj(camera_translations)
        
        cam_tokens = self._apply_checkpoint(extrinsics_proj_func, camera_translations) 

        # ---- State embedding ----
        if branch.state_proj.layers[0].weight.dtype == torch.float32:
            state = state.to(torch.float32)
            camera_extrinsics = camera_extrinsics.to(torch.float32)

        def state_proj_func(state):
            return branch.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state) 
        state_emb = state_emb[:, None, :]

        # ---- Timestep embedding ----
        def time_proj_func(timestep):
            return branch.t_embedder(timestep)
        
        time_emb = self._apply_checkpoint(time_proj_func, timestep * 1000)
        time_emb = branch.t_projector(time_emb).view(B, 6, -1)
        time_emb = time_emb.type(dtype=timestep.dtype)

        # ---- Action embedding ----
        def action_proj_func(noisy_actions):
            return branch.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # ---- Sink token ----
        sink = branch.sink.weight[None].repeat(B, 1, 1)

        # ---- Concatenate tokens ----
        embs = torch.cat([sink, state_emb, cam_tokens, action_emb], dim=1).contiguous()
        seq_len = embs.shape[1]

        # Padding mask (all tokens are valid)
        attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
        return embs, attention_mask, time_emb


    def forward(self, observation, actions, noise_translations=None, noise_rotations=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, camera_extrinsics = self._preprocess_observation(observation, train=True)

        translations, rotations = actions[..., :3], actions[..., 3:]
        
        if noise_translations is None:
            noise_translations = self.sample_noise(translations.shape, actions.device)

        if noise_rotations is None:
            noise_rotations = self.sample_noise(rotations.shape, actions.device) * 0.577

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]

        # ------- TRANSLATIONS --------
        x_t_trans = (time_expanded * translations + (1 - time_expanded) * noise_translations)
        u_t_trans = (translations - noise_translations) 
        
        # ------- ROTATIONS --------
        x_t_rot = (time_expanded * rotations + (1 - time_expanded) * noise_rotations) 
        u_t_rot = (rotations - noise_rotations)

        prefix_emb = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        def detach_recursive(val):
          if isinstance(val, torch.Tensor):
              return val.detach()
          elif isinstance(val, tuple):
              return tuple(detach_recursive(v) for v in val)
          elif isinstance(val, list):
              return list(detach_recursive(v) for v in val)
          elif isinstance(val, dict):
              return {k: detach_recursive(v) for k, v in val.items()}
          return val

        # Detach rotation branch for proper translation training
        prefix_emb_trans = prefix_emb

        prefix_emb_rot = prefix_emb.__class__(
            **{k: detach_recursive(v) for k, v in prefix_emb.items()}
        )

        suffix_emb_trans, suffix_att_mask, time_emb_trans = self.embed_suffix(
            state, x_t_trans, time, camera_extrinsics, self.trans_branch
        )
        suffix_emb_rot, _, time_emb_rot = self.embed_suffix(
            state, x_t_rot, time, camera_extrinsics, self.rot_branch
        )

        if (
            self.qwen_with_expert.mibot.vlm.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_emb_trans = suffix_emb_trans.to(dtype=torch.bfloat16)
            time_emb_trans = time_emb_trans.to(dtype=torch.bfloat16)
            suffix_emb_rot = suffix_emb_rot.to(dtype=torch.bfloat16)
            time_emb_rot = time_emb_rot.to(dtype=torch.bfloat16)
        
        # Run the two separate DiTs
        def forward_trans_func(prefix_emb, suffix_emb, suffix_att_mask, time_emb):
            return self.qwen_with_expert(self.trans_branch.dit, 
                                          prefix_emb, suffix_emb, 
                                          suffix_att_mask, time_emb)
            
        def forward_rot_func(prefix_emb, suffix_emb, suffix_att_mask, time_emb):
            return self.qwen_with_expert(self.rot_branch.dit, 
                                          prefix_emb, suffix_emb, 
                                          suffix_att_mask, time_emb)
        
        suffix_out_trans = self._apply_checkpoint(forward_trans_func, prefix_emb_trans, suffix_emb_trans, suffix_att_mask, time_emb_trans)
        suffix_out_rot = self._apply_checkpoint(forward_rot_func, prefix_emb_rot, suffix_emb_rot, suffix_att_mask, time_emb_rot)

        # Output projections using branch specific layers
        suffix_out_trans = suffix_out_trans[:, -self.config.action_horizon :].to(dtype=torch.float32)
        suffix_out_rot = suffix_out_rot[:, -self.config.action_horizon :].to(dtype=torch.float32)

        def trans_out_proj_func(out): return self.trans_branch.action_out_proj(out)
        def rot_out_proj_func(out): return self.rot_branch.action_out_proj(out)
        
        v_t_trans = self._apply_checkpoint(trans_out_proj_func, suffix_out_trans)
        v_t_rot = self._apply_checkpoint(rot_out_proj_func, suffix_out_rot)

        rot_mse = F.mse_loss(u_t_rot, v_t_rot, reduction="none")
        trans_mse = F.mse_loss(u_t_trans, v_t_trans, reduction="none")

        return {
            "rot_mse": rot_mse,
            "trans_mse": trans_mse
        }


    @torch.no_grad()
    def sample_actions(self, device, observation, noise_translations=None, noise_rotations=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]

        if noise_translations is None:
            noise_translations = self.sample_noise((bsize, self.config.action_horizon, 3), device)

        if noise_rotations is None:
            noise_rotations = self.sample_noise((bsize, self.config.action_horizon, 6), device) * 0.577

        images, img_masks, lang_tokens, lang_masks, state, camera_extrinsics = self._preprocess_observation(observation, train=False)
        prefix_emb = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        dt = 1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        def denoise_step(x_trans_noisy, x_rot_noisy, t):
            return self.denoise_step(
                state = state,
                prefix_emb = prefix_emb,
                x_t_trans=x_trans_noisy,
                x_t_rot=x_rot_noisy,
                time = t,
                ext = camera_extrinsics
            )
        
        x_trans = noise_translations
        x_rot = noise_rotations 
        
        for step in range(num_steps):
            t = torch.ones((bsize, 1, 1), device=device, dtype=torch.float32) * step / num_steps
            v_trans, v_rot = denoise_step(x_trans, x_rot, t)
            x_trans = x_trans + v_trans * dt
            x_rot = x_rot + v_rot * dt
        
        return torch.cat([x_trans, x_rot], dim=-1)


    def denoise_step(
        self,
        state,
        prefix_emb,
        x_t_trans,
        x_t_rot,
        time,
        ext
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""

        suffix_emb_trans, suffix_att_mask, time_emb_trans = self.embed_suffix(
            state, x_t_trans, time, ext, self.trans_branch
        )

        suffix_emb_rot, _, time_emb_rot = self.embed_suffix(
            state, x_t_rot, time, ext, self.rot_branch
        )

        if self.qwen_with_expert.mibot.vlm.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_emb_trans = suffix_emb_trans.to(dtype=torch.bfloat16)
            suffix_emb_rot = suffix_emb_rot.to(dtype=torch.bfloat16)
            time_emb_trans = time_emb_trans.to(dtype=torch.bfloat16)
            time_emb_rot = time_emb_rot.to(dtype=torch.bfloat16)
        
        suffix_out_trans = self.qwen_with_expert(self.trans_branch.dit, prefix_emb, suffix_emb_trans, suffix_att_mask, time_emb_trans)
        suffix_out_rot = self.qwen_with_expert(self.rot_branch.dit, prefix_emb, suffix_emb_rot, suffix_att_mask, time_emb_rot)

        suffix_out_trans = suffix_out_trans[:, -self.config.action_horizon :].to(dtype=torch.float32)
        suffix_out_rot = suffix_out_rot[:, -self.config.action_horizon :].to(dtype=torch.float32)

        v_t_trans = self.trans_branch.action_out_proj(suffix_out_trans)
        v_t_rot = self.rot_branch.action_out_proj(suffix_out_rot)

        return v_t_trans, v_t_rot
