from typing import Literal

import torch
from torch import nn

from typing import Optional
from transformers import AutoModel, AutoProcessor


class MLPProjector(nn.Module):
    def __init__(self, dims: list[int], bias: bool = False):
        """
        Args:
            dims: A list defining the layer sizes. 
                  e.g., [3, 32, hidden_size, hidden_size]
        """
        super(MLPProjector, self).__init__()

        if len(dims) < 2:
            raise ValueError(f"dims must contain at least input and output dimensions, got {dims}")

        self.dims = dims
        self.bias = bias
        self.layers = self._build_layers()

    def _build_layers(self) -> nn.Sequential:
        layers = []
        
        for i in range(len(self.dims) - 1):
            layers.append(nn.Linear(self.dims[i], self.dims[i+1], bias=self.bias))
            
            # Add GELU after every layer except the final output layer
            if i < len(self.dims) - 2:
                layers.append(nn.GELU(approximate="tanh"))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
        

class QwenWithExpertModel(nn.Module):
    def __init__(
        self,
        model_path,
        precision
    ):
        super().__init__()
        self.mibot = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.config = self.mibot.config
        self.qwen_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            # "model.visual.patch_embed.proj.weight",
            # "model.visual.patch_embed.proj.bias",
            # "model.visual.pos_embed.weight",

            # "input_layernorm",
            # "post_attention_layernorm",
            # "model.norm",

            # hack
            "state_projector",
            "action_projector",
            "action_output_layer",
            "t_embedder",
            "t_projector",
            "sink",
            "rotary_emb"
        ]
        
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def get_input_embeddings(self, tokens: torch.Tensor):
        return self.mibot.vlm.get_input_embeddings()(tokens)
 

    def preprocess_images(self, image: torch.FloatTensor):
        # Images of shape (K, 3, H, W)
        image_inputs = self.qwen_processor.image_processor(image, return_tensors='pt', do_rescale=False)
        
        pixel_values  = image_inputs["pixel_values"].to(image.device)
        image_grid_thw = image_inputs["image_grid_thw"].to(image.device)

        return pixel_values, image_grid_thw


    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.mibot.vlm.get_image_features(pixel_values, image_grid_thw)


    def forward(self, dit_module, vlm_outputs, suffix_emb, suffix_att_mask, time_emb):
        B, dit_query_length = suffix_emb.shape[:2]
        position_ids = (
            torch.arange(0, dit_query_length, device=suffix_emb.device).view(1, 1, -1).repeat(3, B, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )
        position_embeds = self.mibot.rotary_emb(suffix_emb, position_ids)
        
        dit_mask = torch.tril(torch.ones((B, dit_query_length, dit_query_length), 
                                         device=suffix_emb.device, dtype=suffix_att_mask.dtype), diagonal=0)
        pad_mask = suffix_att_mask.unsqueeze(1) & suffix_att_mask.unsqueeze(2)
        dit_mask = dit_mask & pad_mask 
        
        cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, dit_query_length, -1)
        attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None]
        attn_mask = attn_mask.bool()

        # Use the passed dit_module instead of self.mibot.dit
        suffix_emb = dit_module(suffix_emb, 
                                vlm_outputs.past_key_values, 
                                attn_mask, 
                                position_embeds, 
                                time_emb)
        
        return suffix_emb