import torch
import transformers.models.llama.modeling_llama
from torch import nn
from transformers import  LlamaConfig
from .modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

class LlamaRotaryEmbedding_modify(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @dynamic_rope_update
    def forward(self, x, position_ids):
        # print(position_ids)
        inv_freq_expanded = self.inv_freq[None, :, None].to(dtype=x.dtype).expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].to(dtype=x.dtype).to(x.device)  # 这里允许梯度传递

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def modify_llama():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbedding_modify