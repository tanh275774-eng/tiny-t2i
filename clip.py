"""冻结的预训练 CLIP 文本编码器：返回每个 token 的特征，而非一个 pooled 向量。"""

import torch
from torch import nn

from model_cache import resolve_model

DEFAULT_CLIP = "openai/clip-vit-base-patch32"


class FrozenCLIP(nn.Module):
    def __init__(self, model_id=DEFAULT_CLIP, revision=None):
        super().__init__()
        model_path = resolve_model(model_id, revision)
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = CLIPTextModel.from_pretrained(model_path, local_files_only=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self.max_length = self.model.config.max_position_embeddings

    @torch.no_grad()
    def forward(self, captions):
        device = next(self.model.parameters()).device
        tokens = self.tokenizer(captions, padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt").to(device)
        self.model.eval()
        features = self.model(**tokens).last_hidden_state
        return features, tokens.attention_mask.bool()
