import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()

        model_name = "nvidia/mit-b0"

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=out_channels,
                ignore_mismatched_sizes=True,
            )
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(
                model_name,
                num_labels=out_channels,
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        input_size = x.shape[-2:]

        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )