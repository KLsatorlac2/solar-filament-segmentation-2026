import torch
import torch.nn as nn
from segment_anything import sam_model_registry


class SAM(nn.Module):
    def __init__(self, checkpoint, model_type="vit_b", freeze_encoder=True):
        super().__init__()

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)

        if freeze_encoder:
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = False

            for p in self.sam.prompt_encoder.parameters():
                p.requires_grad = False

        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

        self.sam.image_encoder.eval()
        self.sam.prompt_encoder.eval()
        self.sam.mask_decoder.train()

    def forward(self, image, point_coords, point_labels):
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(
                self.sam.preprocess(image)
            )

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = self.sam.postprocess_masks(
            low_res_masks,
            image.shape[-2:],
            image.shape[-2:],
        )

        return masks, iou_predictions