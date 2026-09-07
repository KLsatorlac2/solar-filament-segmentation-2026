import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SAMDataset(Dataset):
    def __init__(self, base_dataset, image_size=1024):
        self.base_dataset = base_dataset
        self.image_size = image_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]

        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)

        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)

        image = image * 255.0

        mask_np = mask.squeeze(0).numpy()
        ys, xs = np.where(mask_np > 0)

        if len(xs) > 0:
            i = np.random.randint(len(xs))
            point = np.array(
                [[xs[i], ys[i]]],
                dtype=np.float32,
            )
            label = np.array([1], dtype=np.int64)
        else:
            point = np.array(
                [[self.image_size / 2, self.image_size / 2]],
                dtype=np.float32,
            )
            label = np.array([0], dtype=np.int64)

        return {
            "image": image,
            "mask": mask,
            "point_coords": torch.from_numpy(point),
            "point_labels": torch.from_numpy(label),
        }