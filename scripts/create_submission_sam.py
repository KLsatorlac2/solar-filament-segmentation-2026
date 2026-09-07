import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch

from segment_anything import SamPredictor, sam_model_registry
from pycocotools import mask as mask_utils


def generate_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (0, 0), 15)

    enhanced = cv2.divide(
        gray,
        blur,
        scale=128,
    )

    threshold = np.percentile(
        enhanced,
        25,
    )

    binary = (
        enhanced < threshold
    ).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    points = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 100:
            continue

        x, y = centroids[i]

        points.append(
            [x, y]
        )

    return np.asarray(
        points,
        dtype=np.float32,
    )


def encode_rle(mask):
    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    rle = mask_utils.encode(
        np.asfortranarray(mask)
    )

    counts = rle["counts"]

    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")

    return counts


def get_filament_masks(
    predictor,
    image,
    min_area=100,
):
    points = generate_points(image)

    if len(points) == 0:
        return []

    masks = []

    for point in points:

        point_coords = point.reshape(
            1,
            2,
        )

        point_labels = np.array(
            [1],
            dtype=np.int32,
        )

        mask, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

        mask = mask[0].astype(
            np.uint8
        )

        masks.append(mask)

    if not masks:
        return []

    combined = np.zeros_like(
        masks[0],
        dtype=np.uint8,
    )

    for mask in masks:
        combined = np.maximum(
            combined,
            mask,
        )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            combined,
            connectivity=8,
        )
    )

    filament_masks = []

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA,
        ]

        if area < min_area:
            continue

        filament = (
            labels == i
        ).astype(np.uint8)

        filament_masks.append(
            filament
        )

    return filament_masks


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dir",
        required=True,
    )

    parser.add_argument(
        "--sam_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--decoder_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="outputs/submission.csv",
    )

    parser.add_argument(
        "--min_area",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    sam = sam_model_registry["vit_b"](
        checkpoint=args.sam_checkpoint
    )

    decoder_state = torch.load(
        args.decoder_checkpoint,
        map_location=device,
        weights_only=True,
    )

    sam.mask_decoder.load_state_dict(
        decoder_state
    )

    sam.to(device)

    sam.eval()

    predictor = SamPredictor(sam)

    image_files = sorted(
        [
            f
            for f in os.listdir(args.test_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]
    )

    print(
        f"Test images: {len(image_files)}"
    )

    rows = []

    for index, filename in enumerate(
        image_files,
        start=1,
    ):

        path = os.path.join(
            args.test_dir,
            filename,
        )

        image = cv2.imread(path)

        if image is None:
            print(
                f"[{index}] Failed: {filename}"
            )
            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        predictor.set_image(
            image_rgb
        )

        filament_masks = get_filament_masks(
            predictor,
            image,
            min_area=args.min_area,
        )

        image_id = os.path.splitext(
            filename
        )[0]

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_id}: "
            f"{len(filament_masks)} filaments"
        )

        for filament_index, mask in enumerate(
            filament_masks,
            start=1,
        ):

            rle = encode_rle(mask)

            rows.append(
                {
                    "filament_id":
                        f"{image_id}_{filament_index}",
                    "segmentation_rle":
                        rle,
                }
            )

    submission = pd.DataFrame(
        rows,
        columns=[
            "filament_id",
            "segmentation_rle",
        ],
    )

    os.makedirs(
        os.path.dirname(
            args.output
        ) or ".",
        exist_ok=True,
    )

    submission.to_csv(
        args.output,
        index=False,
    )

    print()
    print("=" * 60)
    print("SUBMISSION CREATED")
    print("=" * 60)

    print(
        f"Rows: {len(submission)}"
    )

    print(
        f"Output: {args.output}"
    )

    print()

    if len(submission) > 0:
        print(
            submission.head()
        )


if __name__ == "__main__":
    main()