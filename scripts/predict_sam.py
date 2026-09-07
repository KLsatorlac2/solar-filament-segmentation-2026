import argparse
import os

import cv2
import numpy as np
import torch

from segment_anything import SamPredictor, sam_model_registry


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

    binary = (enhanced < threshold).astype(
        np.uint8
    ) * 255

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

    return np.array(
        points,
        dtype=np.float32,
    )


def remove_small_components(mask, min_area=100):
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
    )

    result = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            result[labels == i] = 1

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
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
        default="outputs/sam_mask.png",
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(
            args.image
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

    predictor = SamPredictor(sam)

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    predictor.set_image(image_rgb)

    points = generate_points(image)

    if len(points) == 0:
        print("No candidate points found.")

        empty = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        cv2.imwrite(
            args.output,
            empty,
        )

        return

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

        masks.append(
            mask[0].astype(np.uint8)
        )

    final_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    for mask in masks:
        final_mask = np.maximum(
            final_mask,
            mask,
        )

    final_mask = remove_small_components(
        final_mask,
        min_area=100,
    )

    final_mask = (
        final_mask * 255
    ).astype(np.uint8)

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True,
    )

    cv2.imwrite(
        args.output,
        final_mask,
    )

    print(
        f"Points: {len(points)}"
    )

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()