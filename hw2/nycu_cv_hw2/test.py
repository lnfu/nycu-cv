"""Task1.

Detect each digit in the image. The submission file should be a JSON file
in COCO format named pred.json. Specifically, it should be a list of labels,
where each label is represented as a dictionary with the keys:
image_id, bbox, score, and category_id.
"""

import csv
import json
import logging
import time

import click
import torch
import tqdm
from nycu_cv_hw2.config import settings
from nycu_cv_hw2.constants import MODEL_DIR_PATH
from nycu_cv_hw2.data import get_data_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="basic.log",  # TODO 改成加到 tensorboard?
)


@click.command()
@click.option(
    "--model-name",
    prompt="Model name",
    help="Name of the model you want to use, e.g., '2025-04-03_13-02-32'.",
)
def main(model_name: str):
    # Device
    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    # Model
    model: torch.nn.Module = torch.load(
        MODEL_DIR_PATH / f"{model_name}.pt",
        weights_only=False,
        map_location=device,
    )

    # Dataloader
    data_loader = get_data_loader(settings.batch_size, "test")

    # Inference
    task1_predictions = []
    task2_predictions = []
    model.eval()
    # TODO autocast
    for inputs, image_ids in tqdm.tqdm(
        data_loader,
        ncols=100,
    ):
        inputs = [input.to(device, non_blocking=True) for input in inputs]
        with torch.no_grad():
            outputs = model(inputs)
            for image_id, output in zip(image_ids, outputs):
                # 單張圖片

                valid_predictions = list(
                    filter(
                        lambda x: x["score"] > settings.score_threshold,
                        [
                            {
                                "box": box.cpu().numpy(),
                                "label": int(label),  # 0 - 9
                                "score": float(score),
                            }
                            for box, label, score in zip(
                                output["boxes"],
                                output["labels"],
                                output["scores"],
                            )
                        ],
                    )
                )

                valid_predictions.sort(key=lambda x: x["box"][0])

                task1_predictions.extend(
                    [
                        {
                            "image_id": int(image_id),
                            "bbox": [
                                float(pred["box"][0]),
                                float(pred["box"][1]),
                                float(pred["box"][2] - pred["box"][0]),
                                float(pred["box"][3] - pred["box"][1]),
                            ],
                            "score": pred["score"],
                            "category_id": pred["label"] + 1,  # 1 - 10
                        }
                        for pred in valid_predictions
                    ]
                )

                if valid_predictions:
                    task2_predictions.append(
                        [
                            int(image_id),
                            int(
                                "".join(
                                    [
                                        str(pred["label"])
                                        for pred in valid_predictions
                                    ]
                                )
                            ),
                        ]
                    )
                else:
                    task2_predictions.append([int(image_id), -1])

    with open("pred.json", "w") as f:
        json.dump(task1_predictions, f, indent=4)

    with open("pred.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "pred_label"])
        writer.writerows(task2_predictions)


if __name__ == "__main__":
    main()
