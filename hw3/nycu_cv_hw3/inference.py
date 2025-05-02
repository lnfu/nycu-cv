import json
import logging
import zipfile

import click
import numpy as np
import torch
import tqdm
from nycu_cv_hw3.constants import (
    MASK_THRESHOLD,
    MODEL_DIR_PATH,
    OUTPUT_DIR_PATH,
)
from nycu_cv_hw3.data import inference_loader
from nycu_cv_hw3.models import Model
from pycocotools import mask as mask_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="inference.log",
)
logging.info("==========================================")


@click.command()
@click.option(
    "--ckpt-name" "-c",
    prompt="Checkpoint name",
    help="Name of the checkpoint you want to use, e.g., '2025-04-03_13-02-32'",
)
def main(ckpt_name: str):

    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"{device=}")

    # TODO epoch?
    model = Model.from_pretrained(MODEL_DIR_PATH / f"{ckpt_name}.pth", device)
    model = model.to(device)
    model.eval()

    results = []
    for (
        images,
        image_names,
        image_ids,
        image_widths,
        image_heights,
    ) in tqdm.tqdm(inference_loader, desc=("inference"), ncols=100):
        images = list(image.to(device) for image in images)
        outputs = model(images)

        for output, image_id, w, h in zip(
            outputs, image_ids, image_widths, image_heights
        ):
            boxes = output["boxes"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            masks = output["masks"].detach().cpu().numpy()

            for box, label, score, mask in zip(boxes, labels, scores, masks):
                x1, y1, x2, y2 = box
                w_box, h_box = x2 - x1, y2 - y1

                mask = mask.squeeze() > MASK_THRESHOLD
                arr = np.asfortranarray(mask.astype(np.uint8))
                rle = mask_utils.encode(arr)
                rle["counts"] = rle["counts"].decode("utf-8")

                results.append(
                    {
                        "image_id": int(image_id),
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(w_box),
                            float(h_box),
                        ],
                        "score": float(score),
                        "category_id": int(label),
                        "segmentation": {
                            "size": [int(h), int(w)],
                            "counts": rle["counts"],
                        },
                    }
                )

    json_path = OUTPUT_DIR_PATH / "test-results.json"
    with open(json_path, "w") as f:
        json.dump(results, f)

    with zipfile.ZipFile(
        OUTPUT_DIR_PATH / f"{ckpt_name}.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zipf:
        zipf.write(json_path, arcname=json_path.name)
    print(json_path.name)


if __name__ == "__main__":
    main()
