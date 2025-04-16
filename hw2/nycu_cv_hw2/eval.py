import logging

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from nycu_cv_hw2.config import settings
from nycu_cv_hw2.constants import IOU_THRESHOLD, MODEL_DIR_PATH, NUM_CLASSES
from nycu_cv_hw2.data import get_data_loader
from nycu_cv_hw2.utils import compute_iou_matrix
from torch.amp import autocast
from torchmetrics import ConfusionMatrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="basic.log",
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
    data_loader = get_data_loader(settings.batch_size, "val")

    # Inference
    model.eval()

    cm = ConfusionMatrix(
        task="multiclass",
        num_classes=NUM_CLASSES + 1,
    )

    for inputs, targets in tqdm.tqdm(
        data_loader,
        desc=("Evaluation"),
        ncols=100,
    ):
        inputs = [input.to(device) for input in inputs]  # input = image
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(device_type=device.type):
            outputs = model(inputs)

        outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

        # 計算/更新 confusion matrix
        for target, output in zip(targets, outputs):
            target_boxes: torch.Tensor = target["boxes"]
            target_labels: torch.Tensor = target["labels"]

            score_filter: torch.BoolTensor = (
                output["scores"] > settings.score_threshold
            )
            output_boxes: torch.Tensor = output["boxes"][score_filter]
            output_labels: torch.Tensor = output["labels"][score_filter]

            if len(target_boxes) == 0 and len(output_boxes) == 0:
                continue

            if len(target_boxes) == 0:
                # 全部都是 FP, background class = 0
                fake_target_labels = torch.full_like(output_labels, 0)
                cm.update(output_labels, fake_target_labels)
                continue

            if len(output_boxes) == 0:
                # 所有都是 FN, background class = 0
                fake_output_labels = torch.full_like(target_labels, 0)
                cm.update(fake_output_labels, target_labels)
                continue

            iou_matrix = compute_iou_matrix(target_boxes, output_boxes)

            matched_target_indices = set()
            matched_output_indices = set()
            matched_pairs = []

            while True:
                # 找到最大的 IoU
                max_iou = iou_matrix.max()
                if max_iou < IOU_THRESHOLD:  # 全部都 < 0.5 就結束迴圈
                    break

                # 找到最大的 IoU 的 index
                target_idx, output_idx = torch.unravel_index(
                    torch.argmax(iou_matrix), iou_matrix.shape
                )

                matched_target_indices.add(target_idx.item())
                matched_output_indices.add(output_idx.item())
                matched_pairs.append((target_idx.item(), output_idx.item()))

                # 避免重複 matching
                iou_matrix[target_idx, :] = 0
                iou_matrix[:, output_idx] = 0

            final_target_labels = []
            final_output_labels = []

            for t_idx, p_idx in matched_pairs:
                final_target_labels.append(target_labels[t_idx].item())
                final_output_labels.append(output_labels[p_idx].item())

            # FN
            for idx in range(len(target_labels)):
                if idx not in matched_target_indices:
                    final_target_labels.append(target_labels[idx].item())
                    final_output_labels.append(0)

            # FP
            for idx in range(len(output_labels)):
                if idx not in matched_output_indices:
                    final_target_labels.append(0)
                    final_output_labels.append(output_labels[idx].item())
            cm.update(
                torch.tensor(final_output_labels, dtype=torch.long),
                torch.tensor(final_target_labels, dtype=torch.long),
            )

    cm = cm.compute().numpy()
    # print(cm.numpy().shape)
    print(np.sum(np.diagonal(cm)))
    print(np.sum(cm[:, 0]))
    print(np.sum(cm[0, :]))
    df_cm = pd.DataFrame(
        cm,
        index=range(NUM_CLASSES + 1),
        columns=range(NUM_CLASSES + 1),
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    sns.heatmap(df_cm, ax=ax, annot=True, cmap="Spectral", fmt="g")
    fig.savefig(f"{model_name}_{settings.score_threshold}.png")


if __name__ == "__main__":
    main()
