import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from nycu_cv_hw2.constants import DATA_DIR_PATH, IOU_THRESHOLD, NUM_CLASSES
from nycu_cv_hw2.utils import compute_iou_matrix, eprint
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix


class Trainer:

    _scaler = GradScaler()
    min_train_loss = float("inf")  # TODO not used
    min_val_loss = float("inf")  # TODO not used

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        writer: SummaryWriter,
        score_threshold: float,
    ):
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._writer = writer
        self._score_threshold = score_threshold

    def train_batch(self, inputs: list, targets: list) -> float:
        self._model.train()

        self._optimizer.zero_grad()
        with autocast(device_type=self._device.type):
            loss_dict = self._model(inputs, targets)
            # loss: torch.Tensor = sum(loss for loss in loss_dict.values())
            loss = (
                1.0 * loss_dict["loss_classifier"]
                + 1.0 * loss_dict["loss_box_reg"]
                + 0.5 * loss_dict["loss_objectness"]
                + 0.5 * loss_dict["loss_rpn_box_reg"]
            )

        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()
        return loss.item()

    def val_batch_loss(self, inputs: list, targets: list) -> float:
        self._model.train()
        with autocast(device_type=self._device.type):
            loss_dict = self._model(inputs, targets)
            loss = self._compute_loss(loss_dict)
        return loss.item()

    def val_batch_outputs(self, inputs: list) -> torch.Tensor:
        self._model.eval()
        with autocast(device_type=self._device.type):
            outputs = self._model(inputs)
        return outputs

    @torch.enable_grad()
    def train_epoch(
        self,
        epoch: int,
        data_loader: torch.utils.data.DataLoader,
    ) -> float:
        total_loss = 0.0
        for inputs, targets in tqdm.tqdm(
            data_loader,
            desc=(f"Training {epoch}"),
            ncols=100,
        ):

            inputs = self._move_to_device(
                inputs, self._device, non_blocking=True
            )
            targets = self._move_to_device(
                targets, self._device, non_blocking=True
            )

            loss = self.train_batch(inputs, targets)
            total_loss += loss

        return total_loss / len(data_loader)

    @torch.no_grad()
    def val_epoch(
        self,
        epoch: int,
        data_loader: torch.utils.data.DataLoader,
    ) -> float:

        total_loss = 0.0
        cm = ConfusionMatrix(
            task="multiclass",
            num_classes=NUM_CLASSES + 1,
        )

        total_targets_coco = COCO(DATA_DIR_PATH / "valid.json")  # TODO
        total_outputs = []
        # total_outputs = total_targets.loadRes(total_outputs)

        for inputs, targets in tqdm.tqdm(
            data_loader,
            desc=(f"Training {epoch}"),
            ncols=100,
        ):

            inputs_gpu = self._move_to_device(
                inputs, self._device, non_blocking=True
            )

            targets_gpu = self._move_to_device(
                targets, self._device, non_blocking=True
            )

            loss = self.val_batch_loss(inputs_gpu, targets_gpu)
            total_loss += loss

            outputs = self.val_batch_outputs(inputs_gpu)
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]

            for target, output in zip(targets, outputs):
                for box, label, score in zip(
                    output["boxes"], output["labels"], output["scores"]
                ):

                    x1, y1, x2, y2 = box.tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    total_outputs.append(
                        {
                            "image_id": target["image_id"].item(),
                            "category_id": int(label.item()),
                            "bbox": bbox,
                            "score": float(score.item()),
                        }
                    )

            # 計算/更新 confusion matrix
            for target, output in zip(targets, outputs):
                target_boxes: torch.Tensor = target["boxes"]
                target_labels: torch.Tensor = target["labels"]

                score_filter: torch.BoolTensor = (
                    output["scores"] > self._score_threshold
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
                    matched_pairs.append(
                        (target_idx.item(), output_idx.item())
                    )

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
        logging.info(
            f"Epoch {epoch + 1}: Accuracy: {np.trace(cm) / np.sum(cm)}"
        )
        eprint(np.trace(cm), np.sum(cm))
        df_cm = pd.DataFrame(
            cm,
            index=range(NUM_CLASSES + 1),
            columns=range(NUM_CLASSES + 1),
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xlabel("Predictions")
        ax.set_ylabel("Ground Truth")
        sns.heatmap(df_cm, ax=ax, annot=True, cmap="Spectral", fmt="g")
        self._writer.add_figure(
            "Confusion_Matrix (Validation)",
            fig,
            epoch,
        )

        total_outputs_coco = total_targets_coco.loadRes(total_outputs)
        coco_eval = COCOeval(
            total_targets_coco, total_outputs_coco, iouType="bbox"
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        logging.info(coco_eval.stats[0])

        return total_loss / len(data_loader)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        extra_hparams: Optional[Dict[str, Any]] = None,
    ):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch, train_loader)
            val_loss = self.val_epoch(epoch, val_loader)

            self.min_train_loss = min(self.min_train_loss, train_loss)
            self.min_val_loss = min(self.min_val_loss, val_loss)

            self._writer.add_scalars(
                "Loss",
                {"Train": train_loss, "Validation": val_loss},
                global_step=epoch,
            )
            base_hparams = {
                "lr": self._optimizer.param_groups[0]["lr"],
                "weight_decay": self._optimizer.param_groups[0].get(
                    "weight_decay", 0.0
                ),
                "momentum": self._optimizer.param_groups[0].get(
                    "momentum", 0.0
                ),
                "num_epochs": num_epochs,
                "train_batch_size": train_loader.batch_size,
                "val_batch_size": val_loader.batch_size,
                "score_threshold": getattr(self, "_score_threshold", 0.5),
            }
            if extra_hparams:
                base_hparams.update(extra_hparams)
            self._writer.add_hparams(
                hparam_dict=base_hparams,
                metric_dict={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                run_name=self._writer.get_logdir(),
                global_step=epoch,
            )

            logging.info(
                f"Epoch {epoch + 1}: Train Loss: {
                    train_loss:.3f}, Val Loss: {
                    val_loss:.3f}",
            )

    def _move_to_device(self, data, device, non_blocking: bool = True):
        if isinstance(data, list):
            return [
                self._move_to_device(item, device, non_blocking)
                for item in data
            ]
        elif isinstance(data, dict):
            return {
                k: self._move_to_device(v, device, non_blocking)
                for k, v in data.items()
            }
        elif hasattr(data, "to"):
            return data.to(device, non_blocking=non_blocking)
        else:
            return data

    def _compute_loss(
        self, loss_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return (
            1.0 * loss_dict["loss_classifier"]
            + 1.0 * loss_dict["loss_box_reg"]
            + 0.5 * loss_dict["loss_objectness"]
            + 0.5 * loss_dict["loss_rpn_box_reg"]
        )
