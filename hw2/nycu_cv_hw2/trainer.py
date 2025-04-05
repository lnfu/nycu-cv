import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import tqdm
from nycu_cv_hw2.utils import compute_iou_matrix, eprint
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
        self._model = model.to(device)
        self._optimizer = optimizer
        self._device = device
        self._writer = writer
        self._score_threshold = score_threshold

    def train_batch(self, inputs: list, targets: list) -> float:
        self._model.train()

        self._optimizer.zero_grad()
        with autocast(device_type=self._device.type):
            loss_dict = self._model(inputs, targets)
            loss: torch.Tensor = sum(loss for loss in loss_dict.values())
        self._scaler.scale(loss).backward()
        self._scaler.step(self._optimizer)
        self._scaler.update()
        return loss.item()

    def val_batch_loss(self, inputs: list, targets: list) -> float:
        self._model.train()

        with autocast(device_type=self._device.type):
            loss_dict = self._model(inputs, targets)
            loss: torch.Tensor = sum(loss for loss in loss_dict.values())
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
            inputs = [input.to(self._device) for input in inputs]  # input = image
            targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

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
            task="multiclass", num_classes=10 + 1  # TODO 10 from dataloader?
        ).to(self._device)

        for inputs, targets in tqdm.tqdm(
            data_loader,
            desc=(f"Training {epoch}"),
            ncols=100,
        ):
            inputs = [input.to(self._device) for input in inputs]  # input = image
            targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

            loss = self.val_batch_loss(inputs, targets)
            total_loss += loss

            outputs = self.val_batch_outputs(inputs)

            # 計算/更新 confusion matrix
            for target, output in zip(targets, outputs):
                target_boxes = target["boxes"]
                target_labels = target["labels"]

                score_filter = output["scores"] > self._score_threshold
                output_boxes = output["boxes"][score_filter]
                output_labels = output["labels"][score_filter]

                if len(target_boxes) == 0 and len(output_boxes) == 0:
                    continue

                if len(target_boxes) == 0:
                    # 全部都是 FP, 假設 background class = 0
                    fake_target_labels = torch.full_like(output_labels, 0).to(
                        self._device
                    )
                    cm.update(output_labels + 1, fake_target_labels)
                    continue

                if len(output_boxes) == 0:
                    # 所有都是 FN, 假設 background class = 0
                    fake_output_labels = torch.full_like(target_labels, 0).to(
                        self._device
                    )
                    cm.update(fake_output_labels, target_labels + 1)
                    continue

                iou_matrix = compute_iou_matrix(target_boxes, output_boxes)

                matched_target_indices = set()
                matched_output_indices = set()
                matched_pairs = []

                while True:
                    # 找到最大的 IoU
                    max_iou = iou_matrix.max()
                    if max_iou < 0.5:  # 全部都 < 0.5 就結束迴圈 TODO 0.5 or other #???
                        break

                    # 找到最大的 IoU 的 index
                    target_idx, output_idx = torch.unravel_index(
                        torch.argmax(iou_matrix), iou_matrix.shape
                    )

                    matched_target_indices.add(target_idx)
                    matched_output_indices.add(output_idx)
                    matched_pairs.append((target_idx, output_idx))

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
                        final_target_labels.append(target_labels[idx].item() + 1)
                        final_output_labels.append(0)

                # FP
                for idx in range(len(output_labels)):
                    if idx not in matched_output_indices:
                        final_target_labels.append(0)
                        final_output_labels.append(output_labels[idx].item() + 1)

                cm.update(
                    torch.tensor(final_target_labels).to(self._device),
                    torch.tensor(final_output_labels).to(self._device),
                )

        cm = cm.compute()
        df_cm = pd.DataFrame(
            cm.cpu().numpy(), index=range(10 + 1), columns=range(10 + 1)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_cm, ax=ax, annot=True, cmap="Spectral", fmt="g")
        self._writer.add_figure(
            f"Confusion_Matrix (Validation)",
            fig,
            epoch,
        )

        return total_loss / len(data_loader)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
    ):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch, train_loader)
            val_loss = self.val_epoch(epoch, val_loader)

            self.min_train_loss = min(self.min_train_loss, train_loss)
            self.min_val_loss = min(self.min_val_loss, val_loss)

            self._writer.add_scalars(
                "Loss", {"Train": train_loss, "Validation": val_loss}, epoch
            )

            self._writer.add_hparams(
                hparam_dict={
                    "lr": self._optimizer.param_groups[0]["lr"],
                    "num_epochs": num_epochs,
                    "train_batch_size": train_loader.batch_size,
                    "val_batch_size": val_loader.batch_size,
                    "score_threshold": self._score_threshold,
                },
                metric_dict={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
            )

            logging.info(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}",
            )
