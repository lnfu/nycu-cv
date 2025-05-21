import datetime
import gc
import heapq
from pathlib import Path

import torch
import tqdm
from nycu_cv_hw4.config import settings
from nycu_cv_hw4.constants import LOG_DIR_PATH, MODEL_DIR_PATH
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        description: str,
        save_top_k: int = 3,
        save_last: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # loss & optim
        # TODO loss = L1 + 0.1 * Perceptual + 0.5 * (1 - SSIM)
        self.loss_fn = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=settings.lr)

        # tensorboard
        self.id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(LOG_DIR_PATH / self.id)
        self.writer.add_text("description", description)

        # checkpoint
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.best_k_models = []  # min-heap of (loss, path)

    def train(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            train_avg_loss = self.train_one_epoch(epoch)
            val_avg_loss = self.validate(epoch)

            print(f"[Epoch {epoch}] TRAIN AVG LOSS: {train_avg_loss:.4f}")
            print(f"[Epoch {epoch}] VAL AVG LOSS: {val_avg_loss:.4f}")

            # tensorboard
            self.writer.add_scalars(
                "Loss",
                {"train": train_avg_loss, "val": val_avg_loss},
                global_step=epoch,
            )

            # checkpoint
            # TODO 也考慮 validation
            ckpt_path = self.save_checkpoint(epoch, train_avg_loss)
            self.track_best_checkpoints(train_avg_loss, ckpt_path)

            gc.collect()
            torch.cuda.empty_cache()

    @torch.enable_grad()
    def train_one_epoch(self, epoch: int):
        self.model.train()

        total_loss = 0.0
        num_samples = 0

        for batch in tqdm.tqdm(
            self.train_loader, desc=f"train {epoch}", ncols=100
        ):
            loss = self.training_step(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # track loss
            batch_size = batch[2].size(0)  # clean_patch
            total_loss += loss.item() * batch_size
            num_samples += batch_size

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        num_samples = 0

        for batch in tqdm.tqdm(
            self.val_loader, desc=f"train {epoch}", ncols=100
        ):
            loss = self.training_step(batch)

            # track loss
            batch_size = batch[2].size(0)  # clean_patch
            total_loss += loss.item() * batch_size
            num_samples += batch_size

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        return avg_loss

    def training_step(self, batch):

        ([image_id, de_id], de_patch, clean_patch) = batch

        de_patch = de_patch.to(self.device)
        clean_patch = clean_patch.to(self.device)

        restored = self.model(de_patch)
        loss = self.loss_fn(restored, clean_patch)

        return loss

    def save_checkpoint(self, epoch: int, loss: float) -> Path:
        # Save to path: epoch=E-loss=XX.XX.pth
        ckpt_path = (
            MODEL_DIR_PATH / f"{self.id}_epoch={epoch}-loss={loss:.4f}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            ckpt_path,
        )

        # Save latest if enabled
        if self.save_last:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                },
                MODEL_DIR_PATH / f"{self.id}_last.pth",
            )

        return ckpt_path

    def track_best_checkpoints(self, loss: float, ckpt_path: Path):

        # heapq 是 min-heap 所以要用 -loss
        heapq.heappush(self.best_k_models, (-loss, ckpt_path))

        # 拿掉最小的 (最大 loss)
        if len(self.best_k_models) > self.save_top_k:
            _, remove_path = heapq.heappop(self.best_k_models)
            if remove_path.exists():
                remove_path.unlink()
