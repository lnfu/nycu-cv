import gc
import heapq
from pathlib import Path

import torch
import tqdm
from nycu_cv_hw4.constants import DATA_DIR_PATH, LOG_DIR_PATH, MODEL_DIR_PATH
from nycu_cv_hw4.data import PromptTrainDataset
from nycu_cv_hw4.utils import compute_psnr
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def rebuild_dataloader(
        self, de_types=["derain", "desnow"], patch_size=128, batch_size=1
    ):
        dataset = PromptTrainDataset(
            DATA_DIR_PATH / "train", de_types, patch_size=patch_size
        )

        # train_size = int(settings.train_ratio * len(dataset))
        # train_set, val_set = torch.utils.data.random_split(
        #     dataset, [train_size, len(dataset) - train_size]
        # )

        train_set = dataset

        # train_set.dataset.train = True
        train_set.train = True
        # val_set.dataset.train = False

        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=16,
            # generator=torch.Generator().manual_seed(SEED),
        )

        # self.val_loader = DataLoader(
        #     val_set,
        #     batch_size=1,  # TODO
        #     pin_memory=True,
        #     shuffle=False,
        #     drop_last=False,
        #     num_workers=16,
        #     generator=torch.Generator().manual_seed(SEED),
        # )

    def __init__(
        self,
        model,
        device: torch.device,
        description: str,
        save_top_k: int = 3,
        save_last: bool = True,
        run_id="default",
        # ckpt_path=None, # TODO
    ):
        self.alpha = 0.8
        self.start_epoch = 1

        self.model = model
        self.device = device

        # tensorboard
        self.id = run_id
        self.writer = SummaryWriter(LOG_DIR_PATH / self.id)
        self.writer.add_text("description", description)

        # checkpoint
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.best_k_models = []  # min-heap of (loss, path)

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        ckpt = torch.load(
            MODEL_DIR_PATH / "2025-05-28_07-14-55_epoch=4-loss=0.0266.pth",
            map_location=device,
        )
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=30, eta_min=1e-4
        # )

        # Data
        self.rebuild_dataloader()

    # def loss_fn(self, pred, target):
    #     if self.alpha == 1.0:
    #         return torch.nn.functional.l1_loss(pred, target)

    #     l1_loss = torch.nn.functional.l1_loss(pred, target)
    #     ssim_loss = 1 - ms_ssim(
    #         pred, target, data_range=1.0, size_average=True, win_size=7
    #     )
    #     return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss

    def loss_fn(self, pred, target, de_type: str):
        if de_type == "desnow":
            return torch.nn.functional.l1_loss(pred, target)
        elif de_type == "derain":
            l1_loss = torch.nn.functional.l1_loss(pred, target)
            ssim_loss = 1 - ms_ssim(
                pred, target, data_range=1.0, size_average=True, win_size=7
            )
            return l1_loss + 0.3 * ssim_loss
        else:
            raise ValueError(f"Unknown de_type: {de_type}")

    def train_with_curriculum(self, curriculum):
        for stage_id, stage_cfg in enumerate(curriculum):
            print(f"==== Stage {stage_id} ====")
            self.rebuild_dataloader(
                stage_cfg["de_types"],
                stage_cfg["patch_size"],
                stage_cfg["batch_size"],
            )
            self.alpha = stage_cfg["alpha"]
            self.train(stage_cfg["epochs"])
            self.start_epoch += stage_cfg["epochs"]

            print(f"{self.alpha=}")

    def train(self, num_epochs: int):

        for epoch in range(
            self.start_epoch, self.start_epoch + num_epochs + 1
        ):
            train_avg_loss = self.train_one_epoch(epoch)
            # val_avg_loss = self.validate(epoch)

            print(f"[Epoch {epoch}] TRAIN AVG LOSS: {train_avg_loss:.4f}")
            # print(f"[Epoch {epoch}] VAL AVG LOSS: {val_avg_loss:.4f}")

            # self.scheduler.step()

            # tensorboard
            self.writer.add_scalars(
                "Loss",
                {
                    "train": train_avg_loss,
                    # "val": val_avg_loss,
                },
                global_step=epoch,
            )

            # checkpoint
            ckpt_path = self.save_checkpoint(epoch, train_avg_loss)
            self.track_best_checkpoints(train_avg_loss, ckpt_path)

            gc.collect()
            torch.cuda.empty_cache()

    @torch.enable_grad()
    def train_one_epoch(self, epoch: int):
        self.model.train()

        total_loss = 0.0
        total_psnr = 0.0
        num_samples = 0

        for batch in tqdm.tqdm(
            self.train_loader, desc=f"train {epoch}", ncols=100
        ):
            loss, psnr = self.training_step(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # track loss
            batch_size = batch[2].size(0)  # clean_patch
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            total_psnr += psnr

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
        print(f"{avg_psnr=}")

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        total_psnr = 0.0
        num_samples = 0

        for batch in tqdm.tqdm(
            self.val_loader, desc=f"validate {epoch}", ncols=100
        ):
            loss, psnr = self.training_step(batch)

            # track loss
            batch_size = batch[2].size(0)  # clean_patch
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            total_psnr += psnr

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0

        print(f"{avg_psnr=}")

        return avg_loss

    def training_step(self, batch):

        ([image_id, de_id], de_patch, clean_patch) = batch

        de_patch = de_patch.to(self.device)
        clean_patch = clean_patch.to(self.device)

        restored = self.model(de_patch)

        inv_de_dict = {0: "derain", 1: "desnow"}
        de_type = inv_de_dict[de_id.item()]
        loss = self.loss_fn(restored, clean_patch, de_type)

        psnr = compute_psnr(restored, clean_patch)

        return loss, psnr

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
