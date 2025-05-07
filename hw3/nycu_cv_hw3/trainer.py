import logging

import torch
import torch.nn as nn
import tqdm
from nycu_cv_hw3.constants import LOG_DIR_PATH, MASK_THRESHOLD, MODEL_DIR_PATH
from nycu_cv_hw3.utils import encode_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        train_id: str,
        device,
        description: str,
        swa_model,
        swa_scheduler,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.train_id = train_id
        self.writer = SummaryWriter(LOG_DIR_PATH / train_id)
        self.writer.add_text("description", description)
        logging.info(f"Description: {description}")

        self.scaler = torch.amp.GradScaler()

        self.swa_model = swa_model
        self.swa_scheduler = swa_scheduler

    def train(self, num_epochs: int):
        min_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.val_loss(epoch)
            val_ap_metrics = self.val_ap_metrics(epoch)

            # TODO
            if epoch > num_epochs - 10:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                self.save_checkpoint(epoch, True)

            else:

                if self.lr_scheduler:
                    # self.lr_scheduler.step(val_loss)
                    self.lr_scheduler.step()

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    self.save_checkpoint(epoch, False)

            # tensorboard
            self.writer.add_scalars(
                "Loss",
                {
                    "train": train_loss,
                    "val": val_loss,
                },
                global_step=epoch,
            )
            self.writer.add_hparams(
                hparam_dict={
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "momentum": self.optimizer.param_groups[0].get(
                        "momentum", 0.0
                    ),
                    "weight_decay": self.optimizer.param_groups[0].get(
                        "weight_decay", 0.0
                    ),
                },
                metric_dict={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **val_ap_metrics,
                },
                run_name=self.train_id,
                global_step=epoch,
            )

    def loss_fn(self, loss_dict):
        # sum(loss for loss in loss_dict.values())
        return (
            0.15 * loss_dict["loss_classifier"]
            + 0.15 * loss_dict["loss_box_reg"]
            + 0.55 * loss_dict["loss_mask"]
            + 0.15 * loss_dict["loss_objectness"]
        )

    @torch.no_grad()
    def val_loss(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        total_count = 0

        for images, targets in tqdm.tqdm(
            self.val_loader, desc=(f"val_loss {epoch}"), ncols=100
        ):
            images = list(image.to(self.device) for image in images)
            targets = [
                {k: v.to(self.device) for k, v in target.items()}
                for target in targets
            ]

            with torch.amp.autocast(
                device_type=self.device.type, cache_enabled=True
            ):
                loss_dict = self.model(images, targets)
                loss = self.loss_fn(loss_dict)

            batch_size = len(images)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            torch.cuda.empty_cache()

        avg_loss = total_loss / total_count
        logging.info(f"[Epoch {epoch}] Validation Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.enable_grad()
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        total_count = 0

        for images, targets in tqdm.tqdm(
            self.train_loader, desc=(f"train {epoch}"), ncols=100
        ):
            images = list(image.to(self.device) for image in images)
            targets = [
                {k: v.to(self.device) for k, v in target.items()}
                for target in targets
            ]

            with torch.amp.autocast(
                device_type=self.device.type, cache_enabled=True
            ):
                loss_dict = self.model(images, targets)
                loss = self.loss_fn(loss_dict)

            batch_size = len(images)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            torch.cuda.empty_cache()

        avg_loss = total_loss / total_count
        logging.info(f"[Epoch {epoch}] Training Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def val_ap_metrics(self, epoch: int):
        self.model.eval()

        total_images = []
        total_targets = []
        total_outputs = []

        for images, targets in tqdm.tqdm(
            self.val_loader, desc=(f"val_ap {epoch}"), ncols=100
        ):
            images = list(image.to(self.device) for image in images)
            # targets = [
            #     {k: v.to(self.device) for k, v in target.items()}
            #     for target in targets
            # ]

            with torch.amp.autocast(
                device_type=self.device.type, cache_enabled=True
            ):
                outputs = self.model(images)

            for image, target, output in zip(images, targets, outputs):
                total_images.append(image.cpu())
                total_targets.append({k: v.cpu() for k, v in target.items()})
                total_outputs.append({k: v.cpu() for k, v in output.items()})

            torch.cuda.empty_cache()

        coco_gt_dict = {"images": [], "annotations": [], "categories": []}

        # category_id
        category_ids = sorted(
            {
                int(label)
                for target in total_targets
                for label in target["labels"].numpy()
            }
        )
        coco_gt_dict["categories"] = [
            {"id": category_id, "name": str(category_id)}
            for category_id in category_ids
        ]

        annotation_id = 1
        for image, target in zip(total_images, total_targets):
            image_id = int(target["image_id"])
            _, h, w = image.shape
            coco_gt_dict["images"].append(
                {"id": image_id, "width": w, "height": h}
            )

            boxes = target["boxes"].numpy()
            labels = target["labels"].numpy()
            masks = target["masks"].numpy()

            for box, label, mask in zip(boxes, labels, masks):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                rle = encode_mask(mask)

                coco_gt_dict["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                        "segmentation": rle,
                    }
                )
                annotation_id += 1

        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()

        results = []
        for target, output in zip(total_targets, total_outputs):
            image_id = int(target["image_id"])

            boxes = output["boxes"].numpy()
            labels = output["labels"].numpy()
            masks = output["masks"].numpy()
            scores = output["scores"].numpy()

            for box, score, label, mask in zip(boxes, scores, labels, masks):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                mask = mask.squeeze()  # (1, H, W) -> (H, W)
                mask = mask > MASK_THRESHOLD

                rle = encode_mask(mask)

                result = {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                    "segmentation": rle,
                }
                results.append(result)

        coco_dt = coco_gt.loadRes(results)

        print("GT ann:", len(coco_gt.getAnnIds()))
        print("DT ann:", len(coco_dt.getAnnIds()))
        print("GT imgs:", coco_gt.getImgIds()[:10], "…")
        print("DT imgs:", coco_dt.getImgIds())
        print("GT cats:", coco_gt.getCatIds())
        print("DT cats:", coco_dt.getCatIds())

        metrics = {}
        for iou_type in ("bbox", "segm"):
            coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
            coco_eval.evaluate()
            coco_eval.accumulate()
            print(f"\n===== COCOeval {iou_type} =====")
            coco_eval.summarize()

            metrics[f"{iou_type}/ap"] = coco_eval.stats[0]
            metrics[f"{iou_type}/ap50"] = coco_eval.stats[1]
            metrics[f"{iou_type}/ap75"] = coco_eval.stats[2]
            metrics[f"{iou_type}/ar"] = coco_eval.stats[8]
        return metrics

    def save_checkpoint(self, epoch: int, use_swa: bool = False):
        model_to_save = self.swa_model if use_swa else self.model

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            MODEL_DIR_PATH
            / f"{self.train_id}_{epoch}{'_swa' if use_swa else ''}.pth",
        )
