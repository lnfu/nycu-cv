import datetime
import logging

import torch
import torchvision
from nycu_cv_hw2.constants import LOG_DIR_PATH, MODEL_DIR_PATH
from nycu_cv_hw2.data import train_loader, val_loader
from nycu_cv_hw2.trainer import Trainer
from nycu_cv_hw2.utils import eprint
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="basic.log",  # TODO 改成加到 tensorboard?
)


def main():
    # Tensorboard
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(LOG_DIR_PATH / current_time)

    # Device
    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")
    writer.add_text("Device", device.type)

    # Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes=10
        )
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters())

    # Trainer
    trainer = Trainer(model, optimizer, device, writer)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        eprint("Training interrupted! Saving model before exiting...")
    finally:
        torch.save(model, MODEL_DIR_PATH / f"{current_time}.pt")
        eprint(f"Model: {current_time}.pt")
        pass


if __name__ == "__main__":
    main()
