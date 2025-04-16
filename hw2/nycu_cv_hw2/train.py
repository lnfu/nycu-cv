import datetime
import logging

import torch
import torchvision.models.detection as detection
from nycu_cv_hw2.config import settings
from nycu_cv_hw2.constants import LOG_DIR_PATH, MODEL_DIR_PATH, NUM_CLASSES
from nycu_cv_hw2.data import get_data_loader
from nycu_cv_hw2.predictor import CustomFastRCNNPredictor
from nycu_cv_hw2.trainer import Trainer
from nycu_cv_hw2.utils import eprint
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="basic.log",
)


def main():
    # TODO: https://github.com/pytorch/vision/issues/978

    # Tensorboard
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(LOG_DIR_PATH / current_time)

    # Device
    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")
    writer.add_text("Device", device.type)

    # Model
    model = detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    )
    model.rpn.pre_nms_top_n_train = 400
    model.rpn.post_nms_top_n_train = 400
    model.rpn.pre_nms_top_n_test = 200
    model.rpn.post_nms_top_n_test = 200

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = CustomFastRCNNPredictor(
        in_channels=in_features, num_classes=NUM_CLASSES + 1, dropout=0.0
    )
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=settings.learning_rate,
        momentum=settings.momentum,
        weight_decay=settings.weight_decay,
    )
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=settings.learning_rate
    # )

    # Trainer
    trainer = Trainer(
        model, optimizer, device, writer, settings.score_threshold
    )

    # lr_scheduler =
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    try:
        trainer.train(
            train_loader=get_data_loader(settings.batch_size, "train"),
            val_loader=get_data_loader(settings.batch_size, "val"),
            num_epochs=settings.num_epochs,
            extra_hparams={"dropout": 0.0},
        )
    except KeyboardInterrupt:
        eprint("Training interrupted! Saving model before exiting...")
    finally:
        torch.save(model, MODEL_DIR_PATH / f"{current_time}.pt")
        eprint(f"Model name: {current_time}")
        pass


if __name__ == "__main__":
    main()
