import logging
import pathlib

import torch
import torchvision
import tqdm
from nycu_cv_hw2.data import train_loader, val_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="basic.log",  # TODO 改成加到 tensorboard?
)


def main():

    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    # Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes=10
        )
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters())

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    try:
        num_epoch = 10  # TODO
        for epoch in range(num_epoch):
            model.train()
            for inputs, labels in tqdm.tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}", ncols=100
            ):
                inputs = [input.to(device) for input in inputs]  # input = image
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

                optimizer.zero_grad()

                loss_dict = model(inputs, labels)
                # TODO 試試看加權
                losses = sum(loss for loss in loss_dict.values())

                # backward propagation
                losses.backward()
                optimizer.step()

                # running_loss += losses.item()

            model.eval()  # dropout/batchnorm
            with torch.no_grad():  # autograd

                for inputs, labels in tqdm.tqdm(
                    val_loader, desc=f"Epoch {epoch + 1}/{num_epoch}", ncols=100
                ):
                    inputs = [input.to(device) for input in inputs]  # input = image
                    labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

                    loss_dict = model(inputs, labels)
                    # TODO 試試看加權
                    losses = sum(loss for loss in loss_dict.values())

    except KeyboardInterrupt:
        logging.warning("Training interrupted! Saving model before exiting...")
    finally:
        # save model
        pass


if __name__ == "__main__":
    main()
