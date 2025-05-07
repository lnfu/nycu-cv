import datetime
import logging

import click
import torch
import torch.optim as optim
from nycu_cv_hw3.config import settings
from nycu_cv_hw3.data import train_loader, val_loader
from nycu_cv_hw3.models import Model
from nycu_cv_hw3.trainer import Trainer
from nycu_cv_hw3.utils import eprint
from torch.optim.swa_utils import SWALR, AveragedModel

# from timm.scheduler import CosineLRScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="train.log",
)
logging.info("==========================================")


@click.command()
@click.option(
    "--description",
    "-d",
    prompt="Description",
    help="Short description for this training run "
    "(e.g., model architecture changes).",
)
def main(description: str):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(current_time)

    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"{device=}")

    model = Model().to(device)
    swa_model = AveragedModel(model).to(device)

    # Optimizer
    # optimizer = optim.AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=settings.learning_rate,
    # )

    # param_groups = [
    #     {"params": model.maskrcnn.backbone.parameters(), "lr": 5e-3},
    #     {"params": model.maskrcnn.rpn.parameters(), "lr": 5e-3},
    #     {"params": model.maskrcnn.roi_heads.parameters(), "lr": 5e-3},
    # ]
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        # param_groups,
        lr=5e-3,
        weight_decay=0,
        momentum=0.9,
    )
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=settings.swa_lr,
        anneal_epochs=settings.anneal_epochs,
    )

    # lr_scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=30,
    #     warmup_t=5,
    #     warmup_lr_init=1e-5,
    #     lr_min=1e-4,
    #     t_in_epochs=True,
    # )
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=settings.learning_rate,
    #     momentum=settings.momentum,
    #     weight_decay=settings.weight_decay,
    # )

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=settings.factor,
    #     patience=settings.patience,
    #     verbose=True,
    # )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        train_id=current_time,
        device=device,
        description=description,
        swa_model=swa_model,
        swa_scheduler=swa_scheduler,
    )

    try:
        trainer.train(
            num_epochs=settings.num_epochs,
        )
    except KeyboardInterrupt:
        eprint("Training interrupted! Saving model before exiting...")
    finally:
        pass


if __name__ == "__main__":
    main()
