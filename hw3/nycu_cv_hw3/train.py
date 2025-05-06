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

# from torch.optim.swa_utils import SWALR, AveragedModel

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
    # swa_model = AveragedModel(model).to(device)

    # Optimizer
    # optimizer = optim.AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=settings.learning_rate,
    # )
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        momentum=settings.momentum,
    )
    # swa_scheduler = SWALR(optimizer, swa_lr=0.0005)  # TODO setting

    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=settings.learning_rate,
    #     momentum=settings.momentum,
    #     weight_decay=settings.weight_decay,
    # )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=settings.step_size,
        gamma=settings.gamma,
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
    )

    try:
        trainer.train(
            num_epochs=settings.num_epochs,
        )
    except KeyboardInterrupt:
        eprint("Training interrupted! Saving model before exiting...")
    finally:
        trainer.save_checkpoint(-1)


if __name__ == "__main__":
    main()
