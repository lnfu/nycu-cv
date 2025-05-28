import datetime

import click
import torch
from nycu_cv_hw4.constants import MODEL_DIR_PATH
from nycu_cv_hw4.model import PromptIR
from nycu_cv_hw4.trainer import Trainer


@click.command()
@click.option(
    "--description",
    "-d",
    prompt="Description",
    help="Short description for this training run "
    "(e.g., model architecture changes).",
)
def main(description: str):
    # TODO remove
    # SEED = 1234
    # set_seed(SEED)

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda"

    model = PromptIR(decoder=True)
    model = model.to(device)

    ckpt = torch.load(
        MODEL_DIR_PATH / "2025-05-27_15-05-41_epoch=3-loss=0.0193.pth",
        map_location=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    curriculum = [
        # {
        #     "de_types": ["desnow", "derain"],
        #     "patch_size": 64,
        #     "batch_size": 4,
        #     "alpha": 1.0,
        #     "epochs": 10,
        # },
        # {
        #     "de_types": ["desnow", "derain"],
        #     "patch_size": 128,
        #     "batch_size": 2,
        #     "alpha": 0.8,
        #     "epochs": 10,
        # },
        {
            "de_types": ["desnow", "derain"],
            "patch_size": 256,
            "batch_size": 1,
            "alpha": 1.0,
            "epochs": 20,
        },
    ]
    trainer = Trainer(
        model,
        device,
        description,
        run_id=run_id,
    )

    trainer.train_with_curriculum(curriculum)


if __name__ == "__main__":
    main()
