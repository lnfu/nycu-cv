import click
import torch
from nycu_cv_hw4.config import settings
from nycu_cv_hw4.constants import DATA_DIR_PATH
from nycu_cv_hw4.data import PromptTrainDataset
from nycu_cv_hw4.model import PromptIR
from nycu_cv_hw4.trainer import Trainer
from nycu_cv_hw4.utils import set_seed
from torch.utils.data import DataLoader


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
    SEED = 1234
    set_seed(SEED)

    device = torch.torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda"

    dataset = PromptTrainDataset(DATA_DIR_PATH / "train")

    train_size = int(settings.train_ratio * len(dataset))

    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    train_set.dataset.train = True
    val_set.dataset.train = False

    def get_data_loader(dataset, shuffle):
        return DataLoader(
            dataset=dataset,
            batch_size=settings.batch_size,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
            num_workers=16,
            generator=torch.Generator().manual_seed(SEED),
        )

    train_loader = get_data_loader(train_set, True)
    val_loader = get_data_loader(val_set, False)

    model = PromptIR(decoder=True)
    model = model.to(device)

    trainer = Trainer(model, train_loader, val_loader, device, description)
    trainer.train(num_epochs=20)


if __name__ == "__main__":
    main()
