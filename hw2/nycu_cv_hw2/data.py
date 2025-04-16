import logging
import pathlib
import typing

import torch
import torchvision
from nycu_cv_hw2.constants import DATA_DIR_PATH
from nycu_cv_hw2.exceptions import InvalidPathError
from PIL import Image, ImageFile
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class TestDataset(torch.utils.data.Dataset):
    _img_file_paths: typing.List[pathlib.Path]

    def __init__(
        self,
        image_dir_path: typing.Union[str, pathlib.Path],
        transform: typing.Optional[typing.Callable] = None,
    ):
        if isinstance(image_dir_path, str):
            image_dir_path = pathlib.Path(image_dir_path)

        if not isinstance(image_dir_path, pathlib.Path):
            raise InvalidPathError()

        # 字典序
        self._img_file_paths = sorted(
            pathlib.Path(image_dir_path).glob("*.png")
        )
        self._transform = transform

    def __len__(self) -> int:
        return len(self._img_file_paths)

    def __getitem__(self, idx) -> typing.Tuple[ImageFile.ImageFile, str]:
        img_file_path = self._img_file_paths[idx]
        img = Image.open(img_file_path).convert("RGB")
        if self._transform:
            img = self._transform(img)
        return img, img_file_path.stem


default_transform = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()

# transform = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.ColorJitter(
#             brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4
#         ),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         ),
#     ]
# )

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        ),
        torchvision.transforms.RandomGrayscale(p=0.1),
        torchvision.transforms.RandomInvert(p=0.1),
        torchvision.transforms.ToTensor(),
    ]
)

# Dataset
train_dataset = torchvision.datasets.CocoDetection(
    root=DATA_DIR_PATH / "train",
    annFile=DATA_DIR_PATH / "train.json",
    transform=transform,  # TODO
    target_transform=None,  # TODO
)
val_dataset = torchvision.datasets.CocoDetection(
    root=DATA_DIR_PATH / "valid",
    annFile=DATA_DIR_PATH / "valid.json",
    transform=transform,  # TODO
    target_transform=None,  # TODO
)
test_dataset = TestDataset(
    image_dir_path=DATA_DIR_PATH / "test",
    transform=default_transform,  # TODO
)


# DataLoader
def train_and_val_collate_fn(batch):
    inputs, labels = zip(*batch)

    new_labels = []
    for label in labels:  # label is list of dicts
        boxes = []
        # bbox = (x_min, y_min, width, height) -> bbox = (x_min, y_min, x_max,
        # y_max)
        for label_entry in label:
            x_min, y_min, w, h = label_entry["bbox"]
            x_max = x_min + w
            y_max = y_min + h
            if w <= 0 or h <= 0:
                logging.warning(
                    f"Found invalid box {label_entry['bbox']} -> Skipping."
                )
                continue
            boxes.append([x_min, y_min, x_max, y_max])

        new_label = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(
                [label_entry["category_id"] for label_entry in label],
                dtype=torch.int64,
            ),
            "image_id": torch.tensor(
                [label[0]["image_id"]], dtype=torch.int64
            ),
            "area": torch.tensor(
                [label_entry["area"] for label_entry in label],
                dtype=torch.float32,
            ),
            "iscrowd": torch.tensor(
                [label_entry["iscrowd"] for label_entry in label],
                dtype=torch.int64,
            ),
        }
        new_labels.append(new_label)

    return inputs, new_labels


def test_collate_fn(batch):
    images, image_ids = zip(*batch)
    return images, image_ids


def get_data_loader(batch_size: int, split: str):
    assert split in ["train", "val", "test"]
    if split == "train":
        dataset = train_dataset
        collate_fn = train_and_val_collate_fn
    elif split == "val":
        dataset = val_dataset
        collate_fn = train_and_val_collate_fn
    elif split == "test":
        dataset = test_dataset
        collate_fn = test_collate_fn
    else:
        raise ValueError()

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
