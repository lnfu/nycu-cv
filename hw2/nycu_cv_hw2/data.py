import logging
import pathlib
import typing

import torch
import torchvision
from nycu_cv_hw2.constants import DATA_DIR_PATH
from nycu_cv_hw2.exceptions import InvalidPathError
from PIL import Image, ImageFile


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
        self._img_file_paths = sorted(pathlib.Path(image_dir_path).glob("*.png"))
        self._transform = transform

    def __len__(self) -> int:
        return len(self._img_file_paths)

    def __getitem__(self, idx) -> typing.Tuple[ImageFile.ImageFile, str]:
        img_file_path = self._img_file_paths[idx]
        img = Image.open(img_file_path).convert("RGB")
        if self._transform:
            img = self._transform(img)
        return img, img_file_path.stem


# Dataset
train_dataset = torchvision.datasets.CocoDetection(
    root=DATA_DIR_PATH / "train",
    annFile=DATA_DIR_PATH / "train.json",
    transform=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms(),  # TODO
    target_transform=None,
)
val_dataset = torchvision.datasets.CocoDetection(
    root=DATA_DIR_PATH / "valid",
    annFile=DATA_DIR_PATH / "valid.json",
    transform=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms(),  # TODO
    target_transform=None,
)
test_dataset = TestDataset(
    image_dir_path=DATA_DIR_PATH / "test",
    transform=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms(),  # TODO
)


# DataLoader
def train_and_val_collate_fn(batch):
    inputs, labels = zip(*batch)

    new_labels = []
    for label in labels:  # label is list of dicts
        boxes = []
        # bbox = (x_min, y_min, width, height) -> bbox = (x_min, y_min, x_max, y_max)
        for t in label:
            x_min, y_min, w, h = t["bbox"]
            x_max = x_min + w
            y_max = y_min + h
            if w <= 0 or h <= 0:
                logging.warning(f"Found invalid box {t['bbox']} -> Skipping.")
                continue
            boxes.append([x_min, y_min, x_max, y_max])

        new_label = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(
                [t["category_id"] - 1 for t in label],  # category_id =  1-10 -> 0-9
                dtype=torch.int64,
            ),
            "image_id": torch.tensor([label[0]["image_id"]], dtype=torch.int64),
            "area": torch.tensor([t["area"] for t in label], dtype=torch.float32),
            "iscrowd": torch.tensor([t["iscrowd"] for t in label], dtype=torch.int64),
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
