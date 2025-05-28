# TODO tensor_to_image

import csv
from pathlib import Path

import numpy as np
import torch
from nycu_cv_hw4.constants import (
    DATA_DIR_PATH,
    MODEL_DIR_PATH,
    OUTPUT_DIR_PATH,
)
from nycu_cv_hw4.model import PromptIR
from nycu_cv_hw4.utils import crop_img
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm


def load_model(model_path: Path, device: torch.device):
    model = PromptIR(decoder=True).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def preprocess_image(img_path: Path):
    ori_img = np.array(Image.open(img_path).convert("RGB"))

    # crop
    img = crop_img(ori_img)

    # to tensor
    to_tensor = ToTensor()
    img = to_tensor(img)

    # (1, 3, H, W), original shape
    return img.unsqueeze(0), ori_img.shape


def run(
    model,
    image_dir: Path,
    output_dir: Path,
    device: torch.device,
    debug: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dict = {}

    image_paths = sorted(image_dir.glob("*.png"))
    for img_path in tqdm(image_paths, desc="Inference"):
        img, ori_shape = preprocess_image(img_path)

        img = img.to(device)

        with torch.no_grad():
            output = model(img).clamp(0, 1)

        output = output.squeeze(0).cpu()  # (3, H, W)
        output = (output.numpy() * 255.0).round().astype(np.uint8)

        # restore to original size if needed (optional, not done here)

        output_dict[img_path.name] = output

        # Save individual PNG
        # TODO save to DEBUG_DIR not OUTPUT_DIR
        if debug:
            # (H, W, C)
            output_img = Image.fromarray(output.transpose(1, 2, 0))
            output_img.save(output_dir / img_path.name)

    np.savez_compressed(output_dir / "pred.npz", **output_dict)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda"

    ckpt_snow = input("snow: ")
    ckpt_rain = input("rain: ")

    type_map = {}
    head = True
    with open("temp.csv", newline="") as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:

            if head:
                head = False
                continue

            type_map[row[0]] = int(row[1])

    # model = load_model(MODEL_DIR_PATH / ckpt, device)
    output_dir = OUTPUT_DIR_PATH / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dict = {}
    image_dir = DATA_DIR_PATH / "test/degraded"
    image_paths = sorted(image_dir.glob("*.png"))
    for img_path in tqdm(image_paths, desc="Inference"):
        de_type = type_map[img_path.stem]

        if de_type == 0:
            model = load_model(MODEL_DIR_PATH / ckpt_rain, device)
        elif de_type == 1:
            model = load_model(MODEL_DIR_PATH / ckpt_snow, device)
        else:
            print("ERROR")
            exit(0)

        img, ori_shape = preprocess_image(img_path)

        img = img.to(device)

        with torch.no_grad():
            output = model(img).clamp(0, 1)

        output = output.squeeze(0).cpu()  # (3, H, W)
        output = (output.numpy() * 255.0).round().astype(np.uint8)

        # restore to original size if needed (optional, not done here)

        output_dict[img_path.name] = output

        # Save individual PNG
        # TODO save to DEBUG_DIR not OUTPUT_DIR
        # if True:
        #     # (H, W, C)
        output_img = Image.fromarray(output.transpose(1, 2, 0))
        output_img.save(output_dir / img_path.name)

        del model

    np.savez_compressed(output_dir / "pred.npz", **output_dict)


if __name__ == "__main__":
    main()
