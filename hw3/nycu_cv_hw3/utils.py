import sys

import numpy as np
from pycocotools import mask as mask_utils


def eprint(*args, **kwargs):
    """
    Print to stderr instead of stdout.
    """
    print(*args, file=sys.stderr, **kwargs)


def encode_mask(mask: np.ndarray) -> dict:
    arr = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(arr)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
