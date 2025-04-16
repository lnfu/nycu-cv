import pathlib

DATA_DIR_PATH = pathlib.Path("data")
LOG_DIR_PATH = pathlib.Path("logs")
MODEL_DIR_PATH = pathlib.Path("models")
MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_PATH = pathlib.Path("outputs")
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 10  # "0-9" (category_id = 1-10)
IOU_THRESHOLD = 0.5  # validation 計算 IoU 的時候才會用到
