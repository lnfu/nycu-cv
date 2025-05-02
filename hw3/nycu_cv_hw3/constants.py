import pathlib

DATA_DIR_PATH = pathlib.Path("data")
LOG_DIR_PATH = pathlib.Path("logs")
MODEL_DIR_PATH = pathlib.Path("models")
MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_PATH = pathlib.Path("outputs")
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5
MASK_THRESHOLD = 0.5  # TODO
