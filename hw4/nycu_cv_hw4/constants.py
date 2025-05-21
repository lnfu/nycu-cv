import pathlib

DATA_DIR_PATH = pathlib.Path("data/hw4_realse_dataset")
LOG_DIR_PATH = pathlib.Path("logs")
MODEL_DIR_PATH = pathlib.Path("models")
MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_PATH = pathlib.Path("outputs")
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
DEBUG_DIR_PATH = pathlib.Path("debug")
DEBUG_DIR_PATH.mkdir(parents=True, exist_ok=True)
