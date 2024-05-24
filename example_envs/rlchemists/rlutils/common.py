from pathlib import Path
import numpy as np


def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent


def load_en_array(en_array_path):
    with open(en_array_path, "rb") as f:
        en_array = np.load(f)
    print(f"load en_array from {en_array_path} ")
    return en_array
