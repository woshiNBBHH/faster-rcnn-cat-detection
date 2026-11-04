import csv
import os
import random
from pathlib import Path

import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    def expand_environment_variables(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, list):
            return [expand_environment_variables(item) for item in value]
        if isinstance(value, dict):
            return {
                key: expand_environment_variables(item)
                for key, item in value.items()
            }
        return value

    return expand_environment_variables(config)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value):
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

