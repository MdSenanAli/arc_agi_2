import torch
import numpy as np
import json
import torch.nn.functional as F


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train = None
        self.test = None

        self.initialize()

    def initialize(self):
        self.load_data()
        self.train_samples = len(self.train) if self.train else 0
        self.test_samples = len(self.test) if self.test else 0

    def prepare_matrix(self, matrix):
        tensor = torch.tensor(matrix, dtype=torch.float32)
        return tensor

    def load_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)

        for section in ["train", "test"]:
            if section in data:
                setattr(
                    self,
                    section,
                    [
                        {
                            "input": self.prepare_matrix(np.array(sample["input"])),
                            "output": self.prepare_matrix(np.array(sample["output"])),
                        }
                        for sample in data[section]
                    ],
                )

    def get_sample(self, index, set_type="train"):
        if set_type not in ["train", "test"]:
            raise ValueError("set_type must be 'train' or 'test'")

        data_set = getattr(self, set_type)
        if data_set is None:
            raise ValueError(f"{set_type.capitalize()} data not loaded.")

        if not (0 <= index < len(data_set)):
            raise IndexError(f"Index {index} out of range for {set_type} data.")

        return data_set[index]["input"], data_set[index]["output"]
