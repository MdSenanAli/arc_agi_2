from path_generator import PathGenerator
from data_loader import DataLoader
from constants import *
from spp_layer import SPPLayer


class Manager:
    def __init__(self, sample_index, dataset_type="train"):
        self.dataset_type = dataset_type
        self.sample_index = sample_index
        self.spp_layer = SPPLayer(levels=[1, 2, 4], pool_types=["max", "avg"])

        if self.dataset_type == "test":
            self.path_generator = PathGenerator(EVAL_DIR, EVAL_LIST, JSON)
        else:
            self.path_generator = PathGenerator(TRAIN_DIR, TRAIN_LIST, JSON)

        self.filepath = self.path_generator.get_filepath_from_index(self.sample_index)

        self.data = DataLoader(self.filepath)

        self.index = -1

    def get_next_data(self):
        num_samples = (
            self.data.train_samples
            if self.dataset_type == "train"
            else self.data.test_samples
        )
        self.index += 1

        if self.index >= num_samples:
            return None, None

        return self.data.get_sample(self.index, set_type=self.dataset_type)
