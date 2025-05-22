from os.path import join
from constants import *


class PathGenerator:
    def __init__(self, data_dir=TRAIN_DIR, list_file=TRAIN_LIST, extention=".json"):
        self.data_dir = data_dir
        self.list_file = list_file
        self.filenames = list()
        self.extention = extention
        self.add_filenames()

    def add_filenames(self):
        with open(self.list_file, "r") as file:
            for line in file:
                self.filenames.append(line.strip())

    def get_filepath(self, file_name: str):
        return join(self.data_dir, file_name)

    def get_filepath_from_index(self, index: int):
        if index >= len(self.filenames):
            raise IndexError("Index out of range")
        if index < 0:
            raise IndexError("Index cannot be negative")
        return self.get_filepath(self.filenames[index] + self.extention)
