import os

import keras
from PIL import Image


class BatchGeneratorManuscript(keras.utils.Sequence):
    def __init__(self, path_to_data):
        super().__init__()

        self.path_to_data = path_to_data

        # walk data set directory to find images and labels
        x_paths = []
        y_paths = []
        for root, dirs, files in os.walk(self.path_to_data):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = file_path.split('.')[-1]
                if file_ext is 'png':
                    x_paths.append(file_path)
                if file_ext is 'txt':
                    y_paths.append(file_path)

        # keep only files if there is both data and label
        stripped_x_paths = set([x_path.strip('.png') for x_path in x_paths])
        stripped_y_paths = set([y_path.strip('.png') for y_path in y_paths])
        stripped_paths = stripped_x_paths & stripped_y_paths

        # save paths (file_path_x, file_path_y)
        self.paths = [(stripped_path + '.png', stripped_path + '.txt') for stripped_path in stripped_paths]

        # TODO build alphabet
        self.alphabet = ''

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x_file_path, y_file_path = self.paths[idx]
        x = Image.open(x_file_path).convert('L')
        x_width = x.shape[0]

        with open(y_file_path) as f:
            y_text = f.readline()
        y = [self.alphabet.find(c) for c in y_text]
        y_width = len(y)

        return {'x': x, 'x_width': x_width, 'y': y, 'y_width': y_width}
