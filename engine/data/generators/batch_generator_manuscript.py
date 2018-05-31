import os
from xml.etree import ElementTree

import keras
import numpy as np
from PIL import Image


class BatchGeneratorManuscript(keras.utils.Sequence):
    def __init__(self, path_to_data: str):
        super().__init__()

        self.path_to_data = path_to_data
        self.path_to_meta_data = os.path.join(path_to_data, 'meta')
        self.path_to_img_data = os.path.join(path_to_data, 'img')

        # walk data set directory to find meta data files
        meta_file_paths = []
        for root, dirs, files in os.walk(self.path_to_meta_data):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = file_path.split('.')[-1]
                if file_ext == 'xml':
                    meta_file_paths.append(file_path)

        self.lines = []
        # read meta files and extract lines information
        for meta_file_path in meta_file_paths:
            meta_data = self.extract_meta_data(meta_file_path)
            # each metadata file corresponds to a sample that has some lines
            sample_lines = meta_data['lines']
            for sample_line in sample_lines:
                sample_line_path = sample_line['path']
                # check if img files exists
                img_file_exists = os.path.isfile(sample_line_path)
                if img_file_exists:
                    sample_line_text = sample_line['text']
                    # definition of the data for a line, important !
                    line = (sample_line_path, sample_line_text)
                    self.lines.append(line)

        # TODO build alphabet
        self.alphabet = ''

        # NOTE convert line text to encoded int array label ?

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]

        # line image data
        img_file_path = line[0]
        img = Image.open(img_file_path).convert('L')
        x = np.array(img)
        x_width = x.shape[0]

        # line text data
        y_text = line[1]
        y = np.array([self.alphabet.find(c) for c in y_text])
        y_width = len(y)

        return {'x': x, 'x_widths': x_width, 'y': y, 'y_widths': y_width}

    def extract_meta_data(self, meta_file_path: str):
        # XML element tree
        xml_tree = ElementTree.parse(meta_file_path)
        xml_root = xml_tree.getroot()
        meta_data = {}
        sample_id = xml_root.attrib['id']
        sample_lines = []
        for xml_line in xml_root[1]:
            line_id = xml_line.attrib['id']
            line_path = os.path.join(self.path_to_img_data, sample_id, line_id + '.png')
            line_text = xml_line.attrib['text']
            line = {'path': line_path, 'text': line_text}
            sample_lines.append(line)

        meta_data['id'] = sample_id
        meta_data['lines'] = sample_lines
        return meta_data
