from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from engine.data.generators.batch_generator_iam_handwriting import BatchGeneratorIAMHandwriting


class TestBatchGeneratorIAM(TestCase):
    def setUp(self):
        self.img_height = 48
        self.generator = BatchGeneratorIAMHandwriting('fixtures/iam_handwriting/', self.img_height)

    def test_len(self):
        self.assertEqual(len(self.generator), 7)

    def test_getitem(self):
        item = self.generator.__getitem__(0)
        x_dict, y = item
        keys = x_dict.keys()

        self.assertIn('x', keys)
        self.assertIn('y', keys)
        self.assertIn('x_widths', keys)
        self.assertIn('y_widths', keys)

        x, x_widths, y, y_widths = x_dict['x'], x_dict['x_widths'], x_dict['y'], x_dict['y_widths']
        self.assertIs(type(x), np.ndarray)
        self.assertIs(type(y), np.ndarray)
        self.assertIs(type(x_widths), np.ndarray)
        self.assertIs(type(y_widths), np.ndarray)

        self.assertEqual(len(x.shape), 4)
        self.assertEqual(len(y.shape), 2)
        self.assertEqual(len(x_widths.shape), 1)
        self.assertEqual(len(y_widths.shape), 1)

        plt.title(y)
        plt.imshow(x[0, :, :, 0].T)
        plt.show(block=True)
