from unittest import TestCase

import numpy as np

from engine.data.generators.batch_generator_iam_handwriting import BatchGeneratorIAMHandwriting


class TestBatchGeneratorManuscript(TestCase):
    def setUp(self):
        self.generator = BatchGeneratorIAMHandwriting('../../fixtures/manuscript/')

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
        self.assertIs(type(x_widths), int)
        self.assertIs(type(y_widths), int)
