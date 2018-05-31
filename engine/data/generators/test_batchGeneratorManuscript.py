from unittest import TestCase

import numpy as np

from engine.data.generators.batch_generator_manuscript import BatchGeneratorManuscript


class TestBatchGeneratorManuscript(TestCase):
    def setUp(self):
        self.generator = BatchGeneratorManuscript('fixtures/manuscript/')

    def test_len(self):
        self.assertEqual(len(self.generator) == 7)

    def test_getitem(self):
        item: dict = self.generator.__getitem__(0)
        keys = item.keys()

        self.assertIn('x' in keys)
        self.assertIn('y' in keys)
        self.assertIn('x_widths' in keys)
        self.assertIn('y_widths' in keys)

        x, x_widths, y, y_widths = item['x'], item['x_widths'], item['y'], item['y_widths']
        self.assertIs(type(x), np.ndarray)
        self.assertIs(type(y), np.ndarray)
        self.assertIs(type(x_widths), int)
        self.assertIs(type(y_widths), int)
