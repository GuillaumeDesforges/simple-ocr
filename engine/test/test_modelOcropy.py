from unittest import TestCase

from engine.models.model_ocropy import ModelOcropy


class TestModelOcropy(TestCase):
    def setUp(self):
        self.img_height = 48
        self.model = ModelOcropy('abcdefg', self.img_height)

    def test(self):
        pass
