from unittest import TestCase

from engine.models.model_ocropy import ModelOcropy


class TestModelOcropy(TestCase):
    def setUp(self):
        self.model = ModelOcropy('abcdefg')

    def test(self):
        pass
