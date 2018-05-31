from unittest import TestCase

from engine.managers.lr import ConstantLearningRateManager


class TestConstantLearningRateManager(TestCase):
    def setUp(self):
        self.lr = 0.1
        self.manager = ConstantLearningRateManager(self.lr)

    def test_update_lr(self):
        self.manager.update_lr()
        self.assertEqual(self.manager.lr, self.lr)

    def test_get_lr(self):
        self.assertEqual(self.manager.get_lr(), self.lr)
