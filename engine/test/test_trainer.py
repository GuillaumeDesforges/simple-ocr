from unittest import TestCase

from engine.data.generators.batch_generator_manuscript import BatchGeneratorManuscript
from engine.managers.lr import ConstantLearningRateManager
from engine.models.model_ocropy import ModelOcropy
from engine.trainer import Trainer


class TestTrainer(TestCase):
    def setUp(self):
        self.alphabet = 'abcdefg'
        self.model = ModelOcropy(self.alphabet)
        self.train_batch_generator = BatchGeneratorManuscript('../../fixtures/manuscript/')
        self.test_batch_generator = BatchGeneratorManuscript('../../fixtures/manuscript/')
        self.lr_manager = ConstantLearningRateManager(lr=0.01)
        self.trainer = Trainer(self.model, self.train_batch_generator, self.test_batch_generator, self.lr_manager)

    def test_train(self):
        self.trainer.train()
