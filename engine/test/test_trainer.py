from unittest import TestCase

from engine.data.generators.batch_generator_manuscript import BatchGeneratorManuscript
from engine.managers.lr import ConstantLearningRateManager
from engine.models.model_ocropy import ModelOcropy
from engine.trainer import Trainer


class TestTrainer(TestCase):
    def setUp(self):
        self.img_height = 48
        self.train_batch_generator = BatchGeneratorManuscript('../../fixtures/manuscript/', self.img_height)
        self.test_batch_generator = BatchGeneratorManuscript('../../fixtures/manuscript/', self.img_height)
        self.alphabet = self.test_batch_generator.alphabet
        self.model = ModelOcropy(self.alphabet, self.img_height)
        self.lr_manager = ConstantLearningRateManager(lr=0.01)
        self.epochs = 5
        self.trainer = Trainer(self.model,
                               self.train_batch_generator,
                               self.test_batch_generator,
                               self.lr_manager,
                               epochs=self.epochs)

    def test_train(self):
        self.trainer.train()
