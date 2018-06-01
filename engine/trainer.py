import keras

from engine.managers.lr import BaseLearningRateManager


class Trainer:
    def __init__(self,
                 model: keras.Model,
                 train_batch_generator: keras.utils.Sequence,
                 test_batch_generator: keras.utils.Sequence,
                 lr_manager: BaseLearningRateManager,
                 initial_epoch: int = 0,
                 epochs: int = 1):
        self.model = model
        self.train_batch_generator = train_batch_generator
        self.test_batch_generator = test_batch_generator
        self.lr_manager = lr_manager
        self.initial_epoch = initial_epoch
        self.epochs = epochs

    def train(self):
        lr = self.lr_manager.get_lr()
        optimizer = keras.optimizers.Adam(lr, clipnorm=5)
        self.model.compile(optimizer=optimizer, loss={'ctc_loss': lambda _, loss: loss})

        self.model.fit_generator(self.train_batch_generator,
                                 validation_data=self.test_batch_generator,
                                 use_multiprocessing=True,
                                 workers=2,
                                 initial_epoch=self.initial_epoch,
                                 epochs=self.epochs)
