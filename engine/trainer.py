import keras


class Trainer:
    def __init__(self,
                 model: keras.Model,
                 train_batch_generator: keras.utils.Sequence,
                 test_batch_generator: keras.utils.Sequence,
                 initial_epoch: int = 0,
                 epochs: int = 1,
                 base_lr: int = 0.01,
                 callbacks=None):
        self.model = model
        self.train_batch_generator = train_batch_generator
        self.test_batch_generator = test_batch_generator
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.base_lr = base_lr
        self.callbacks = callbacks

    def train(self):
        lr = self.base_lr
        optimizer = keras.optimizers.Adam(lr, clipnorm=5)
        self.model.compile(optimizer=optimizer, loss={'ctc': lambda _, loss: loss})

        self.model.fit_generator(self.train_batch_generator,
                                 validation_data=self.test_batch_generator,
                                 use_multiprocessing=True,
                                 workers=4,
                                 max_queue_size=10,
                                 initial_epoch=self.initial_epoch,
                                 epochs=self.epochs,
                                 callbacks=self.callbacks)
