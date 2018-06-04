import keras


class Trainer:
    def __init__(self, model: keras.Model, train_batch_generator: keras.utils.Sequence,
                 test_batch_generator: keras.utils.Sequence, multi_process_batch_generation: bool = True,
                 initial_epoch: int = 0, epochs: int = 1, lr: float = 0.01, callbacks: list = None,
                 steps_per_epochs=None):
        self.model = model
        self.train_batch_generator = train_batch_generator
        self.test_batch_generator = test_batch_generator
        self.multi_process_batch_generation = multi_process_batch_generation
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.steps_per_epochs = steps_per_epochs
        self.lr = lr
        self.callbacks = callbacks

    def train(self):
        optimizer = keras.optimizers.Adam(lr=self.lr, clipnorm=5)
        losses = {
            'ctc': lambda _, loss: loss,
        }
        self.model.compile(optimizer=optimizer, loss=losses)

        train_params = {
            'generator': self.train_batch_generator,
            'validation_data': self.test_batch_generator,
            'max_queue_size': 10,
            'initial_epoch': self.initial_epoch,
            'epochs': self.epochs,
            'callbacks': self.callbacks
        }
        if self.multi_process_batch_generation:
            multi_process_params = {
                'use_multiprocessing': True,
                'workers': 4,
            }
            train_params.update(multi_process_params)
        if self.steps_per_epochs is not None:
            steps_per_epochs_params = {
                'steps_per_epoch': self.steps_per_epochs
            }
            train_params.update(steps_per_epochs_params)

        self.model.fit_generator(**train_params)
