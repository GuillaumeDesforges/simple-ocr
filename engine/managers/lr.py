import abc


class BaseLearningRateManager(abc.ABC):
    @abc.abstractmethod
    def update_lr(self, *args):
        pass

    @abc.abstractmethod
    def get_lr(self):
        return 0


class ConstantLearningRateManager(BaseLearningRateManager):
    def __init__(self, lr):
        self.lr = lr

    def update_lr(self, *args):
        pass

    def get_lr(self):
        return self.lr
