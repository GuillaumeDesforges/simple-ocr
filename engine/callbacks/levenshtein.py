import editdistance
import keras
import numpy as np


class LevenshteinCallback(keras.callbacks.Callback):
    def __init__(self, validation_data_generator: keras.utils.Sequence, size: int = 5, random: bool = False):
        self.validation_data_generator = validation_data_generator
        self.size = size
        self.random = random
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        print("Computing Levenshtein for validation set :")

        if not self.random:
            indexes = list(range(self.size))
        else:
            indexes = np.arange(len(self.validation_data_generator))
            np.random.choice(indexes, size=self.size)

        sum_dist = 0
        for i in indexes:
            data = self.validation_data_generator[i]
            x_dict, _ = data
            x, x_widths, y_true, y_true_widths = map(lambda k: x_dict[k][0], ['x', 'x_widths', 'y', 'y_widths'])
            y_pred = self.model.predict(x_dict)

            list_decoded = y_pred[0][0].tolist()
            list_true = y_true.tolist()
            text_decoded = ''.join([self.validation_data_generator.alphabet[l] for l in list_decoded])
            text_true = ''.join([self.validation_data_generator.alphabet[l] for l in list_true])

            sample_dist = editdistance.eval(text_decoded, text_true) / y_true_widths
            sum_dist += sample_dist
            print('{:3d} DECO'.format(i), text_decoded)
            print('    TRUE', text_true)
            print('    DIST', sample_dist, ' X_WIDTH ', x_widths, ' Y_WIDTH ', y_true_widths)

        mean_dist = sum_dist / self.size

        print("Mean Levenshtein dist: {}".format(mean_dist))
