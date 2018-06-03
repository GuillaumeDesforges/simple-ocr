import editdistance
import keras


class LevenshteinCallback(keras.callbacks.Callback):
    def __init__(self, validation_data_generator: keras.utils.Sequence, size: int = 5):
        self.validation_data_generator = validation_data_generator
        self.size = size
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        sum_dist = 0
        print("Computing Levenshtein for validation set :")
        for i in range(self.size):
            data = self.validation_data_generator[i]
            x_dict, _ = data
            x, x_widths, y_true, y_true_widths = map(lambda k: x_dict[k][0], ['x', 'x_widths', 'y', 'y_widths'])
            y_pred = self.model.predict(x_dict)
            list_pred = y_pred[0][0].tolist()
            text_pred = ''.join([self.validation_data_generator.alphabet[l] for l in list_pred])
            list_true = y_true.tolist()
            text_true = ''.join([self.validation_data_generator.alphabet[l] for l in list_true])
            print('{:3d} PRED '.format(i), text_pred)
            print('    TRUE', text_true)
            sum_dist += editdistance.eval(list_pred, list_true) / y_true_widths

        mean_dist = sum_dist / len(self.validation_data_generator)

        print("Levenshtein loss : {}".format(mean_dist))
