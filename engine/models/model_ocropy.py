import keras


def ctc_lambda_func(args):
    y_pred, y_true, input_x_width, input_y_width = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_x_width, input_y_width)


class ModelOcropy(keras.Model):
    def __init__(self, alphabet: str):
        self.img_height = 48
        self.lstm_size = 100
        self.alphabet_size = len(alphabet)

        # check backend input shape (channel first/last)
        if keras.backend.image_data_format() == "channels_first":
            input_shape = (1, None, self.img_height)
        else:
            input_shape = (None, self.img_height, 1)

        # data input
        input_x = keras.layers.Input(input_shape, name='X')

        # training inputs
        input_y = keras.layers.Input((None,), name='Y')
        input_x_widths = keras.layers.Input([1], name='X_widths')
        input_y_widths = keras.layers.Input([1], name='Y_widths')

        # network
        lstm = keras.layers.LSTM(self.lstm_size, return_sequences=True, name='lstm')
        bidirectional_lstm = keras.layers.Bidirectional(lstm, name='bidirectional lstm')(input_x)
        dense = keras.layers.Dense(self.alphabet_size, activation='relu')(bidirectional_lstm)

        # ctc loss
        ctc = keras.layers.Lambda(ctc_lambda_func, output_shape=[1], name='ctc')(
            [dense, input_y, input_x_widths, input_y_widths])

        super().__init__(inputs=[input_x, input_x_widths, input_y, input_y_widths], outputs=[dense, ctc])
