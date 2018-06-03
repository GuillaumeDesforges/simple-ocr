import keras


def ctc_loss_func(args):
    y_pred, y_true, input_x_width, input_y_width = args
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_x_width, input_y_width)


def ctc_decode_func(args):
    y_pred, input_x_widths = args
    flattened_input_x_width = keras.backend.reshape(input_x_widths, (-1,))
    top_k_decoded, _ = keras.backend.ctc_decode(y_pred, flattened_input_x_width)
    return top_k_decoded[0]


class ModelOcropy(keras.Model):
    def __init__(self, alphabet: str, img_height):
        self.img_height = img_height
        self.lstm_size = 500
        self.alphabet_size = len(alphabet)

        # check backend input shape (channel first/last)
        if keras.backend.image_data_format() == "channels_first":
            input_shape = (1, None, self.img_height)
        else:
            input_shape = (None, self.img_height, 1)

        # data input
        input_x = keras.layers.Input(input_shape, name='x')

        # training inputs
        input_y = keras.layers.Input((None,), name='y')
        input_x_widths = keras.layers.Input([1], name='x_widths')
        input_y_widths = keras.layers.Input([1], name='y_widths')

        # network
        flattened_input_x = keras.layers.Reshape((-1, self.img_height))(input_x)
        lstm1 = keras.layers.CuDNNLSTM(self.lstm_size, return_sequences=True, name='lstm1')(flattened_input_x)
        lstm2 = keras.layers.CuDNNLSTM(self.lstm_size, return_sequences=True, name='lstm2', go_backwards=True)(
            flattened_input_x)
        lstm_out = keras.layers.Concatenate(axis=-1, name='lstm_out')([lstm1, lstm2])
        dense = keras.layers.Dense(self.alphabet_size + 1, activation='relu')(lstm_out)
        y_pred = keras.layers.Softmax(name='y_pred')(dense)

        # ctc loss
        ctc = keras.layers.Lambda(ctc_loss_func, output_shape=[1], name='ctc')(
            [dense, input_y, input_x_widths, input_y_widths]
        )
        decode = keras.layers.Lambda(ctc_decode_func, output_shape=[None], name='decode')(
            [y_pred, input_x_widths]
        )

        # init keras model
        super().__init__(inputs=[input_x, input_x_widths, input_y, input_y_widths], outputs=[decode, ctc])

        # ctc decoder
        # decoded_sequences = self.decoder([input_x, flattened_input_x_width])
