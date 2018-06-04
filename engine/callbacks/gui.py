import editdistance
import keras
import numpy as np
from PyQt4.QtGui import QApplication


class GUICallback(keras.callbacks.Callback):
    def __init__(self, validation_data_generator: keras.utils.Sequence, gui, random: bool = False):
        self.validation_data_generator = validation_data_generator
        self.gui = gui
        self.random = random
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if not self.random:
            i = 0
        else:
            i = np.random.randint(0, len(self.validation_data_generator))

        data = self.validation_data_generator[i]
        x_dict, _ = data
        x, x_widths, y_true, y_true_widths = map(lambda k: x_dict[k][0], ['x', 'x_widths', 'y', 'y_widths'])
        y_pred = self.model.predict(x_dict)

        loss = y_pred[2][0][0]
        list_decoded = y_pred[0][0].tolist()
        list_true = y_true.tolist()
        text_decoded = ''.join([self.validation_data_generator.alphabet[l] for l in list_decoded])
        text_true = ''.join([self.validation_data_generator.alphabet[l] for l in list_true])

        sample_dist = editdistance.eval(text_decoded, text_true) / y_true_widths

        self.gui.set_dist(str(sample_dist))
        self.gui.set_loss(str(loss))
        self.gui.set_gt(str(text_true))
        self.gui.set_predicted(str(text_decoded))
        line_img_path = self.validation_data_generator.lines[i][0]
        self.gui.preview(line_img_path)

        QApplication.processEvents()
        self.gui.ui.update()
