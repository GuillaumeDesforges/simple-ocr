import os
from time import strftime, gmtime

import editdistance
import keras
import numpy as np
from PyQt4 import QtCore

from app.Msg_screen import *
from app.Progress_screen import ProgressWindow
from engine.callbacks.gui import GUICallback
from engine.data.generators.batch_generator_manuscript import BatchGeneratorManuscript
from engine.models.model_ocropy import ModelOcropy
from engine.trainer import Trainer


class Controller:
    def __init__(self, ui):
        #to link with the screen
        
        self.ui = ui
        
        self.lrate = 0.001
        self.model = "Bidirectionnal LSTM"
        self.optimizer = "SGD"
        self.network_names = ["Network #1", "Network #2"]
        self.network_name = self.network_names[0]
        self.page_names = ["Bodmer - p1", "Bodmer - p2"]
        self.page_name = self.page_names[0]
        self.nb_epoch = 1

        self.models = {}
    
    ## Train Window ##

    def bind_train(self):
        self.ui.apply_lrate.clicked.connect(self.set_lrate)
        self.ui.apply_optimizer.clicked.connect(self.set_optimizer)
        self.ui.apply_model.clicked.connect(self.set_model)
        self.ui.apply_epoch.clicked.connect(self.set_epoch)

        self.ui.start_train_button.clicked.connect(self.start_train)

        
    
    def start_train(self):
        self.progress()

        # data generators
        data_path = '/home/arsleust/projects/simple-ocr/data/bodmer'
        img_height = 48
        train_data_generator = BatchGeneratorManuscript(data_path,
                                                        img_height=img_height)
        test_data_generator = BatchGeneratorManuscript(data_path,
                                                       img_height=img_height,
                                                       sample_size=10,
                                                       alphabet=train_data_generator.alphabet)

        # model
        self.model = ModelOcropy(train_data_generator.alphabet, img_height)
        print(self.model.summary())

        # callbacks
        str_date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        callbacks = []
        if True:
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints")
            checkpoints_path = os.path.join("checkpoints", str_date_time + '.hdf5')
            callback_checkpoint = keras.callbacks.ModelCheckpoint(checkpoints_path, monitor='val_loss', verbose=1,
                                                                  save_best_only=True, save_weights_only=True)
            callbacks.append(callback_checkpoint)
        if True:
            callback_gui = GUICallback(test_data_generator, self)
            callbacks.append(callback_gui)

        # trainer
        trainer = Trainer(
            self.model,
            train_data_generator,
            test_data_generator,
            lr=self.lrate,
            epochs=self.nb_epoch,
            steps_per_epochs=20,
            callbacks=callbacks)

        trainer.train()
        print("Training done")

        self.end_train()

        
    def end_train(self):
        print("End train")
        # Lets the user chose the network name
        self.savewindow = SaveWindow(self)

        self.ui.apply_lrate.setEnabled(True)
        self.ui.apply_model.setEnabled(True)
        self.ui.start_train_button.setEnabled(True)
        self.ui.apply_optimizer.setEnabled(True)

    def save_model(self, name):
        if not name == '':
            self.network_names.append(name)
            self.models[name] = self.model
            print("Network registered as : " + name)


    def progress(self):
        window = ProgressWindow(self)


    def set_lrate(self):
        if self.ui.set_lrate.currentText() == "Adaptative" :
            # Todo
            pass
        else : 
            self.lrate=float(self.ui.set_lrate.currentText())
        
        
    def set_epoch(self):
        self.nb_epoch = int(self.ui.set_epoch.currentText())


    def set_model(self):
        self.model = self.ui.set_model.currentText()
        
        
    def set_optimizer(self):
        self.optimizer = self.ui.set_optimizer.currentText()
        
        
    def set_gt(self, text):
        self.ui.ground_truth.setText(text)
    
    
    def set_predicted(self, text):
        self.ui.predicted.setText(text)


    def set_loss(self, text):
        self.ui.loss.setText(text)

    
    def set_dist(self, text):
        self.ui.dist.setText(text)
    

    def preview(self, picture):
        qimg = QtGui.QImage(picture)
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(int(2.9*self.ui.res), self.ui.res, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.ui.scene.clear()
        item = self.ui.scene.addPixmap(pixmap)
        self.ui.screen.setScene(self.ui.scene)
    


    ## Test Window ##
    
    def bind_test(self):
        self.ui.apply_network.clicked.connect(self.set_network)
        self.ui.start_test_button.clicked.connect(self.start_test)


    def start_test(self):
        print('Testing Network "' + self.network_name + '"')

        data_path = '/home/arsleust/projects/simple-ocr/data/bodmer'
        img_height = 48
        test_data_generator = BatchGeneratorManuscript(data_path,
                                                       img_height=img_height)
        i = np.random.randint(0, len(test_data_generator))

        data = test_data_generator[i]
        x_dict, _ = data
        x, x_widths, y_true, y_true_widths = map(lambda k: x_dict[k][0], ['x', 'x_widths', 'y', 'y_widths'])
        y_pred = self.model.predict(x_dict)

        list_decoded = y_pred[0][0].tolist()
        list_true = y_true.tolist()
        text_decoded = ''.join([test_data_generator.alphabet[l] for l in list_decoded])
        text_true = ''.join([test_data_generator.alphabet[l] for l in list_true])

        sample_dist = editdistance.eval(text_decoded, text_true)

        self.set_dist_test(str(sample_dist))
        self.set_gt_test(str(text_true))
        self.set_predicted_test(str(text_decoded))
        line_img_path = test_data_generator.lines[i][0]
        self.preview_test(line_img_path)

        self.end_test()

    def end_test(self):
        pass

    def set_network(self):
        self.network_name = self.ui.select_network.currentText()
        if self.network_name in self.models.keys():
            self.model = self.models[self.network_name]
        else:
            raise Exception("No model found", self.network_name)

    def set_gt_test(self, text):
        self.ui.ground_truth2.setText(text)

    def set_predicted_test(self, text):
        self.ui.predicted2.setText(text)

    def set_dist_test(self, text):
        self.ui.dist2.setText(text)

    def preview_test(self, picture):
        qimg = QtGui.QImage(picture)
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(int(2.9 * self.ui.res), self.ui.res, QtCore.Qt.KeepAspectRatio,
                                                      QtCore.Qt.SmoothTransformation)
        self.ui.scene2.clear()
        item = self.ui.scene2.addPixmap(pixmap)
        self.ui.screen2.setScene(self.ui.scene2)

    ## Validation window
    
    def bind_valid(self):
        self.ui.apply_network2.clicked.connect(self.set_network_eval)
        self.ui.apply_page.clicked.connect(self.set_page)
        self.ui.pause_eval_button.clicked.connect(self.pause_eval)
        self.ui.start_eval_button.clicked.connect(self.start_eval)
        self.ui.end_eval_button.clicked.connect(self.end_eval)
        
        self.ui.pause_eval_button.setEnabled(False)
        self.ui.end_eval_button.setEnabled(False)
        
    
    def start_eval(self):
        # TODO

        print('Translating page  "' + self.page_name + '", with network  "' + self.network_name + '"')
        
        self.ui.start_eval_button.setEnabled(False)
        self.ui.pause_eval_button.setEnabled(True)
        self.ui.end_eval_button.setEnabled(True)
        self.ui.apply_page.setEnabled(False)
        self.ui.apply_network2.setEnabled(False)
    
    
    def pause_eval(self):
        # TODO
        


        self.ui.start_eval_button.setEnabled(True)
        self.ui.pause_eval_button.setEnabled(False)

        
    def end_eval(self):
        # TODO
        


        self.ui.start_eval_button.setEnabled(True)
        self.ui.pause_eval_button.setEnabled(False)
        self.ui.end_eval_button.setEnabled(False)
        self.ui.apply_page.setEnabled(True)
        self.ui.apply_network2.setEnabled(True)
    


    def set_network_eval(self):
        self.network_name = self.ui.select_network2.currentText()
    
    
    def set_page(self):
        self.page_name = self.ui.select_page.currentText()
    
    
    def set_predicted_eval(self, text):
        self.ui.predicted3.setText(text)

