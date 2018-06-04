import os
import inspect
path = os.path.dirname(inspect.getfile(inspect.currentframe()))
os.chdir(path)

from Screen import *
from Msg_screen import *
from Progress_screen import *

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
        self.nb_epoch = 100
    
    
    ## Train Window ##

    def bind_train(self):
        self.ui.apply_lrate.clicked.connect(self.set_lrate)
        self.ui.apply_optimizer.clicked.connect(self.set_optimizer)
        self.ui.apply_model.clicked.connect(self.set_model)
        
        self.ui.start_train_button.clicked.connect(self.start_train)

        
    
    def start_train(self):
        # TODO
        self.progress()
        
        # self.ui.start_train_button.setEnabled(False)
        # self.ui.apply_lrate.setEnabled(False)
        # self.ui.apply_model.setEnabled(False)
        # self.ui.apply_optimizer.setEnabled(False)
        # self.ui.apply_epoch.setEnabled(False)

        
    def end_train(self):
        # TODO
        
        
        # Lets the user chose the network name
        app = QtGui.QApplication(sys.argv)
        window = SaveWindow(self)
        app.exec_()


        self.ui.apply_lrate.setEnabled(True)
        self.ui.apply_model.setEnabled(True)
        self.ui.start_train_button.setEnabled(True)
        self.ui.apply_optimizer.setEnabled(True)
    
    
    def progress(self):     
        app = QtGui.QApplication(sys.argv)
        window = ProgressWindow(self)
        app.exec_()


    def set_lrate(self):
        if self.ui.set_lrate.currentText() == "Adaptative" :
            # Todo
            pass
        else : 
            self.lrate=float(self.ui.set_lrate.currentText())
        
        
    def set_epoch(self):
        self.nb_epoch=float(self.ui.set_epoch.currentText())
        
        
    def set_model(self):
        self.model=self.ui.set_model.currentText()
        
        
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
        print("Displaying line")
        qimg = QtGui.QImage(picture)
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(int(2.9*self.ui.res), self.ui.res, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        item = self.ui.scene.addPixmap(pixmap)
        self.ui.screen.setScene(self.ui.scene)
    

    
    ## Test Window ##
    
    def bind_test(self):
        self.ui.apply_network.clicked.connect(self.set_network)
        self.ui.pause_test_button.clicked.connect(self.pause_test)
        self.ui.start_test_button.clicked.connect(self.start_test)
        self.ui.end_test_button.clicked.connect(self.end_test)
        
        self.ui.pause_test_button.setEnabled(False)
        self.ui.end_test_button.setEnabled(False)
        
    
    def start_test(self):
        # TODO
        
        print('Testing Network "' + self.network_name + '"')
        
        self.ui.start_test_button.setEnabled(False)
        self.ui.pause_test_button.setEnabled(True)
        self.ui.end_test_button.setEnabled(True)
        self.ui.apply_network.setEnabled(False)
    
    
    def pause_test(self):
        # TODO
        
        self.ui.start_test_button.setEnabled(True)
        self.ui.pause_test_button.setEnabled(False)

        
    def end_test(self):
        # TODO
        
        self.ui.start_test_button.setEnabled(True)
        self.ui.pause_test_button.setEnabled(False)
        self.ui.end_test_button.setEnabled(False)
        self.ui.apply_network.setEnabled(True)
    
    
    def set_network(self):
        self.network_name = self.ui.select_network.currentText()
    
    
    def set_gt_test(self, text):
        self.ui.ground_truth2.setText(text)
    
    
    def set_predicted_test(self, text):
        self.ui.predicted2.setText(text)

    
    def set_dist_test(self, text):
        self.ui.dist2.setText(text)
    
    
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

    