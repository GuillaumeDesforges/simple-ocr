import os
import inspect
path = os.path.dirname(inspect.getfile(inspect.currentframe()))
os.chdir(path)

from Screen import *

class Controller:
    def __init__(self, ui):
        
        #to link with the screen
        
        self.ui = ui
        
        self.lrate = 0.001
        self.model = "Bidirectionnal LSTM"
        self.optimizer = "SGD"
    

    def bind(self):     #Assigns each widget to the correspinding function
        self.ui.apply_lrate.clicked.connect(self.set_lrate)
        self.ui.apply_optimizer.clicked.connect(self.set_optimizer)
        self.ui.apply_model.clicked.connect(self.set_model)
        self.ui.pause_train_button.clicked.connect(self.pause_train)
        self.ui.start_train_button.clicked.connect(self.start_train)
        self.ui.end_train_button.clicked.connect(self.end_train)
        

        
        self.ui.pause_train_button.setEnabled(False)
        self.ui.end_train_button.setEnabled(False)
    
    def start_train(self):
        # TODO
        self.ui.start_train_button.setEnabled(False)
        self.ui.pause_train_button.setEnabled(True)
        self.ui.end_train_button.setEnabled(True)
        self.ui.apply_lrate.setEnabled(False)
        self.ui.apply_model.setEnabled(False)
        self.ui.apply_optimizer.setEnabled(False)
    
    def pause_train(self):
        # TODO
        
        self.ui.start_train_button.setEnabled(True)
        self.ui.pause_train_button.setEnabled(False)
        self.ui.apply_lrate.setEnabled(True)
        self.ui.apply_model.setEnabled(True)
        self.ui.apply_optimizer.setEnabled(True)
        
    def end_train(self):
        # TODO
        
        self.ui.pause_train_button.setEnabled(False)
        self.ui.end_train_button.setEnabled(False)
        self.ui.apply_lrate.setEnabled(True)
        self.ui.apply_model.setEnabled(True)
        self.ui.start_train_button.setEnabled(True)
        self.ui.apply_optimizer.setEnabled(True)
    
    def set_lrate(self):
        self.lrate=float(self.ui.set_lrate.currentText())
        
        
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
    
    
    
    
    