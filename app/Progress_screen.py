from PyQt4 import QtGui, QtCore
import sys
import os
import numpy as np


class ProgressWindow():
    def __init__(self, controller):

        self.controller = controller 
        self.window = QtGui.QWidget()
        self.window.setMinimumSize(600,100)
        self.open = True

        #Bottom of screen buttons
        self.cancel_button = QtGui.QPushButton('Cancel')
        self.cancel_button.setMinimumSize(200,10)
        self.progress = QtGui.QProgressBar()
        self.progress.setAlignment(QtCore.Qt.AlignHCenter)
        self.progress.setRange(0,0)
        self.progress.setMinimumSize(500,10)
        
        
        
        self.main_panel = QtGui.QVBoxLayout()
        self.main_panel.addWidget(self.progress)
        self.main_panel.setAlignment(self.progress, QtCore.Qt.AlignHCenter)
        self.main_panel.addWidget(self.cancel_button)
        self.main_panel.setAlignment(self.cancel_button, QtCore.Qt.AlignHCenter)

        self.window.setLayout(self.main_panel)
        self.window.setWindowTitle('Waiting for end of epoch')
        self.window.show()
        
        #Binding
        self.cancel_button.clicked.connect(self.cancel)
        
        if controller.epoch_end : 
            self.open = False
            self.window.close()
    
    
    def cancel(self):
        self.controller.pause = False
        self.controller.pause = False
        self.open = False
        self.window.close()