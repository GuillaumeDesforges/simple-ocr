from PyQt4 import QtGui, QtCore
import sys
import os
import numpy as np


class ProgressWindow():
    def __init__(self, controller):

        self.controller = controller 
        self.window = QtGui.QWidget()
        self.window.setMinimumSize(600,100)

        self.progress = QtGui.QProgressBar()
        self.progress.setAlignment(QtCore.Qt.AlignHCenter)
        self.progress.setRange(0,0)
        self.progress.setMinimumSize(500,10)
        
        self.main_panel = QtGui.QVBoxLayout()
        self.main_panel.addWidget(self.progress)
        self.main_panel.setAlignment(self.progress, QtCore.Qt.AlignHCenter)

        self.window.setLayout(self.main_panel)
        self.window.setWindowTitle('Training ...')
        self.window.show()
        
