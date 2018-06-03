from PyQt4 import QtGui, QtCore
import sys
import os
import numpy as np


class SaveWindow():
    def __init__(self, controller):

        self.controller = controller 
        self.window = QtGui.QWidget()

        #Bottom of screen buttons
        self.save_button = QtGui.QPushButton('Save')
        self.cancel_button = QtGui.QPushButton('Cancel')
        
        self.label_save = QtGui.QLabel("Save network as :")
        self.save_box = QtGui.QLineEdit()
        
        #Lower panel
        self.lower_panel = QtGui.QHBoxLayout()
        self.lower_panel.addWidget(self.save_button)
        self.lower_panel.addWidget(self.cancel_button)
        
        self.main_panel = QtGui.QVBoxLayout()
        self.main_panel.addWidget(self.label_save)
        self.main_panel.addWidget(self.save_box)
        self.main_panel.addLayout(self.lower_panel)
        
        self.window.setLayout(self.main_panel)
        self.window.setWindowTitle('Save')
        self.window.show()
        
        #Binding
        self.save_button.clicked.connect(self.save)
        self.cancel_button.clicked.connect(self.cancel)
        
    
    def cancel(self):
        self.window.close()


    def save(self):
        name = self.save_box.text()
        if not name == '':
            self.controller.network_names.append(name)
            #TODO : Saving the model as <name> somewhere
            print("Network saved as : " + name)
            self.window.close()
            self.controller.ui.select_network.addItem(name)
            self.controller.ui.select_network2.addItem(name)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = SaveWindow()
    app.exec_()