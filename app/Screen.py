import os
import inspect
import sys 
import codecs
import unicodedata

path = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
os.chdir(path+ "\\app")


from PyQt4 import QtGui, QtCore
from Controller import *



class Ocr_screen(QtGui.QTabWidget):
    def __init__(self, parent = None):

        self.res = 200
        
        
        self.window = super(Ocr_screen, self).__init__(parent)
        self.tab1 = QtGui.QWidget()
        self.tab2 = QtGui.QWidget()
        self.tab3 = QtGui.QWidget()
        
        self.addTab(self.tab1,"Train")
        self.addTab(self.tab2,"Test")
        self.addTab(self.tab3,"Evaluate")
        
        self.trainUI()
        # self.testUI()
        # self.evalUI()
        
        self.setWindowTitle("Simple Ocr")
        # self.setMinimumSize(int(5*self.res), int(4*self.res))
        
        self.controller = Controller(self)
        self.controller.bind()

        
    def trainUI(self):
        
        #Larger font
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setWeight(40)
        self.big_spacer = QtGui.QSpacerItem(50, 50)
        self.small_spacer = QtGui.QSpacerItem(20, 20)
        
      
        # Display screen 
        
        self.label_right = QtGui.QLabel("Preview :")
        self.label_right.setFont(font)
        
        self.screen = QtGui.QGraphicsView()
        self.scene = QtGui.QGraphicsScene(self.screen)
        self.screen.setScene(self.scene)
        self.screen.setGeometry(QtCore.QRect(0, 0, 3*self.res, 1*self.res))
        self.screen.setMinimumSize(3*self.res, 1*self.res)
        self.screen.setMaximumSize(3*self.res, 1*self.res)
        
        
        # Select lrate
        self.label_lrate = QtGui.QLabel("Select learning rate :")
        self.set_lrate = QtGui.QComboBox()
        self.set_lrate.addItem("0.001")
        self.set_lrate.addItem("0.0001")
        self.set_lrate.addItem("0.00001")
        self.set_lrate.addItem("0.000001")
        self.set_lrate.addItem("Adaptative")
        self.apply_lrate = QtGui.QPushButton("Apply")
        
        self.lrate_panel = QtGui.QHBoxLayout()
        self.lrate_panel.addWidget(self.set_lrate)
        self.lrate_panel.addWidget(self.apply_lrate)
        
        # Select model
        
        self.label_model = QtGui.QLabel("Select model :")
        self.set_model = QtGui.QComboBox()
        self.set_model.addItem("Bidirectionnal LSTM")
        self.apply_model = QtGui.QPushButton("Apply")
        
        self.model_panel = QtGui.QHBoxLayout()
        self.model_panel.addWidget(self.set_model)
        self.model_panel.addWidget(self.apply_model)
        
        # Select optimizer
        
        self.label_optimizer = QtGui.QLabel("Select optimizer :")
        self.set_optimizer = QtGui.QComboBox()
        self.set_optimizer.addItem("SGD")
        self.apply_optimizer = QtGui.QPushButton("Apply")
        
        self.optimizer_panel = QtGui.QHBoxLayout()
        self.optimizer_panel.addWidget(self.set_optimizer)
        self.optimizer_panel.addWidget(self.apply_optimizer)
        
        
        # Left layout title
        self.label_left = QtGui.QLabel("Select training parameters :")
        self.label_left.setFont(font)
        
        # Bottom layout
        self.start_train_button = QtGui.QPushButton("Start training")
        self.start_train_button.setFont(font)
        self.pause_train_button = QtGui.QPushButton("Pause training")
        self.pause_train_button.setFont(font)
        self.end_train_button = QtGui.QPushButton("End training")
        self.end_train_button.setFont(font)
        
        # Text stuff :
        
        self.label_gt = QtGui.QLabel("Ground truth :")
        self.ground_truth = QtGui.QLineEdit()
        self.ground_truth.setText("Bonjour je suis le baguette")
        self.gt_panel = QtGui.QHBoxLayout()
        self.gt_panel.addWidget(self.label_gt)
        self.gt_panel.addWidget(self.ground_truth)
        
        self.label_pred = QtGui.QLabel("Predicted    :")
        self.predicted = QtGui.QLineEdit()
        self.predicted.setText("Bondour je suis le faguotte")
        self.pred_panel = QtGui.QHBoxLayout()
        self.pred_panel.addWidget(self.label_pred)
        self.pred_panel.addWidget(self.predicted)
        
        self.label_loss = QtGui.QLabel("Loss :")
        self.loss = QtGui.QLineEdit()
        self.label_dist = QtGui.QLabel("Distance :")
        self.dist = QtGui.QLineEdit()
        self.loss_panel = QtGui.QHBoxLayout()
        self.loss_panel.addWidget(self.label_loss)
        self.loss_panel.addWidget(self.loss)
        self.loss_panel.addWidget(self.label_dist)
        self.loss_panel.addWidget(self.dist)
        
        
        # Left layout
        
        self.left_panel = QtGui.QVBoxLayout()

        self.left_panel.addWidget(self.label_left)

        self.left_panel.addWidget(self.label_model)
        self.left_panel.addLayout(self.model_panel)
        
        self.left_panel.addWidget(self.label_lrate)
        self.left_panel.addLayout(self.lrate_panel)
        
        self.left_panel.addWidget(self.label_optimizer)
        self.left_panel.addLayout(self.optimizer_panel)
        self.left_panel.addItem(self.small_spacer)
        
        # Right layout
        
        self.right_panel = QtGui.QVBoxLayout()
        self.right_panel.addWidget(self.label_right)
        self.right_panel.addWidget(self.screen)
        self.right_panel.setAlignment(self.screen, QtCore.Qt.AlignHCenter)
        self.right_panel.addLayout(self.gt_panel)
        self.right_panel.addLayout(self.pred_panel)
        self.right_panel.addLayout(self.loss_panel)

        
        # Main Layout
        self.main_panel = QtGui.QHBoxLayout()
        self.main_panel.addLayout(self.left_panel)
        self.main_panel.addItem(self.big_spacer)
        self.main_panel.addLayout(self.right_panel)
        
        # Bottom Layout
        
        self.bottom_panel = QtGui.QHBoxLayout()
        self.bottom_panel.addWidget(self.start_train_button)
        # self.bottom_panel.addItem(self.small_spacer)
        self.bottom_panel.addWidget(self.pause_train_button)
        # self.bottom_panel.addItem(self.small_spacer)
        self.bottom_panel.addWidget(self.end_train_button)
        


        
        # Final Layout
        self.Final_panel = QtGui.QVBoxLayout()
        self.Final_panel.addLayout(self.main_panel)
        self.Final_panel.addLayout(self.bottom_panel)
        
    
        self.tab1.setLayout(self.Final_panel)



def main():
    app = QtGui.QApplication(sys.argv)
    gui = Ocr_screen()
    gui.show()
    
    # Test
    gui.controller.preview("test.png")
    text =  codecs.open("gt.txt", 'r', encoding='utf-8').readline()
    gui.controller.set_gt(text)
    gui.controller.set_predicted(text)
    gui.controller.set_loss("3.14")
    gui.controller.set_dist("0")
    
    sys.exit(app.exec_())
    
    

if __name__ == '__main__':
    main()