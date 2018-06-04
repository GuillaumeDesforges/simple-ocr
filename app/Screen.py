import codecs
import inspect
import os

path = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
os.chdir(path)

from app.Controller import *


class Ocr_screen(QtGui.QTabWidget):
    def __init__(self, appli, parent = None, ):

        self.res = 200
        self.appli = appli
        self.parent = parent
        
        self.create_window()
        
        self.controller.bind_test()
        self.controller.bind_train()
        # self.controller.bind_valid()
        
    
    def create_window(self):
        
        self.window = super(Ocr_screen, self).__init__(self.parent)
        
        self.tab1 = QtGui.QWidget()
        self.tab2 = QtGui.QWidget()
        self.tab3 = QtGui.QWidget()
        
        self.addTab(self.tab1,"Train")
        self.addTab(self.tab2,"Test")
        # self.addTab(self.tab3,"Validation")
        
        self.controller = Controller(self)
        
        self.trainUI()
        self.testUI()
        # self.evalUI()
        
        self.setWindowTitle("Simple Ocr")
        # self.setMinimumSize(int(5*self.res), int(4*self.res))
        #Logo
        
        app_icon = QtGui.QIcon()
        app_icon.addFile('logo2.png', QtCore.QSize(65,65))
        self.appli.setWindowIcon(app_icon)
        
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
        
        # Start
        self.start_train_button = QtGui.QPushButton("Start training")
        self.start_train_button.setFont(font)
        
        # Select nb epoch
        self.label_epoch = QtGui.QLabel("Select number of epochs :")
        self.set_epoch = QtGui.QComboBox()
        self.set_epoch.addItem("1")
        self.set_epoch.addItem("100")
        self.set_epoch.addItem("500")
        self.set_epoch.addItem("1000")
        self.apply_epoch = QtGui.QPushButton("Apply")
        self.bottom_panel = QtGui.QHBoxLayout()
        self.bottom_panel.addWidget(self.label_epoch)
        self.bottom_panel.addWidget(self.set_epoch)
        self.bottom_panel.addWidget(self.apply_epoch)
        self.spacerino = QtGui.QSpacerItem(100, 20)
        self.bottom_panel.addItem(self.spacerino)
        self.bottom_panel.addWidget(self.start_train_button)
        
        # Right pannel :
        self.label_gt = QtGui.QLabel("Ground truth :")
        self.ground_truth = QtGui.QLineEdit()
        self.ground_truth.setText("Bonjour je suis le baguette")
        self.gt_panel = QtGui.QHBoxLayout()
        self.gt_panel.addWidget(self.label_gt)
        self.gt_panel.addWidget(self.ground_truth)
        
        self.label_pred = QtGui.QLabel("Predicted      :")
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
    
        
        # Final Layout
        self.Final_panel = QtGui.QVBoxLayout()
        self.Final_panel.addLayout(self.main_panel)
        self.spacerino2 = QtGui.QSpacerItem(20, 30)
        self.Final_panel.addItem(self.spacerino2)
        self.Final_panel.addLayout(self.bottom_panel)
    
        self.tab1.setLayout(self.Final_panel)

    
    
    def testUI(self):
        
        #Larger font
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setWeight(40)
        self.big_spacer = QtGui.QSpacerItem(50, 50)
        self.small_spacer = QtGui.QSpacerItem(20, 20)
      
        # Display screen 
        self.screen2 = QtGui.QGraphicsView()
        self.scene2 = QtGui.QGraphicsScene(self.screen2)
        self.screen2.setScene(self.scene2)
        self.screen2.setGeometry(QtCore.QRect(0, 0, 3*self.res, 1*self.res))
        self.screen2.setMinimumSize(3*self.res, 1*self.res)
        self.screen2.setMaximumSize(3*self.res, 1*self.res)
        
        # Select network
        self.label_network = QtGui.QLabel("Select Network to test :")
        self.label_network.setFont(font)
        self.select_network = QtGui.QComboBox()
        
        for name in self.controller.network_names:
            self.select_network.addItem(name)
        
        self.apply_network = QtGui.QPushButton("Apply")
        self.network_panel = QtGui.QHBoxLayout()
        self.network_panel.addWidget(self.label_network)
        self.network_panel.addWidget(self.select_network)
        self.network_panel.addWidget(self.apply_network)
        
        # Bottom layout
        self.start_test_button = QtGui.QPushButton("Compute")
        self.start_test_button.setFont(font)

        
        # Text stuff :
        
        self.label_gt2 = QtGui.QLabel("Ground truth :")
        self.ground_truth2 = QtGui.QLineEdit()
        self.gt_panel2 = QtGui.QHBoxLayout()
        self.gt_panel2.addWidget(self.label_gt2)
        self.gt_panel2.addWidget(self.ground_truth2)
        
        self.label_pred2 = QtGui.QLabel("Predicted      :")
        self.predicted2 = QtGui.QLineEdit()
        self.pred_panel2 = QtGui.QHBoxLayout()
        self.pred_panel2.addWidget(self.label_pred2)
        self.pred_panel2.addWidget(self.predicted2)
        
        self.label_dist2 = QtGui.QLabel("Distance       :")
        self.dist2 = QtGui.QLineEdit()
        self.loss_panel2 = QtGui.QHBoxLayout()
        self.loss_panel2.addWidget(self.label_dist2)
        self.loss_panel2.addWidget(self.dist2)
        

        # Right layout
        self.right_panel2 = QtGui.QVBoxLayout()
        self.right_panel2.addWidget(self.screen2)
        self.right_panel2.setAlignment(self.screen2, QtCore.Qt.AlignHCenter)
        self.right_panel2.addLayout(self.gt_panel2)
        self.right_panel2.addLayout(self.pred_panel2)
        self.right_panel2.addLayout(self.loss_panel2)

        
        # Bottom Layout
        self.bottom_panel2 = QtGui.QHBoxLayout()
        self.bottom_panel2.addWidget(self.start_test_button)
        

        # Final Layout
        self.Final_panel2 = QtGui.QVBoxLayout()
        self.Final_panel2.addLayout(self.network_panel)
        self.Final_panel2.addLayout(self.right_panel2)
        self.Final_panel2.addLayout(self.bottom_panel2)
        
    
        self.tab2.setLayout(self.Final_panel2)
    
    
    def evalUI(self):
        
        #Larger font
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setWeight(40)
        self.big_spacer = QtGui.QSpacerItem(50, 50)
        self.small_spacer = QtGui.QSpacerItem(20, 20)
      
        # Display screen 
        self.screen3 = QtGui.QGraphicsView()
        self.scene3 = QtGui.QGraphicsScene(self.screen3)
        self.screen3.setScene(self.scene3)
        self.screen3.setGeometry(QtCore.QRect(0, 0, 3*self.res, 1*self.res))
        self.screen3.setMinimumSize(3*self.res, 1*self.res)
        self.screen3.setMaximumSize(3*self.res, 1*self.res)
        
        # Select network
        self.label_network2 = QtGui.QLabel("Select Network for translation :")
        self.label_network2.setFont(font)
        self.select_network2 = QtGui.QComboBox()
        
        for name in self.controller.network_names:
            self.select_network2.addItem(name)
        
        self.apply_network2 = QtGui.QPushButton("Apply")
        self.network_panel2 = QtGui.QHBoxLayout()
        self.network_panel2.addWidget(self.label_network2)
        self.network_panel2.addWidget(self.select_network2)
        self.network_panel2.addWidget(self.apply_network2)
        
        # Select book page
        self.label_page = QtGui.QLabel("Select Book Page to translate  :")
        self.label_page.setFont(font)
        self.select_page = QtGui.QComboBox()
        
        for name in self.controller.page_names:
            self.select_page.addItem(name)
        
        self.apply_page = QtGui.QPushButton("Apply")
        self.page_panel = QtGui.QHBoxLayout()
        self.page_panel.addWidget(self.label_page)
        self.page_panel.addWidget(self.select_page)
        self.page_panel.addWidget(self.apply_page)
        
        # Bottom layout
        self.start_eval_button = QtGui.QPushButton("Start validation")
        self.start_eval_button.setFont(font)
        self.pause_eval_button = QtGui.QPushButton("Pause validation")
        self.pause_eval_button.setFont(font)
        self.end_eval_button = QtGui.QPushButton("End validation")
        self.end_eval_button.setFont(font)
        
        # Text stuff :
        
        self.label_pred3 = QtGui.QLabel("Predicted      :")
        self.predicted3 = QtGui.QLineEdit()
        self.pred_panel3 = QtGui.QHBoxLayout()
        self.pred_panel3.addWidget(self.label_pred3)
        self.pred_panel3.addWidget(self.predicted3)
        

        # Right layout
        self.right_panel3 = QtGui.QVBoxLayout()
        self.right_panel3.addWidget(self.screen3)
        self.right_panel3.setAlignment(self.screen3, QtCore.Qt.AlignHCenter)
        self.right_panel3.addLayout(self.pred_panel3)


        
        # Bottom Layout
        self.bottom_panel3 = QtGui.QHBoxLayout()
        self.bottom_panel3.addWidget(self.start_eval_button)
        self.bottom_panel3.addWidget(self.pause_eval_button)
        self.bottom_panel3.addWidget(self.end_eval_button)
        

        # Final Layout
        self.Final_panel3 = QtGui.QVBoxLayout()
        self.Final_panel3.addLayout(self.page_panel)
        self.Final_panel3.addLayout(self.network_panel2)
        self.Final_panel3.addLayout(self.right_panel3)

        self.Final_panel3.addLayout(self.bottom_panel3)
        
    
        self.tab3.setLayout(self.Final_panel3)

def main():
    app = QtGui.QApplication(sys.argv)
    gui = Ocr_screen(app)
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
    if os.name == 'nt':
        path = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
        os.chdir(path + "\\app")
    main()