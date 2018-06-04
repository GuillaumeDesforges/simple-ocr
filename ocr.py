import inspect
import os
import sys

from PyQt4 import QtGui

from app.Screen import Ocr_screen


def main():
    app = QtGui.QApplication(sys.argv)
    gui = Ocr_screen(app)
    gui.show()

    # Test
    gui.controller.preview("test.png")

    sys.exit(app.exec_())


if __name__ == '__main__':
    if os.name == 'nt':
        path = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
        os.chdir(path + "\\app")
    main()
