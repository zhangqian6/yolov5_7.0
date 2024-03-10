from PyQt5.QtWidgets import QMainWindow,QApplication
import firstMain22
import sys

class main_window(QMainWindow, firstMain22.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = main_window()
    window.show()
    sys.exit(app.exec_())

