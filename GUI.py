from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QDialog,QTabWidget, QGroupBox, QComboBox ,QVBoxLayout, QGridLayout, QWidget, QLabel, QLineEdit
from PredictionController import *
if __name__ == "__main__":
    app = QApplication(sys.argv)
    MW = MainWindow()
    MW.show()
    sys.exit(app.exec_())