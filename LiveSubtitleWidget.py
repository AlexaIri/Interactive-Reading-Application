import sys
from PyQt5 import QtWidgets, QtCore, QtGui

from PyQt5.QtCore import pyqtSignal, QObject, QCoreApplication
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QLabel, QPushButton, QGroupBox 
from PyQt5.QtGui import QFont
import tkinter as tk


class LiveSubtitleWidget(QMainWindow):
    def __init__(self,font_size, lang = "en-US"):
        QMainWindow.__init__(self, None, QtCore.Qt.WindowStaysOnTopHint )
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        #above line makes sure that the screen is always on top, and that there is no frame(i.e. no close, minimise...)
        #QtCore.Qt.FramelessWindowHint|
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.win_width = screen_width - 0.2*screen_width
        self.win_height = 0.15*screen_height
        x_co_ord = (screen_width - self.win_width)/2
        y_co_ord = screen_height - 0.15*screen_height
        self.setGeometry(x_co_ord, y_co_ord, self.win_width, self.win_height)
        self.setWindowTitle("")
        self.setStyleSheet("background-color: black;") #bg colour of window
        self.setWindowOpacity(0.78) # make translucent
        
        self.initUI(font_size)
        
    
    def initUI(self,font_size):
        self.notificationText = QLabel(self)
        self.notificationText.setText("Live Subtitles to appear on here")
        self.notificationText.setFont(QFont('Arial', font_size))
        self.notificationText.setStyleSheet("color: white;") #change text colour
        self.notificationText.adjustSize()
        self.notificationText.move(( self.win_width - self.notificationText.width() ) / 2, ( self.win_height - self.notificationText.height() ) / 2)
        self.notificationText.setAlignment(QtCore.Qt.AlignCenter)
        
    
    def adjust(self):
        self.notificationText.adjustSize()
        self.notificationText.move(( self.win_width - self.notificationText.width() ) / 2, ( self.win_height - self.notificationText.height() ) / 2)
        

    

# app = QApplication(sys.argv)
# win = LiveSubtitleWidget()
# win.show()
# sys.exit(app.exec_())
