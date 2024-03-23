# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'io.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1068, 671)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(0, 10, 1101, 661))
        self.groupBox.setStyleSheet("background-color: rgb(170, 255, 255);")
        self.groupBox.setObjectName("groupBox")
        self.button1 = QtWidgets.QPushButton(self.groupBox)
        self.button1.setGeometry(QtCore.QRect(10, 180, 121, 51))
        self.button1.setStyleSheet("background-color: rgb(199, 195, 200);")
        self.button1.setObjectName("button1")
        self.button2 = QtWidgets.QPushButton(self.groupBox)
        self.button2.setGeometry(QtCore.QRect(10, 260, 121, 51))
        self.button2.setStyleSheet("background-color: rgb(199, 195, 200);")
        icon = QtGui.QIcon.fromTheme("accessories-calculator")
        self.button2.setIcon(icon)
        self.button2.setObjectName("button2")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 340, 121, 51))
        self.pushButton.setStyleSheet("background-color: rgb(199, 195, 200);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 420, 121, 51))
        self.pushButton_2.setStyleSheet("background-color: rgb(199, 195, 200);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 40, 1051, 51))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setStyleSheet("font-size:20px")
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setStyleSheet("font-size:20px")
        self.label_6.setObjectName("label_6")
        self.horizontalLayout.addWidget(self.label_6)
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setStyleSheet("font-size:20px")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(660, 140, 401, 431))
        self.widget.setMaximumSize(QtCore.QSize(401, 431))
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(230, 140, 401, 431))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(100, 0))
        self.label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox.setGeometry(QtCore.QRect(90, 490, 91, 31))
        self.doubleSpinBox.setMaximum(1.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setProperty("value", 0.25)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(12, 108, 121, 51))
        self.pushButton_3.setStyleSheet("background-color: rgb(199, 195, 200);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 500, 69, 20))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(10, 540, 69, 20))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(90, 530, 91, 31))
        self.doubleSpinBox_2.setMaximum(1.0)
        self.doubleSpinBox_2.setSingleStep(0.01)
        self.doubleSpinBox_2.setProperty("value", 0.45)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(230, 0, 651, 51))
        self.label_4.setStyleSheet("font-size:27px")
        self.label_4.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.label_4.setObjectName("label_4")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(130, 609, 871, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pause = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.pause.setStyleSheet("background-color: rgb(199, 195, 200);")
        self.pause.setObjectName("pause")
        self.horizontalLayout_2.addWidget(self.pause)
        self.stop = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.stop.setStyleSheet("background-color: rgb(199, 195, 200);")
        self.stop.setObjectName("stop")
        self.horizontalLayout_2.addWidget(self.stop)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "操作区"))
        self.button1.setText(_translate("MainWindow", "模型初始化"))
        self.button2.setText(_translate("MainWindow", "图片检测"))
        self.pushButton.setText(_translate("MainWindow", "视频检测"))
        self.pushButton_2.setText(_translate("MainWindow", "摄像头检测"))
        self.label_5.setText(_translate("MainWindow", "    功能选择"))
        self.label_6.setText(_translate("MainWindow", "                 检测结果"))
        self.label_3.setText(_translate("MainWindow", "                  直方图"))
        self.label.setText(_translate("MainWindow", "图片"))
        self.pushButton_3.setText(_translate("MainWindow", "选择模型"))
        self.label_2.setText(_translate("MainWindow", "置信度"))
        self.label_7.setText(_translate("MainWindow", "IoU"))
        self.label_4.setText(_translate("MainWindow", "人脸表情检测工具"))
        self.pause.setText(_translate("MainWindow", "暂停/继续"))
        self.stop.setText(_translate("MainWindow", "结束"))