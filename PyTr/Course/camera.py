# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'camera_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_for_cam = QtWidgets.QLabel(self.centralwidget)
        self.label_for_cam.setText("")
        self.label_for_cam.setObjectName("label_for_cam")
        self.verticalLayout.addWidget(self.label_for_cam)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.camera_toolbar = QtWidgets.QToolBar(MainWindow)
        self.camera_toolbar.setObjectName("camera_toolbar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.camera_toolbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Распознавание жестов"))
        self.camera_toolbar.setWindowTitle(_translate("MainWindow", "toolBar"))
import res_img
