# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NOMAKEUP.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import MAKEUP
import main

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1292, 922)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_6 = QtWidgets.QPushButton(Dialog)
        self.pushButton_6.setMinimumSize(QtCore.QSize(420, 420))
        self.pushButton_6.setText("")
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout.addWidget(self.pushButton_6, 2, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setMinimumSize(QtCore.QSize(420, 420))
        self.pushButton_2.setText("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 1, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setMinimumSize(QtCore.QSize(420, 420))
        self.pushButton.setText("")
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 1, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setMinimumSize(QtCore.QSize(420, 420))
        self.pushButton_4.setText("")
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 2, 3, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(Dialog)
        self.pushButton_5.setMinimumSize(QtCore.QSize(420, 420))
        self.pushButton_5.setText("")
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 2, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setMinimumSize(QtCore.QSize(420, 420))
        self.pushButton_3.setText("")
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 1, 3, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setMinimumSize(QtCore.QSize(0, 50))
        self.label.setMaximumSize(QtCore.QSize(16777215, 50))
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)


        #self.pushButton.setStyleSheet('border-image:url(imgs/no_makeup/01.jpg);border:0px;')
        #self.pushButton_2.setStyleSheet('border-image:url(imgs/no_makeup/02.jpg);border:0px;')
        #self.pushButton_3.setStyleSheet('border-image:url(imgs/no_makeup/03.jpg);border:0px;')
        #self.pushButton_4.setStyleSheet('border-image:url(imgs/no_makeup/04.jpg);border:0px;')
        #self.pushButton_5.setStyleSheet('border-image:url(imgs/no_makeup/05.jpg);border:0px;')
        #self.pushButton_6.setStyleSheet('border-image:url(imgs/no_makeup/06.jpg);border:0px;')



    def initUI(self):
        # 버튼 클릭할때 연결되는 함수
        self.pushButton.clicked.connect(self.button_1)
        self.pushButton_2.clicked.connect(self.button_2)
        self.pushButton_3.clicked.connect(self.button_3)
        self.pushButton_4.clicked.connect(self.button_4)
        self.pushButton_5.clicked.connect(self.button_5)
        self.pushButton_6.clicked.connect(self.button_6)

    def button_1(self):
        print('버튼 1 눌림')
        '''
        self.hide()
        self.second = MAKEUP.secondwindow()
        self.second.exec()
        self.show()
        '''


    def button_2(self):
        #nmimg = self.pushButton_2
        print("버튼 2 눌림")


    def button_3(self):
        #nmimg = self.pushButton_3
        print("버튼 3 눌림")


    def button_4(self):
       #nmimg = self.pushButton_4
        print("버튼 4 눌림")


    def button_5(self):
    #  nmimg = self.pushButton_5
     print("버튼 5 눌림")

    def button_6(self):
    #   nmimg = self.pushButton_6
     print("버튼 6 눌림")

    def retranslateUi(self, Dialog):
     _translate = QtCore.QCoreApplication.translate
     Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
     self.label.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:26pt; font-weight:600;\">NOMAKE UP IMAGE</span></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())