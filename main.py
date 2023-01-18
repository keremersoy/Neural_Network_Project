# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cnn.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import os
import os.path
projectDir=str(os.path.dirname(os.path.abspath(__file__)))

from keras.models import load_model
from PIL import Image
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import shutil

def err_msg(err):
    print(err)
    msg = QtWidgets.QMessageBox()
    msg.setWindowTitle("Hata!")
    msg.setText("Error: "+str(err))
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.exec_()


class Ui_MainWindow(object):

    def searchModelDialog(self):
        try:
            fileName = QtWidgets.QFileDialog.getOpenFileName(
                None, "Open File", "./train_model/temp", "Model Files (*.h5)")
            if fileName:
                self.loadModel(str(fileName[0]))

        except Exception as err:
            err_msg(err)

    def addImg(self, image_path):

        self.imgPred.setScene(None)
        scene = QtWidgets.QGraphicsScene()
        pixmap = QtGui.QPixmap(image_path)
        pixmapScaled = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        item = QtWidgets.QGraphicsPixmapItem(pixmapScaled)
        scene.addItem(item)
        self.imgPred.setScene(scene)

    def loadModel(self, path):
        try:
            global loadedModel
            loadedModel = load_model(path)
            self.lblModel.setText(path)
            p=path.split("/")
            self.lblModel.setText(p[-1])
        except Exception as err:
            err_msg(err)

    def searcImgDialog(self):
        image_path = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open File", "./predict_image", "Model Files (*.png *.jpg)")
        if image_path:
            self.addImg(str(image_path[0]))
            loaded_img=image_path[0]
            shutil.copyfile(loaded_img, projectDir+"\\train_model\\temp\\pred.png")

    def pred(self):
        try:
            img=Image.open(projectDir+"\\train_model\\temp\\pred.png")
            img = img.resize((75, 75))
            img_array = np.array(img)
            img_array=img_array*1./255
            img_array = img_array.reshape(1, 75, 75, 3)
            result = loadedModel.predict(img_array, batch_size=1)
            result=np.round(result)[0]
            pred="Vehicle"
            if result==0:
                pred="Non-Vehicle"

            self.lblPred.setText(pred)
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Tahmin")
            msg.setText(pred)
            msg.exec_()
        except Exception as err:
            err_msg(err)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(456, 589)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lblModel = QtWidgets.QLabel(self.centralwidget)
        self.lblModel.setGeometry(QtCore.QRect(129, 445, 291, 41))
        self.lblModel.setObjectName("lblModel")

        self.btnPhoto = QtWidgets.QPushButton(self.centralwidget)
        self.btnPhoto.setGeometry(QtCore.QRect(32, 423, 401, 28))
        self.btnPhoto.setObjectName("btnPhoto")
        self.btnPhoto.clicked.connect(self.searcImgDialog)

        self.imgPred = QtWidgets.QGraphicsView(self.centralwidget)
        self.imgPred.setGeometry(QtCore.QRect(32, 23, 400, 400))
        self.imgPred.setObjectName("imgPred")
        self.lblPred = QtWidgets.QLabel(self.centralwidget)
        self.lblPred.setGeometry(QtCore.QRect(249, 523, 80, 41))
        self.lblPred.setObjectName("lblPred")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(109, 510, 131, 61))
        self.label_4.setObjectName("label_4")

        self.btnLoadModel = QtWidgets.QPushButton(self.centralwidget)
        self.btnLoadModel.setGeometry(QtCore.QRect(32, 450, 93, 28))
        self.btnLoadModel.setObjectName("btnLoadModel")
        self.btnLoadModel.clicked.connect(self.searchModelDialog)

        self.btnPred = QtWidgets.QPushButton(self.centralwidget)
        self.btnPred.setGeometry(QtCore.QRect(32, 480, 93, 28))
        self.btnPred.setObjectName("btnPred")
        self.btnPred.clicked.connect(self.pred)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lblModel.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:7pt;\">X</span></p></body></html>"))
        self.btnPhoto.setText(_translate("MainWindow", "Fotoğraf Seç"))
        self.lblPred.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:11pt;\">X</span></p></body></html>"))
        self.label_4.setText(_translate(
            "MainWindow", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600;\">Tahmin:</span></p></body></html>"))
        self.btnLoadModel.setText(_translate("MainWindow", "Model Seç"))
        self.btnPred.setText(_translate("MainWindow", "Tahmin"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
