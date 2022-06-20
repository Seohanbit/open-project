# pyQt5
import imageio
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# tensorflow
import tensorflow as tf
        #tf.disable_v2_behavior()

import os
import cv2
import sys

# image
import dlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')


class QPushButtonIcon(QPushButton):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setFixedHeight(400)
        self.setFixedWidth(300)
        self.setIconSize(QSize(392, 392))

class QPushButtonReset(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        font = QFont("Helvetica", 12)
        font.setBold(True)
        self.setFont(font)

class QPushButtonSolution(QPushButton):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setFixedHeight(50)
        font = QFont("Helvetica", 12)
        font.setBold(True)
        self.setFont(font)

class Main(QDialog):
    def __init__(self):
        super().__init__()
        self.set_default()
        self.init_ui()

    def set_default(self):
        self.selection_list = []        # 선택한 사진 들어갈 공간
        self.figures = ['C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/makeup/민서윤.jpg',
                        'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/makeup/선미.jpg',
                        'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/makeup/서예지.jpg',
                        'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/makeup/한예슬.jpg',
                        'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/makeup/03.jpg',
                        'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/makeup/08.jpg']
        self.nomakeup=['C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/no_makeup/07.jpg',
                       'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/no_makeup/02.jpg',
                       'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/no_makeup/조세호.jpg',
                       'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/no_makeup/슬기.jpg',
                       'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/no_makeup/xfsy_0068.jpg',
                       'C:/Users/ADMIN/Desktop/BeautyGAN-master/imgs/no_makeup/06.jpg']

        self.icons = {}
        for index, filename in enumerate(self.figures):
            pixmap = QPixmap(filename)
            pixmap = pixmap.scaled(400, 350, Qt.IgnoreAspectRatio)
            icon = QIcon()
            icon.addPixmap(pixmap)
            self.icons[index] = icon

        self.icons_nomakeup = {}
        for index, filename in enumerate(self.nomakeup):
            pixmap = QPixmap(filename)
            pixmap = pixmap.scaled(400, 350, Qt.IgnoreAspectRatio)
            icon = QIcon()
            icon.addPixmap(pixmap)
            self.icons_nomakeup[index] = icon

    def init_ui(self):
        main_layout = QVBoxLayout()

        layout_1 = QHBoxLayout()
        layout_2 = QHBoxLayout()
        layout_3 = QVBoxLayout()

        self.qbuttons = {}      # 선택된 파일을 딕셔너리 형식으로 저장하기 위함
        for index, icon in self.icons.items():
            button = QPushButtonIcon()
            button.setIcon(icon)
            button.clicked.connect(lambda state, button = button, idx = index :
                                   self.qbutton_clicked(state, idx, button))
            layout_1.addWidget(button)
            self.qbuttons[index] = button

        self.sbuttons ={}
        for index, icon in self.icons_nomakeup.items():
            button = QPushButtonIcon()
            button.setIcon(icon)
            button.clicked.connect(lambda state, button = button, idx = index :
                                   self.qbutton_clicked(state, idx, button))
            layout_2.addWidget(button)
            self.qbuttons[index] = button
        self.button_reset = QPushButtonReset("Reset")
        self.button_reset.clicked.connect(self.action_reset)

        self.button_start = QPushButtonSolution("Start")
        self.button_start.clicked.connect(self.action_start)

        layout_3.addWidget(self.button_reset)
        layout_3.addWidget(self.button_start)

        main_layout.addLayout(layout_1)
        main_layout.addLayout(layout_2)
        main_layout.addLayout(layout_3)

        main_layout.addLayout(main_layout)

        self.setLayout(main_layout)
        self.setFixedSize(main_layout.sizeHint())
        self.setWindowTitle("Beauty_GAN")
        self.show()


    def qbutton_clicked(self, state, idx, button):
        self.selection_list.append(idx)     # 화장한 사진 선택
        button.setDisabled(True)

    def sbutton_clicked(self, state, idx, button):
        self.selection_list.append(idx)     # 화장안한 사진 선택
        button.setDisabled(True)

    def set_button_selected_index(self, button, idx):
        sol_index = self.selection_list[idx]
        icon = self.icons[sol_index]
        button.setIcon(icon)

    def check_all_selected(self):
        return len(self.selection_list) == len(self.figures)

    # reset 버튼 눌렀을 때 위버튼과 아랫버튼이 같이 reset되야 한다.
    def action_reset(self):
        self.selection_list = []
        for button in self.qbuttons.values():
            button.setEnabled(True)
        #    button.setDisabled(False)
        for button in self.sbuttons.values():
            button.setDisabled(False)

    def preprocess(img):
        return (img / 255. - 0.5) * 2

    def postprocess(img):
        return ((img + 1.) * 127.5).astype(np.uint8)


    def action_start(self):         # tensorflow를 사용한 이미지 처리 함수
        tf.compat.v1.disable_eager_execution()

        tf.compat.v1.reset_default_graph()

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())    #tensorflow initializer()
        graph = tf.compat.v1.get_default_graph()

    #    saver = tf.compat.v1.train.import_meta_graph(os.path.join('model', 'model.meta'))     # 학습된 GAN 가져오기
        saver = tf.compat.v1.train.import_meta_graph('C:/Users/ADMIN/Desktop/BeautyGAN-master/model/model.meta')
    #    saver.restore(sess, tf.compat.v1.train.latest_checkpoint('model'))
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint('C:/Users/ADMIN/Desktop/BeautyGAN-master/model'))


        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')


        ######  Image    ####

        img_size = 256

        no_makeup = cv2.resize(imageio.v2.imread(self.nomakeup[self.selection_list[1]]), (img_size, img_size))
        X_img_expand = np.expand_dims(no_makeup,0)
        X_img = np.reshape(X_img_expand, (1, 256, 256, 3))

        makeup = cv2.resize(imageio.v2.imread(self.figures[self.selection_list[0]]), (img_size, img_size))
        Y_img_expand = np.expand_dims(makeup,0)
        Y_img = np.reshape(Y_img_expand,(1, 256, 256, 3))


        output = sess.run(Xs, feed_dict={
            X: X_img,
            Y: Y_img
        })

        output_img=np.squeeze(output)
        plt.imshow(output_img)
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    sys.exit(app.exec_())
