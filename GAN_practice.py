# pyQt5
import sys

import imageio
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# tensorflow
import tensorflow as tf
        #tf.disable_v2_behavior()

import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
import sys

# image
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')


class QPushButtonIcon(QPushButton):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setFixedHeight(200)
        self.setFixedWidth(200)
        self.setIconSize(QSize(192, 192))

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
        self.figures = ['imgs/makeup/01.jpg', 'imgs/makeup/02.jpg', 'imgs/makeup/03.jpg', 'imgs/makeup/04.jpg', 'imgs/makeup/05.jpg', 'imgs/makeup/06.jpg']
        self.nomakeup=['imgs/no_makeup/윤아.jpeg','imgs/no_makeup/02.jpg','imgs/no_makeup/03.jpg','imgs/no_makeup/04.jpg','imgs/no_makeup/05.jpg','imgs/no_makeup/06.jpg']

        self.icons = {}
        for index, filename in enumerate(self.figures):
            pixmap = QPixmap(filename)
            pixmap = pixmap.scaled(200, 200, Qt.IgnoreAspectRatio)
            icon = QIcon()
            icon.addPixmap(pixmap)
            self.icons[index] = icon

        self.icons_nomakeup = {}
        for index, filename in enumerate(self.nomakeup):
            pixmap = QPixmap(filename)
            pixmap = pixmap.scaled(200, 200, Qt.IgnoreAspectRatio)
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


    #    for index in range(len(self.figures)):
    #        button = QPushButtonIcon()
    #        self.sbuttons[index] = button
    #        button.clicked.connect(lambda state, button = button, idx = index:
    #                               self.sbutton_clicked(state, idx, button))
    #        layout_2.addWidget(button)

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
    #    if len(self.selection_list) > idx:
    #        self.set_button_selected_index(button, idx)

    def set_button_selected_index(self, button, idx):
        sol_index = self.selection_list[idx]
        icon = self.icons[sol_index]
        button.setIcon(icon)

    def check_all_selected(self):
        return len(self.selection_list) == len(self.figures)

    # reset 버튼 눌렀을 때 위버튼과 아랫버튼이 같이 reset되야 한다.
    def action_reset(self):
        self.selection_list = []
        for button in self.sbuttons.values():
            button.setIcon(QIcon())
        #for button in self.qbuttons.values():
        #    button.setDisabled(False)
        for button in self.sbuttons.values():
            button.setIcon(QIcon())

    def preprocess(img):
        return (img / 255. - 0.5) * 2
        # return img.astype(np.float32)/127.5+1.

    def postprocess(img):
    #    return (img + 1) / 2
        return ((img + 1.) * 127.5).astype(np.uint8)


    def align_faces(img):
        dets = detector(img, 1)

        objs = dlib.full_object_detections()

        for detection in dets:
            s = sp(img, detection)
            objs.append(s)

        faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)

        return faces


    def action_start(self):         # tensorflow를 사용한 이미지 처리 함수
        tf.compat.v1.disable_eager_execution()

        tf.compat.v1.reset_default_graph()

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())    #tensorflow initializer()
        graph = tf.compat.v1.get_default_graph()

        saver = tf.compat.v1.train.import_meta_graph(os.path.join('model', 'model.meta'))     # 학습된 GAN 가져오기
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint('model'))

        # X = tf.compat.v1.placeholder(dtype=tf.float32, name='X')
        # Y = tf.compat.v1.placeholder(dtype=tf.float32, name='Y')
        # Xs = tf.compat.v1.placeholder(dtype=tf.float32, name='Xs')

        # Y = graph.get_tensor_by_name('Y:0')  # reference

       # X = sess.graph.get_tensor_by_name('X:0')
       # Y = sess.graph.get_tensor_by_name('Y:0')
       # Xs = sess.graph.get_tensor_by_name('xs:0')


        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')

       # X = tf.compat.v1.get_default_graph()              # source
       # Y = tf.compat.v1.get_default_graph()              # reference
       # Xs = tf.compat.v1.get_default_graph()             # output

        ######  Image    ####

        batch_size = 1
        img_size = 256

        nomakeup=dlib.load_rgb_image(self.nomakeup[self.selection_list[1]])
    #    nomakeup_align=self.align_faces(nomakeup)

        makeup=dlib.load_rgb_image(self.figures[self.selection_list[0]])
    #    makeup_align = self.align_faces(makeup)

        src_img=nomakeup[0]
        ref_img=makeup[0]

        X_img=src_img.astype(np.float32)/127.5-1
        X_img=np.expand_dims(X_img,axis=0)

        Y_img=ref_img.astype(np.float32)/127.5-1
        Y_img=np.expand_dims(Y_img,axis=0)

        output=sess.run(Xs,feed_dict={
            X:X_img,
            Y:Y_img
        })

        output_img=((output+1.)*127.5).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        axes[0].set_title('Source')
        axes[0].imshow(src_img)
        axes[1].set_title('Reference')
        axes[1].imshow(ref_img)
        axes[2].set_title('Result')
        axes[2].imshow(output_img)





    #    no_makeup = cv2.resize(imageio.v2.imread(self.nomakeup[self.selection_list[1]]), (img_size, img_size))
    #    X_img = np.expand_dims(no_makeup,0)

    #    makeup = cv2.resize(imageio.v2.imread(self.figures[self.selection_list[0]]), (img_size, img_size))
    #    Y_img = np.expand_dims(makeup,0)

    #    result = np.ones((2 * img_size, (len(makeup) + 1) * img_size, 3))
    #    result[img_size: 2 * img_size, :img_size] = no_makeup / 255.

        # img_src = dlib.load_rgb_image(self.selection_list[1])
    #    img_src = dlib.load_rgb_image(self.nomakeup[self.selection_list[1]])
       # img_src_faces_align = self.align_faces(img_src)


        # img_ref = dlib.load_rgb_image(self.selection_list[0])
    #    img_ref = dlib.load_rgb_image(self.figures[self.selection_list[0]])
       # img_ref_align=self.align_faces(img_ref)

    #    src_img = img_src[0]
    #    ref_img = img_ref[0]

        # X_img = self.preprocess(src_img)
        # X_img = np.expand_dims(X_img, axis=0)

       # X_img = np.reshape(src_img,(1,256,256,3))
       # X_img = np.expand_dims(src_img, axis=0)

        # Y_img = self.preprocess(ref_img)
        # Y_img = np.expand_dims(Y_img, axis=0)

       # Y_img=np.reshape(ref_img(1,256,256,3))
       # Y_img = np.expand_dims(ref_img, axis=0)

      #  for i in range(len(makeup)):
    #    makeup = cv2.resize(imread(self.figures[self.selection_list[0]]), (img_size, img_size))
    #    Y_img = np.expand_dims(makeup, 0)

    #    output = sess.run(Xs, feed_dict={
    #        X: X_img,
    #        Y: Y_img
    #    })

    #    output_img= ((output + 1.) * 127.5).astype(np.uint8)
    #    output_img = self.postprocess(output[0])

       # result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
       # result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = output[0]

       # output_img = self.postprocess(output)

       # imsave('result.jpg', result)
       # result_png = cv2.imread('result.jpg', 1)
       # cv2.imshow('result_png', result_png)
       # cv2.waitKey()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    sys.exit(app.exec_())




'''
    #    parser = argparse.ArgumentParser()
    #    parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup', 'xfsy_0068.png'),
    #                        help='path to the no_makeup image')
    #    args = parser.parse_args()

        def preprocess(img):
            return (img / 255. - 0.5) * 2

        def deprocess(img):
            return (img + 1) / 2

    #    batch_size = 1
        img_size = 256
        no_makeup = self.selection_list[1]   #cv2.resize(imread(args.no_makeup), (img_size, img_size))
        X_img = np.expand_dims(preprocess(no_makeup), 0)
        makeups = self.selection_list[0]     #glob.glob(os.path.join('imgs', 'makeup', '*.*'))

        result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
        result[img_size: 2 * img_size, :img_size] = no_makeup / 255.

        tf.reset_default_graph()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))
        saver.restore(sess, tf.train.latest_checkpoint('model'))

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        Y = graph.get_tensor_by_name('Y:0')
        Xs = graph.get_tensor_by_name('generator/xs:0')

        for i in range(len(makeups)):
            makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
            Y_img = np.expand_dims(preprocess(makeup), 0)
            Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
            Xs_ = deprocess(Xs_)
            result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
            result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]

        imsave('result.jpg', result)
        result_png = cv2.imread('result.jpg', 1)
        cv2.imshow('result_png', result_png)
        cv2.waitKey()

'''
