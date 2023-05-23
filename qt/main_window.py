import sys
import detect
import argparse
import torch
import os
import numpy as np
import pyqtgraph as pg

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from pyqt.io import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5 import QtWidgets, QtCore
from PyQt5 import QtGui
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import check_img_size, Profile, increment_path, non_max_suppression, scale_boxes, xyxy2xywh, \
    check_imshow
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.augmentations import letterbox

from utils.plots import Annotator, colors, save_one_box
from MyThread import MyThread
from typing import Dict, Any
import cv2

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class main_window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.timer_video = QtCore.QTimer()  # 创建定时器，作用是定时执行某个函数
        self.timer_video.timeout.connect(self.detect_frame)
        self.color_bar = (107, 200, 224)
        # 子线程
        self.my_thread = MyThread(self)
        self.color_bar = (107, 200, 224)
        # 清除直方图内容
        self.iii = 0
        # 用于暂停和继续
        self.ooo = 0
        # 用于判断是视频还是摄像头
        self.ppp=0
        # 初始化直方图
        self.init_pic()
        # 初始化槽函数
        self.init_slot()




    # 直方图初始化
    def init_pic(self):
        # 直方图
        xax = pg.AxisItem(orientation='bottom', maxTickLength=5)
        # xax.setHeight(h=40)
        self.pw = pg.PlotWidget(axisItems={'bottom': xax})
        self.pw.setYRange(0, 1)
        # x轴
        xTick = [(0, '生气'), (1, '厌恶'), (2, '害怕'), (3, '高兴'), (4, '中性'), (5, '悲伤'),
                 (6, '惊讶')]
        xax = self.pw.getAxis('bottom')
        xax.setTicks([xTick])
        # y轴名字
        y_name = '置信度'
        self.pw.setLabel('left', y_name)

        self.y_data = [0, 0, 0, 0, 0, 0, 0]

        # 将直方图添加到widget中
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pw)
        self.widget.setLayout(layout)
        self.pre_data()
    # 槽函数初始化
    def init_slot(self):
        self.button1.clicked.connect(self.model_init)
        self.button2.clicked.connect(self.detect_picture)
        self.pushButton.clicked.connect(self.detect_vedio)
        self.pushButton_2.clicked.connect(self.detect_camera)
        self.pushButton_3.clicked.connect(self.select_model)
        self.pause.clicked.connect(self.pause_cont)
        self.stop.clicked.connect(self.stop_button)
        self.doubleSpinBox.valueChanged.connect(self.conf_value_changed)
        self.doubleSpinBox_2.valueChanged.connect(self.iou_value_changed)

    def conf_value_changed(self,value):
        try:
            self.opt.conf_thres = value
        except:
            print('模型未初始化')

    def iou_value_changed(self,value):
        try:
            self.opt.iou_thres = value
        except:
            print('模型未初始化')
    # 选择模型
    def select_model(self):
        self.button1.setDisabled(True)
        self.button2.setDisabled(True)
        self.pushButton.setDisabled(True)
        self.pushButton_2.setDisabled(True)
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.pushButton_3, '选择weights文件',
                                                                  '../weight/')
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开权重失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

        else:
            print('加载weights文件地址为：' + str(self.openfile_name_model))
            self.button1.setEnabled(True)
            self.button2.setEnabled(True)
            self.pushButton.setEnabled(True)
            self.pushButton_2.setEnabled(True)


    # 初始化模型
    def model_init(self):
        self.pushButton_3.setDisabled(True)
        self.button2.setDisabled(True)
        self.pushButton.setDisabled(True)
        self.pushButton_2.setDisabled(True)
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='../runs/train/exp24/weights/best.pt',
                            help='model path or triton URL')
        parser.add_argument('--source', type=str, default='X:\毕业设计\源码\yolov5_7.0\yolov5\data\images\zidane.jpg',
                            help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default='X:\毕业设计\源码\yolov5_7.0\yolov5\data\coco128.yaml',
                            help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold')
        parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_false', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')



        self.opt = parser.parse_args()
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand
        # 更换opt中模型
        if self.openfile_name_model:
            self.opt.weights = self.openfile_name_model
            print("Using button choose model")
        print(self.opt)

        source = str(self.opt.source)
        self.save_img = not self.opt.nosave and not source.endswith('.txt')  # save inference images

        # Load model
        self.device = select_device(self.opt.device)
        self.model = DetectMultiBackend(self.opt.weights, device=self.device, dnn=self.opt.dnn, data=self.opt.data,
                                        fp16=self.opt.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.opt.imgsz, s=self.stride)  # check image size

        QtWidgets.QMessageBox.information(self, u"Notice", u"模型加载完成", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        self.pushButton_3.setEnabled(True)
        self.button2.setEnabled(True)
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(True)

    # 图片检测
    def detect_picture(self):
        self.button1.setDisabled(True)
        self.pushButton_3.setDisabled(True)
        self.pushButton.setDisabled(True)
        self.pushButton_2.setDisabled(True)
        # 导入图片
        try:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption="打开图片",
                                                                directory="X:\毕业设计\源码\yolov5_7.0\yolov5\data\images",
                                                                filter="*.jpg;;*.png;;All Files(*)")

        except OSError as reason:
            print('文件打开出错啦！核对路径是否正确' + str(reason))
        else:
            # 判断图片是否为空
            if not img_name:
                QtWidgets.QMessageBox.warning(self, "Warning", "打开图片失败")
            else:
                img = cv2.imread(img_name)
                print("img_name: ", img_name)
                # info = self.detect(img)
                # 通过线程来完成帧检测
                self.my_thread.finished_signal.connect(self.handle_result)
                # thread.start()
                self.my_thread.runss(img)

    # 子线程推理
    def handle_result(self, data):
        img = data[0]
        info = data[1]
        # 直接将原始img上的检测结果进行显示
        show = cv2.resize(img, (640, 480))
        self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label.setScaledContents(True)  # 设置图像自适应界面大小

        # 提取出表情类别和置信度
        xxx = info.split(' ')
        if xxx[0] == 'anger':
            self.y_data = [float(xxx[1]), 0, 0, 0, 0, 0, 0]
        elif xxx[0] == 'disgust':
            self.y_data = [0, float(xxx[1]), 0, 0, 0, 0, 0]
        elif xxx[0] == 'fear':
            self.y_data = [0, 0, float(xxx[1]), 0, 0, 0, 0]
        elif xxx[0] == 'happy':
            self.y_data = [0, 0, 0, float(xxx[1]), 0, 0, 0]
        elif xxx[0] == 'neutral':
            self.y_data = [0, 0, 0, 0, float(xxx[1]), 0, 0]
        elif xxx[0] == 'sad':
            self.y_data = [0, 0, 0, 0, 0, float(xxx[1]), 0]
        elif xxx[0] == 'surprised':
            self.y_data = [0, 0, 0, 0, 0, 0, float(xxx[1])]
        self.pre_data()
    # 打开视频并检测
    def detect_vedio(self):
        self.ppp = 1
        self.cap = cv2.VideoCapture()
        fname,x2 = QtWidgets.QFileDialog.getOpenFileName(self,caption='打开视频', directory='../data/images')
        flag = self.cap.open(fname)
        if not flag:
            QtWidgets.QMessageBox.warning(self,"Warning","打开视频失败", buttons=QtWidgets.QMessageBox.Ok,defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.timer_video.start(20)
            # self.button1.setDisabled(True)
            # self.button2.setDisabled(True)
            # self.pushButton_2.setDisabled(True)

    # 摄像头流检测
    def detect_camera(self):
        self.ppp = 0
        self.cap = cv2.VideoCapture(0)
        # 检查摄像头是否打开
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.timer_video.start(30)
            self.button1.setDisabled(True)
            self.button2.setDisabled(True)
            self.pushButton.setDisabled(True)
            self.pushButton_3.setDisabled(True)

    # 对一帧的推理
    def detect_frame(self):
        # 逐帧读取摄像头画面
        ret, frame = self.cap.read()
        if frame is not None:
            # 通过线程来完成帧检测
            # 通过线程来完成帧检测
            self.my_thread.finished_signal.connect(self.handle_result)
            # thread.start()
            self.my_thread.runss(frame)

    # 帧更改后改变直方图
    def pre_data(self):
        if self.iii == 1:
            self.pw.removeItem(self.barItem)
            self.iii = 0
        x = [0, 1, 2, 3, 4, 5, 6]
        self.barItem = pg.BarGraphItem(x=x, height=self.y_data, width=0.8, brush=self.color_bar)
        self.pw.addItem(self.barItem)
        self.iii = 1

    # 摄像头检测的暂停和继续
    def pause_cont(self):
        if self.ooo == 0:
            self.timer_video.stop()
            self.ooo=1
        else:
            self.timer_video.start()
            self.ooo = 0

    # 结束检测
    def stop_button(self):
        self.timer_video.stop()
        self.pw.removeItem(self.barItem)
        self.label.clear()
        self.cap.release()
        # 恢复按钮
        if self.ppp==0:
            self.button1.setEnabled(True)
            self.button2.setEnabled(True)
            self.pushButton.setEnabled(True)
        else:
            self.button1.setEnabled(True)
            self.button2.setEnabled(True)
            self.pushButton_2.setEnabled(True)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = main_window()
    window.show()
    sys.exit(app.exec_())
