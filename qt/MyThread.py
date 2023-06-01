from PyQt5.QtCore import QThread,pyqtSignal
from utils.augmentations import letterbox
from PyQt5 import QtWidgets
import numpy as np
import torch
import pyqtgraph as pg
from utils.general import non_max_suppression,scale_boxes
from utils.plots import Annotator,colors

class MyThread(QThread):
    finished_signal = pyqtSignal(object)

    def __init__(self,main_thread):
        super().__init__()
        self.main_thread = main_thread


    def runss(self,img):
        # 耗时操作
        img,ratio = self.detect(img)

        #发射信号，通知主线程完成
        data = [img,ratio]
        self.finished_signal.emit(data)
    # 单张图片检测
    def detect(self, img):
        # Run inference对导入的图片进行推理
        label = ...
        im = letterbox(img, self.main_thread.imgsz)[0]  # padded resize将原图变成特定大小的图片(640,480,3)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.main_thread.model.device)
        im = im.half() if self.main_thread.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.main_thread.model(im, augment=self.main_thread.opt.augment, visualize=False)
        # NMS
        pred = non_max_suppression(pred, self.main_thread.opt.conf_thres, self.main_thread.opt.iou_thres, self.main_thread.opt.classes,
                                   self.main_thread.opt.agnostic_nms, max_det=self.main_thread.opt.max_det)

        self.names = self.main_thread.model.names
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        labels = {'anger':0.000,'disgust':0.000,'fear':0.000,'happy':0.000,'neutral':0.000,'sad':0.000,'surprised':0.000}
        count = [0,0,0,0,0,0,0]
        max=...
        max_conf=...
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(img, line_width=self.main_thread.opt.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                print(len(det))
                # Write results
                x = 0
                for *xyxy, conf, cls in reversed(det):
                    print('ccccllllssss为',cls,'conf为',conf)
                    if self.main_thread.save_img or self.main_thread.opt.save_crop or self.main_thread.opt.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.main_thread.opt.hide_labels else (
                            self.names[c] if self.main_thread.opt.hide_conf else f'{self.names[c]} {conf:.2f}')
                        x += 1
                        xxx = label.split(' ')
                        name = xxx[0]
                        if x==len(det):
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        else:
                            count[c] += 1
                        max=c
                        max_conf=float(xxx[1])
                        labels[name] += float(xxx[1])

            img = annotator.result()
        print('anger:',labels['anger'],'disgust:',labels['disgust'],'fear:',labels['fear'],'happy:',labels['happy'],'neutral:',labels['neutral'],'sad',labels['sad'],'surprised:',labels['surprised'])
        total = sum(count)
        ratio = [x/total*0.13 for x in count]
        for x in range(len(ratio)):
            if ratio[x]<0.01:
                ratio[x]=0
            else:
                ratio[x]=float(ratio[x])
                ratio[x]=round(ratio[x],2)
        ratio[max] += max_conf
        print(ratio)
        return img,ratio


