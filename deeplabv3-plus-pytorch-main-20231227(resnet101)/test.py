import time

import cv2
import numpy as np
from PIL import Image
import torch
from numpy import half
from torch import device

from pspnet import PSPNet


def predict_(img):
    if __name__ == "__main__":
        pspnet = PSPNet()
    mode = "predict"
    count           = False
    name_classes    = ["background","theme","test","image"]
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    test_interval = 100
    fps_image_path  = (img)
    dir_origin_path = "../test_image/"
    dir_save_path   = "./test-result/"
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

def predict(img):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                               agnostic=opt.agnostic_nms)
    t2 = time_synchronized()
    InferNms = round((t2 - t1), 2)

    return pred, InferNms


def cv_imread(filePath):
    # 读取图片
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)

    if len(cv_img.shape) > 2:
        if cv_img.shape[2] > 3:
            cv_img = cv_img[:, :, :3]
    return cv_img


def plot_one_box(img, x, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    img_path = "./UI_rec/test_/ori_XYGOC20200407144715047-4_0.jpg"
    image = cv_imread(img_path)
    image = cv2.resize(image, (850, 500))
    img0 = image.copy()
    img = letterbox(img0, new_shape=imgsz)[0]
    img = np.stack(img, 0)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    pred, useTime = predict(img)

    det = pred[0]
    p, s, im0 = None, '', img0
    if det is not None and len(det):  # 如果有检测信息则进入
        det[:, :4] = scale_coords(img.shape[1:], det[:, :4], im0.shape).round()  # 把图像缩放至im0的尺寸
        number_i = 0  # 类别预编号
        detInfo = []
        for *xyxy, conf, cls in reversed(det):  # 遍历检测信息
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            # 将检测信息添加到字典中
            detInfo.append([names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf])
            number_i += 1  # 编号数+1

            label = '%s %.2f' % (names[int(cls)], conf)

            # 画出检测到的目标物
            plot_one_box(image, xyxy, label=label, color=colors[int(cls)])
    # 实时显示检测画面
    cv2.imshow('Stream', image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    c = cv2.waitKey(0) & 0xff