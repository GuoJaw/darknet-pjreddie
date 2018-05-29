
#coding=utf-8


# main程序头文件


import os
import sys
import cv2
import numpy as np
import argparse
import time  # fps

sys.path.append("./python")  # 导入某个头文件下的.py
from sort import Sort

# darknet头文件
from ctypes import *
import math
import random
import cv2
import time
from PIL import Image
import numpy as np

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int




predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def video_detect(net, meta, frame, frame_tmp, thresh=.5, hier_thresh=.5, nms=.45):
    img_arr = Image.fromarray(frame)  # 将frame保存到本地frame_tmp
    img_goal = img_arr.save(frame_tmp)

    im = load_image(frame_tmp, 0, 0)  # 从本地读取图像

    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)

    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)

    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    result = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                ltx = int(b.x - b.w / 2)
                lty = int(b.y - b.h / 2)
                rbx = int(b.x + b.w / 2)
                rby = int(b.y + b.h / 2)
                cv2.rectangle(frame, (ltx, lty), (rbx, rby), (0, 255, 0), 2)
                result.append([ltx, lty, rbx, rby, int(dets[j].prob[i])])  # 左上角，右下角，置信度
    free_image(im)
    free_detections(dets, num)

    result = np.array(result)  # 提出result，给det

    det = []
    if result != []:  # 非常重要，非空判断：否则检测不到物体时，程序会崩溃
        det = result[:, 0:5]
    return det



def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()

    # track_sort的参数
    parser.add_argument('--sort_max_age',default=5,type=int)
    parser.add_argument('--sort_min_hit',default=3,type=int)

    return parser.parse_args()

# 全局变量
colours = np.random.rand(32,3)*255

if __name__=="__main__":
    args=parse_args()

    ## track_sort初始化
    mot_tracker = Sort(args.sort_max_age, args.sort_min_hit)

    # 检测器加载网络
    net = load_net("/home/gjw/darknet-pjreddie/kitti/TestFile/yolov3_kitti.cfg",
                   "/home/gjw/darknet-pjreddie/kitti/TestFile/yolov3_kitti_final.weights", 0)
    meta = load_meta("/home/gjw/darknet-pjreddie/kitti/TestFile/kitti.data")

    frame_tmp = "/home/gjw/darknet-pjreddie/kitti/video_tmp.jpg"  # 暂时保存图像

    cap = cv2.VideoCapture('/home/gjw/darknet-pjreddie/kitti1.avi')  # 打开

    while (1) :
        ret, frame = cap.read()
        if ret is False:
            print("load video or capture error !")
            break

        start = time.time()  # fps开始时间
####
        det = video_detect(net, meta, frame, frame_tmp)  # frame获取的当前帧，frame_tmp临时存放图像的绝对路径
####
        trackers = mot_tracker.update(det) # 用mot_tracker的update接口去更新det，进行多目标的跟踪

        for track in trackers:
            # 左上角坐标(x,y)
            lrx=int(track[0])
            lry=int(track[1])

            # 右下角坐标(x,y)
            rtx=int(track[2])
            rty=int(track[3])

            #track_id
            trackID=int(track[4])

            cv2.putText(frame, str(trackID), (lrx,lry), cv2.FONT_ITALIC, 0.6, (int(colours[trackID%32,0]),int(colours[trackID%32,1]),int(colours[trackID%32,2])),2)
            cv2.rectangle(frame,(lrx,lry),(rtx,rty),(int(colours[trackID%32,0]),int(colours[trackID%32,1]),int(colours[trackID%32,2])),2)
####
        end = time.time() # fps结束时间
        fps = 1 / (end - start);
        print('FPS = %.2f' %(fps))


        #显示图像
        # frame = cv2.resize(frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # 图像放大为原来两倍
        cv2.imshow("frame",frame)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break

    cap.release()








