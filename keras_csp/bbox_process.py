import numpy as np
from keras_csp.nms_wrapper import nms
import cv2

import scipy.io as scio

def parse_det(Y, C, score=0.1, down=4, scale='h'):
    seman = Y[0][0, :, :, 0]
    if scale == 'h':
        height = np.exp(Y[1][0, :, :, 0]) * down
        width = 0.41 * height
    elif scale == 'w':
        width = np.exp(Y[1][0, :, :, 0]) * down
        height = width / 0.41
    elif scale == 'hw':
        height = np.exp(Y[1][0, :, :, 0]) * down
        width = np.exp(Y[1][0, :, :, 1]) * down
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = height[y_c[i], x_c[i]]
            w = width[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + 0.5) * down - w / 2), max(0, (y_c[i] + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs


def parse_det_top(Y, C, score=0.1):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * 4
            w = 0.41 * h
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, x_c[i] * 4 + 2 - w / 2), max(0, y_c[i] * 4 + 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs


def parse_det_bottom(Y, C, score=0.1):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * 4
            w = 0.41 * h
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, x_c[i] * 4 + 2 - w / 2), max(0, y_c[i] * 4 + 2 - h)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs


def parse_det_offset(Ycsp,Y, C, score=0.1, down=4):
    seman = Ycsp[0, :, :, 0]
    height = Y[1][0, :, :, 0]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41 * h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms3(Y, C, score=0.1,down=4):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]

    seman1 = Y[3][0, :, :, 0]
    height1 = Y[4][0, :, :, 0]
    offset_y1 = Y[5][0, :, :, 0]
    offset_x1 = Y[5][0, :, :, 1]
    y_c1, x_c1 = np.where(seman1 > score)
    boxs1 = []
    if len(y_c1) > 0:
        for i in range(len(y_c1)):
            h1 = np.exp(height1[y_c1[i], x_c1[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w1 = 0.41 * h1
            o_y1 = offset_y1[y_c1[i], x_c1[i]]
            o_x1 = offset_x1[y_c1[i], x_c1[i]]
            s1 = seman1[y_c1[i], x_c1[i]]
            x11, y11 = max(0, (x_c1[i] + o_x1 + 0.5) * down - w1 / 2), max(0, (y_c1[i] + o_y1 + 0.5) * down - h1 / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs1.append([x11, y11, min(x11 + w1, C.size_test[1]), min(y11 + h1, C.size_test[0]), s1])
        boxs1 = np.asarray(boxs1, dtype=np.float32)
        keep1 = nms(boxs1, 0.5, usegpu=False, gpu_id=0)
        boxs1 = boxs1[keep1, :]


    seman2 = Y[6][0, :, :, 0]
    height2 = Y[7][0, :, :, 0]
    offset_y2 = Y[8][0, :, :, 0]
    offset_x2 = Y[8][0, :, :, 1]
    y_c2, x_c2 = np.where(seman2 > score)
    boxs2 = []
    if len(y_c2) > 0:
        for i in range(len(y_c2)):
            h2 = np.exp(height2[y_c2[i], x_c2[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w2 = 0.41 * h2
            o_y2 = offset_y2[y_c2[i], x_c2[i]]
            o_x2 = offset_x2[y_c2[i], x_c2[i]]
            s2 = seman2[y_c2[i], x_c2[i]]
            x12, y12 = max(0, (x_c2[i] + o_x2 + 0.5) * down - w2 / 2), max(0, (y_c2[i] + o_y2 + 0.5) * down - h2 / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs2.append([x12, y12, min(x12 + w2, C.size_test[1]), min(y12 + h2, C.size_test[0]), s2])
        boxs2 = np.asarray(boxs2, dtype=np.float32)
        keep2 = nms(boxs2, 0.5, usegpu=False, gpu_id=0)
        boxs2 = boxs2[keep2, :]

    if (boxs !=[] and boxs1!=[]):
        boxs =np.vstack((boxs,boxs1))
    elif(boxs !=[] and boxs1==[]):
        boxs = boxs
    elif(boxs ==[] and boxs1!=[]):
        boxs = boxs1

    if (boxs !=[] and boxs2!=[]):
        boxs =np.vstack((boxs,boxs2))
    elif(boxs ==[] and boxs2!=[]):
        boxs = boxs2
    boxs = np.asarray(boxs, dtype=np.float32)
    keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
    boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms3anchor(Ys, Y, C, score=0.1,down=4):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    offset_y = Ys[2][0, :, :, 0]
    offset_x = Ys[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]

    seman1 = Y[2][0, :, :, 0]
    height1 = Y[3][0, :, :, 0]
    offset_y1 = Ys[2][0, :, :, 0]
    offset_x1 = Ys[2][0, :, :, 1]
    y_c1, x_c1 = np.where(seman1 > score)
    boxs1 = []
    if len(y_c1) > 0:
        for i in range(len(y_c1)):
            h1 = np.exp(height1[y_c1[i], x_c1[i]]) *2 * down
            # h = height[y_c[i], x_c[i]] * down
            w1 = 0.41 * h1
            o_y1 = offset_y1[y_c1[i], x_c1[i]]
            o_x1 = offset_x1[y_c1[i], x_c1[i]]
            s1 = seman1[y_c1[i], x_c1[i]]
            x11, y11 = max(0, (x_c1[i] + o_x1 + 0.5) * down - w1 / 2), max(0, (y_c1[i] + o_y1 + 0.5) * down - h1 / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs1.append([x11, y11, min(x11 + w1, C.size_test[1]), min(y11 + h1, C.size_test[0]), s1])
        boxs1 = np.asarray(boxs1, dtype=np.float32)
        keep1 = nms(boxs1, 0.5, usegpu=False, gpu_id=0)
        boxs1 = boxs1[keep1, :]


    seman2 = Y[4][0, :, :, 0]
    height2 = Y[5][0, :, :, 0]
    offset_y2 = Ys[2][0, :, :, 0]
    offset_x2 = Ys[2][0, :, :, 1]
    y_c2, x_c2 = np.where(seman2 > score)
    boxs2 = []
    if len(y_c2) > 0:
        for i in range(len(y_c2)):
            h2 = np.exp(height2[y_c2[i], x_c2[i]])*4 * down
            # h = height[y_c[i], x_c[i]] * down
            w2 = 0.41 * h2
            o_y2 = offset_y2[y_c2[i], x_c2[i]]
            o_x2 = offset_x2[y_c2[i], x_c2[i]]
            s2 = seman2[y_c2[i], x_c2[i]]
            x12, y12 = max(0, (x_c2[i] + o_x2 + 0.5) * down - w2 / 2), max(0, (y_c2[i] + o_y2 + 0.5) * down - h2 / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs2.append([x12, y12, min(x12 + w2, C.size_test[1]), min(y12 + h2, C.size_test[0]), s2])
        boxs2 = np.asarray(boxs2, dtype=np.float32)
        keep2 = nms(boxs2, 0.5, usegpu=False, gpu_id=0)
        boxs2 = boxs2[keep2, :]

    if (boxs !=[] and boxs1!=[]):
        boxs =np.vstack((boxs,boxs1))
    elif(boxs !=[] and boxs1==[]):
        boxs = boxs
    elif(boxs ==[] and boxs1!=[]):
        boxs = boxs1

    if (boxs !=[] and boxs2!=[]):
        boxs =np.vstack((boxs,boxs2))
    elif(boxs ==[] and boxs2!=[]):
        boxs = boxs2
    boxs = np.asarray(boxs, dtype=np.float32)
    keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
    boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms3down8(Y, C, score=0.1,down=8):
    seman = Y[0][0, :, :, 1]
    height = Y[1][0, :, :, 0]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]

    seman1 = Y[0][0, :, :, 2]
    height1 = Y[1][0, :, :, 1]
    offset_y1 = Y[2][0, :, :, 0]
    offset_x1 = Y[2][0, :, :, 1]
    y_c1, x_c1 = np.where(seman1 > score)
    boxs1 = []
    if len(y_c1) > 0:
        for i in range(len(y_c1)):
            h1 = np.exp(height1[y_c1[i], x_c1[i]]) *2 * down
            # h = height[y_c[i], x_c[i]] * down
            w1 = 0.41 * h1
            o_y1 = offset_y1[y_c1[i], x_c1[i]]
            o_x1 = offset_x1[y_c1[i], x_c1[i]]
            s1 = seman1[y_c1[i], x_c1[i]]
            x11, y11 = max(0, (x_c1[i] + o_x1 + 0.5) * down - w1 / 2), max(0, (y_c1[i] + o_y1 + 0.5) * down - h1 / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs1.append([x11, y11, min(x11 + w1, C.size_test[1]), min(y11 + h1, C.size_test[0]), s1])
        boxs1 = np.asarray(boxs1, dtype=np.float32)
        keep1 = nms(boxs1, 0.5, usegpu=False, gpu_id=0)
        boxs1 = boxs1[keep1, :]


    seman2 = Y[0][0, :, :, 3]
    height2 = Y[1][0, :, :, 2]
    offset_y2 = Y[2][0, :, :, 0]
    offset_x2 = Y[2][0, :, :, 1]
    y_c2, x_c2 = np.where(seman2 > score)
    boxs2 = []
    if len(y_c2) > 0:
        for i in range(len(y_c2)):
            h2 = np.exp(height2[y_c2[i], x_c2[i]])*4 * down
            # h = height[y_c[i], x_c[i]] * down
            w2 = 0.41 * h2
            o_y2 = offset_y2[y_c2[i], x_c2[i]]
            o_x2 = offset_x2[y_c2[i], x_c2[i]]
            s2 = seman2[y_c2[i], x_c2[i]]
            x12, y12 = max(0, (x_c2[i] + o_x2 + 0.5) * down - w2 / 2), max(0, (y_c2[i] + o_y2 + 0.5) * down - h2 / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs2.append([x12, y12, min(x12 + w2, C.size_test[1]), min(y12 + h2, C.size_test[0]), s2])
        boxs2 = np.asarray(boxs2, dtype=np.float32)
        keep2 = nms(boxs2, 0.5, usegpu=False, gpu_id=0)
        boxs2 = boxs2[keep2, :]

    if (boxs !=[] and boxs1!=[]):
        boxs =np.vstack((boxs,boxs1))
    elif(boxs !=[] and boxs1==[]):
        boxs = boxs
    elif(boxs ==[] and boxs1!=[]):
        boxs = boxs1

    if (boxs !=[] and boxs2!=[]):
        boxs =np.vstack((boxs,boxs2))
    elif(boxs ==[] and boxs2!=[]):
        boxs = boxs2
    boxs = np.asarray(boxs, dtype=np.float32)
    keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
    boxs = boxs[keep, :]
    return boxs



def parse_det_offset_ms2max(Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    offset_y1 = Y[2][0, :, :, 0]
    offset_x1 = Y[2][0, :, :, 1]
    semant2 = Y[3][0, :, :, 0]
    height2 = Y[4][0, :, :, 0]
    offset_y2 = Y[5][0, :, :, 0]
    offset_x2 = Y[5][0, :, :, 1]

    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_y = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_x = np.zeros((semant1.shape[0], semant1.shape[1]))
    for i in range(semant1.shape[0]):
        for j in range(semant1.shape[1]):
            if semant1[i,j]>semant2[i,j]:
                seman[i, j] = semant1[i,j]
                height[i, j] = height1[i, j]
                offset_y[i, j] = offset_y1[i,j]
                offset_x[i, j] = offset_x1[i, j]
            else:
                seman[i, j] = semant2[i,j]
                height[i, j] = height2[i, j]
                offset_y[i, j] = offset_y2[i, j]
                offset_x[i, j] = offset_x2[i, j]


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs


def parse_det_offset_ms3max(Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    offset_y1 = Y[2][0, :, :, 0]
    offset_x1 = Y[2][0, :, :, 1]

    semant2 = Y[3][0, :, :, 0]
    height2 = Y[4][0, :, :, 0]
    offset_y2 = Y[5][0, :, :, 0]
    offset_x2 = Y[5][0, :, :, 1]


    semant3 = Y[6][0, :, :, 0]
    height3 = Y[7][0, :, :, 0]
    offset_y3 = Y[8][0, :, :, 0]
    offset_x3 = Y[8][0, :, :, 1]

    #scio.savemat( '/media/xwj/Day/dominikandreaCSP/eval_caltech/1.mat', {'semant1': semant1,'semant2': semant2,'semant3': semant3})

    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_y = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_x = np.zeros((semant1.shape[0], semant1.shape[1]))
    for i in range(semant1.shape[0]):
        for j in range(semant1.shape[1]):
            s = [semant1[i,j],semant2[i,j],semant3[i,j]]
            index = np.argmax(s)
            if index ==0:
                seman[i, j] = semant1[i,j]
                height[i, j] = height1[i, j]
                offset_y[i, j] = offset_y1[i,j]
                offset_x[i, j] = offset_x1[i, j]
            elif index ==1:
                seman[i, j] = semant2[i,j]
                height[i, j] = height2[i, j]
                offset_y[i, j] = offset_y2[i, j]
                offset_x[i, j] = offset_x2[i, j]
            else:
                seman[i, j] = semant3[i,j]
                height[i, j] = height3[i, j]
                offset_y[i, j] = offset_y3[i, j]
                offset_x[i, j] = offset_x3[i, j]


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms3maxii(Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    offset_y1 = Y[2][0, :, :, 0]
    offset_x1 = Y[2][0, :, :, 1]
    semant2 = Y[3][0, :, :, 0]
    height2 = Y[4][0, :, :, 0]
    offset_y2 = Y[5][0, :, :, 0]
    offset_x2 = Y[5][0, :, :, 1]


    semant3 = Y[6][0, :, :, 0]
    height3 = Y[7][0, :, :, 0]
    offset_y3 = Y[8][0, :, :, 0]
    offset_x3 = Y[8][0, :, :, 1]
    semants = Y[9][0, :, :, 0]

    # plt.imshow(semants)
    # plt.show()

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    t = [semant1,semant2,semant3]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_y = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_x = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height1[y_c[k], x_c[k]]
            offset_y[y_c[k], x_c[k]] = offset_y1[y_c[k], x_c[k]]
            offset_x[y_c[k], x_c[k]] = offset_x1[y_c[k], x_c[k]]
    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height2[y_c[k], x_c[k]]
            offset_y[y_c[k], x_c[k]] = offset_y2[y_c[k], x_c[k]]
            offset_x[y_c[k], x_c[k]] = offset_x2[y_c[k], x_c[k]]

    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height3[y_c[k], x_c[k]]
            offset_y[y_c[k], x_c[k]] = offset_y3[y_c[k], x_c[k]]
            offset_x[y_c[k], x_c[k]] = offset_x3[y_c[k], x_c[k]]

    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms3maxcc(Y, C, score=0.1,down=4):
    seman = Y[0][0, :, :, 0]

    semant1 = Y[0][0, :, :, 1]
    semant2 = Y[0][0, :, :, 2]
    semant3 = Y[0][0, :, :, 3]

    height1 = Y[1][0, :, :, 0]
    height2 = Y[1][0, :, :, 1]
    height3 = Y[1][0, :, :, 2]

    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]


    # plt.imshow(semants)
    # plt.show()

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    t = [semant1,semant2,semant3]
    indext = np.argmax(t,0)

    height = np.zeros((semant1.shape[0], semant1.shape[1]))


    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            height[y_c[k], x_c[k]] = np.exp(height3[y_c[k], x_c[k]])* 4


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            # h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms3maxccT(Y, C, score=0.1,down=4):
    seman = Y[0][0, :, :, 0]
    semant1 = Y[0][0, :, :, 1]
    semant2 = Y[0][0, :, :, 2]
    semant3 = Y[0][0, :, :, 3]

    height1 = Y[1][0, :, :, 0]
    height2 = Y[1][0, :, :, 1]
    height3 = Y[1][0, :, :, 2]

    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]

    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            # h = np.exp(height[y_c[i], x_c[i]]) * down
            h1 =np.exp(height1[y_c[i], x_c[i]])* down
            h2 =np.exp(height2[y_c[i], x_c[i]])* 2* down
            h3 =np.exp(height3[y_c[i], x_c[i]])* 4* down
            if h1 < 112 and h1 >= 30:
                h=h1
            elif h2 < 224 and h2 >= 112:
                h=h2
            elif h3 >= 224:
                h=h3
            else:
                continue
                # h=(h1+h2 +h3)/3

            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs


def parse_det_offset_ms2maxi(Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    offset_y1 = Y[2][0, :, :, 0]
    offset_x1 = Y[2][0, :, :, 1]
    semant2 = Y[3][0, :, :, 0]
    height2 = Y[4][0, :, :, 0]
    offset_y2 = Y[5][0, :, :, 0]
    offset_x2 = Y[5][0, :, :, 1]

    semants = Y[6][0, :, :, 0]

    # plt.imshow(semants)
    # plt.show()

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)

    t = [semant1,semant2]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_y = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_x = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height1[y_c[k], x_c[k]]
            offset_y[y_c[k], x_c[k]] = offset_y1[y_c[k], x_c[k]]
            offset_x[y_c[k], x_c[k]] = offset_x1[y_c[k], x_c[k]]
    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height2[y_c[k], x_c[k]]
            offset_y[y_c[k], x_c[k]] = offset_y2[y_c[k], x_c[k]]
            offset_x[y_c[k], x_c[k]] = offset_x2[y_c[k], x_c[k]]


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms2maxii(Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]
    semant2 = Y[3][0, :, :, 0]
    height2 = Y[4][0, :, :, 0]
    semants = Y[5][0, :, :, 0]

    # plt.imshow(semants)
    # plt.show()

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)

    t = [semant1,semant2]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height1[y_c[k], x_c[k]]
    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height2[y_c[k], x_c[k]]



    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms2maxscore(Ys,Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    semant2 = Y[2][0, :, :, 0]
    height2 = Y[3][0, :, :, 0]

    semants = Ys[0][0, :, :, 0]
    offset_y = Ys[2][0, :, :, 0]
    offset_x = Ys[2][0, :, :, 1]
    # plt.imshow(semants)
    # plt.show()

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)

    t = [semant1,semant2]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height1[y_c[k], x_c[k]]
    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height2[y_c[k], x_c[k]]



    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms3maxscore(Ys,Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    semant2 = Y[2][0, :, :, 0]
    height2 = Y[3][0, :, :, 0]
    semant3 = Y[4][0, :, :, 0]
    height3 = Y[5][0, :, :, 0]


    semants = Ys[0][0, :, :, 0]
    offset_y = Ys[2][0, :, :, 0]
    offset_x = Ys[2][0, :, :, 1]
    # plt.imshow(semants)
    # plt.show()

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    t = [semant1,semant2,semant3]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = np.exp(height3[y_c[k], x_c[k]])* 4


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs
def parse_det_offset_ms3maxscorec(Ys,Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    semant2 = Y[0][0, :, :, 1]
    height2 = Y[1][0, :, :, 1]
    semant3 = Y[0][0, :, :, 2]
    height3 = Y[1][0, :, :, 2]


    semants = Ys[0][0, :, :, 0]
    offset_y = Ys[2][0, :, :, 0]
    offset_x = Ys[2][0, :, :, 1]
    # plt.imshow(semants)
    # plt.show()
    scio.savemat('/media/xwj/Day/dominikandreaCSP/eval_caltech/1.mat',
                 {'semant1': semant1, 'semant2': semant2, 'semant3': semant3, 'semants': semants})
    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    t = [semant1,semant2,semant3]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = np.exp(height3[y_c[k], x_c[k]])* 4


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms3maxscorect(Ys,Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    semant2 = Y[0][0, :, :, 1]
    height2 = Y[1][0, :, :, 1]
    semant3 = Y[0][0, :, :, 2]
    height3 = Y[1][0, :, :, 2]


    semants = Ys[0][0, :, :, 0]
    offset_y = Ys[3][0, :, :, 0]
    offset_x = Ys[3][0, :, :, 1]
    # plt.imshow(semants)
    # plt.show()
    # scio.savemat('/media/xwj/Day/dominikandreaCSP/eval_caltech/1.mat',
    #              {'semant1': semant1, 'semant2': semant2, 'semant3': semant3, 'semants': semants})
    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    t = [semant1,semant2,semant3]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):

            if semants[y_c[k], x_c[k]]> score and semant1[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0

    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant2[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0
    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant3[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height3[y_c[k], x_c[k]])* 4
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]]=0
    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_h(Ys,Y, C, score=0.1,down=4):
    height = Y[0, :, :, 0]

    seman = Ys[0][0, :, :, 0]
    offset_y = Ys[2][0, :, :, 0]
    offset_x = Ys[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down

            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs



def parse_det_offset_ms3maxanchorct(Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 1]
    height1 = Y[1][0, :, :, 0]
    semant2 = Y[0][0, :, :, 2]
    height2 = Y[1][0, :, :, 1]
    semant3 = Y[0][0, :, :, 3]
    height3 = Y[1][0, :, :, 2]


    semants = Y[0][0, :, :, 0]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    t = [semant1,semant2,semant3]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):

            if semants[y_c[k], x_c[k]]> score and semant1[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0

    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant2[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0
    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant3[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height3[y_c[k], x_c[k]])* 4
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]]=0
    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms2losscatct(Ys,Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    semant2 = Y[0][0, :, :, 1]
    height2 = Y[1][0, :, :, 1]



    semants = Ys[0][0, :, :, 0]
    offset_y = Ys[2][0, :, :, 0]
    offset_x = Ys[2][0, :, :, 1]

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)

    t = [semant1,semant2]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):

            if semants[y_c[k], x_c[k]]> score and semant1[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0

    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant2[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0

    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs


def parse_det_offset_ms4maxcls(Ycsp,Y, C, score=0.1,down=4):

    semants = Ycsp[0][0, :, :, 0]

    semant1 = Y[1][0, :, :, 0]
    height1 = Y[2][0, :, :, 0]
    semant2 = Y[1][0, :, :, 1]
    height2 = Y[2][0, :, :, 1]
    semant3 = Y[1][0, :, :, 2]
    height3 = Y[2][0, :, :, 2]
    semant4 = Y[1][0, :, :, 3]
    height4 = Y[2][0, :, :, 3]



    offset_y = Y[3][0, :, :, 0]
    offset_x = Y[3][0, :, :, 1]



    # plt.imshow(semants)
    # plt.show()
    # scio.savemat('/media/xwj/Data/dominikandreaCSPcity/eval_city/1.mat',
    #              {'semant1': semant1, 'semant2': semant2, 'semant3': semant3, 'semants': semants})
    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    semant4 = np.array(semant4)
    t = [semant1,semant2,semant3,semant4]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):

            if semants[y_c[k], x_c[k]]> score and semant1[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0

    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant2[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0
    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant3[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height3[y_c[k], x_c[k]])* 4
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]]=0

    y_c, x_c = np.where(indext ==3)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant4[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height4[y_c[k], x_c[k]])* 8
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]]=0



    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs




def parse_det_offset_ms3maxcls(Y, C, score=0.1,down=4):

    semants = Y[0][0, :, :, 0]

    semant1 = Y[1][0, :, :, 0]
    height1 = Y[2][0, :, :, 0]
    semant2 = Y[1][0, :, :, 1]
    height2 = Y[2][0, :, :, 1]
    semant3 = Y[1][0, :, :, 2]
    height3 = Y[2][0, :, :, 2]



    offset_y = Y[3][0, :, :, 0]
    offset_x = Y[3][0, :, :, 1]



    # plt.imshow(semants)
    # plt.show()
    # scio.savemat('/media/xwj/Day/dominikandreaCSP/eval_caltech/1.mat',
    #              {'semant1': semant1, 'semant2': semant2, 'semant3': semant3, 'semants': semants})
    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    t = [semant1,semant2,semant3]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):

            if semants[y_c[k], x_c[k]]> score and semant1[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0

    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant2[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0
    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant3[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height3[y_c[k], x_c[k]])* 4
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]]=0
    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs



def parse_det_offset_ms2maxcls(Y, C, score=0.1,down=4):

    semants = Y[0][0, :, :, 0]

    semant1 = Y[1][0, :, :, 0]
    height1 = Y[2][0, :, :, 0]
    semant2 = Y[1][0, :, :, 1]
    height2 = Y[2][0, :, :, 1]



    offset_y = Y[3][0, :, :, 0]
    offset_x = Y[3][0, :, :, 1]



    # plt.imshow(semants)
    # plt.show()
    # scio.savemat('/media/xwj/Day/dominikandreaCSP/eval_caltech/1.mat',
    #              {'semant1': semant1, 'semant2': semant2, 'semant3': semant3, 'semants': semants})
    semant1 = np.array(semant1)
    semant2 = np.array(semant2)

    t = [semant1,semant2]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):

            if semants[y_c[k], x_c[k]]> score and semant1[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0

    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            if semants[y_c[k], x_c[k]]> score and semant2[y_c[k], x_c[k]]> score:
                seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
                height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
            else:
                seman[y_c[k], x_c[k]] = 0
                height[y_c[k], x_c[k]] = 0

    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs



def parse_det_offset_ms3w(Ys,Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    semant2 = Y[0][0, :, :, 1]
    height2 = Y[1][0, :, :, 1]
    semant3 = Y[0][0, :, :, 2]
    height3 = Y[1][0, :, :, 2]


    semants = Ys[0][0, :, :, 0]
    offset_y = Ys[2][0, :, :, 0]
    offset_x = Ys[2][0, :, :, 1]
    # plt.imshow(semants)
    # plt.show()

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    t = [semant1,semant2,semant3]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = np.exp(height1[y_c[k], x_c[k]])
    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = np.exp(height2[y_c[k], x_c[k]])* 2
    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = np.exp(height3[y_c[k], x_c[k]])* 4


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs



def parse_det_offset_ms3maxscoreh(Ys,Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    semant2 = Y[2][0, :, :, 0]
    height2 = Y[3][0, :, :, 0]
    semant3 = Y[4][0, :, :, 0]
    height3 = Y[5][0, :, :, 0]


    semants = Ys[0][0, :, :, 0]
    offset_y = Ys[2][0, :, :, 0]
    offset_x = Ys[2][0, :, :, 1]
    # plt.imshow(semants)
    # plt.show()

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    height1 = np.exp(height1)* down
    height2 = np.exp(height2)* 2* down
    height3 = np.exp(height3)* 4* down

    scio.savemat('/media/xwj/Day/dominikandreaCSP/eval_caltech/1.mat',
                 {'semant1': height1, 'semant2': height2, 'semant3': height3, 'semants': semants})
    y_c, x_c = np.where(height1 > 112)
    height1[y_c, x_c] = 0


    t = [height1,height2,height3]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))

    y_c, x_c = np.where(indext ==0)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] = height1[y_c[k], x_c[k]]
    y_c, x_c = np.where(indext ==1)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] =  height2[y_c[k], x_c[k]]
    y_c, x_c = np.where(indext ==2)
    if len(y_c) > 0:
        for k in range(len(y_c)):
            seman[y_c[k], x_c[k]] = semants[y_c[k], x_c[k]]
            height[y_c[k], x_c[k]] =  height3[y_c[k], x_c[k]]


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            #  h = np.exp(height[y_c[i], x_c[i]]) * down
            h = height[y_c[i], x_c[i]]
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs


def parse_det_offset_ms3maxi(Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    offset_y1 = Y[2][0, :, :, 0]
    offset_x1 = Y[2][0, :, :, 1]
    semant2 = Y[3][0, :, :, 0]
    height2 = Y[4][0, :, :, 0]
    offset_y2 = Y[5][0, :, :, 0]
    offset_x2 = Y[5][0, :, :, 1]


    semant3 = Y[6][0, :, :, 0]
    height3 = Y[7][0, :, :, 0]
    offset_y3 = Y[8][0, :, :, 0]
    offset_x3 = Y[8][0, :, :, 1]
    semants = Y[9][0, :, :, 0]

    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_y = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_x = np.zeros((semant1.shape[0], semant1.shape[1]))
    for i in range(semant1.shape[0]):
        for j in range(semant1.shape[1]):
            s = [semant1[i,j],semant2[i,j],semant3[i,j]]
            index = np.argmax(s)
            if index ==0:
                seman[i, j] = semants[i,j]
                height[i, j] = height1[i, j]
                offset_y[i, j] = offset_y1[i,j]
                offset_x[i, j] = offset_x1[i, j]
            elif index ==1:
                seman[i, j] = semants[i,j]
                height[i, j] = height2[i, j]
                offset_y[i, j] = offset_y2[i, j]
                offset_x[i, j] = offset_x2[i, j]
            else:
                seman[i, j] = semants[i,j]
                height[i, j] = height3[i, j]
                offset_y[i, j] = offset_y3[i, j]
                offset_x[i, j] = offset_x3[i, j]


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs

def parse_det_offset_ms3maii(Y, C, score=0.1,down=4):
    semant1 = Y[0][0, :, :, 0]
    height1 = Y[1][0, :, :, 0]
    offset_y1 = Y[2][0, :, :, 0]
    offset_x1 = Y[2][0, :, :, 1]
    semant2 = Y[3][0, :, :, 0]
    height2 = Y[4][0, :, :, 0]
    offset_y2 = Y[5][0, :, :, 0]
    offset_x2 = Y[5][0, :, :, 1]


    semant3 = Y[6][0, :, :, 0]
    height3 = Y[7][0, :, :, 0]
    offset_y3 = Y[8][0, :, :, 0]
    offset_x3 = Y[8][0, :, :, 1]
    semants = Y[9][0, :, :, 0]

    # plt.imshow(semants)
    # plt.show()

    semant1 = np.array(semant1)
    semant2 = np.array(semant2)
    semant3 = np.array(semant3)
    t = [semant1,semant2,semant3]
    indext = np.argmax(t,0)
    seman = np.zeros((semant1.shape[0], semant1.shape[1]))
    height = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_y = np.zeros((semant1.shape[0], semant1.shape[1]))
    offset_x = np.zeros((semant1.shape[0], semant1.shape[1]))
    for i in range(semant1.shape[0]):
        for j in range(semant1.shape[1]):
            # s = [semant1[i,j],semant2[i,j],semant3[i,j]]
            # index = np.argmax(s)
            index = indext[i,j]
            if index ==0:
                seman[i, j] = semants[i,j]
                height[i, j] = height1[i, j]
                offset_y[i, j] = offset_y1[i,j]
                offset_x[i, j] = offset_x1[i, j]
            elif index ==1:
                seman[i, j] = semants[i,j]
                height[i, j] = height2[i, j]
                offset_y[i, j] = offset_y2[i, j]
                offset_x[i, j] = offset_x2[i, j]
            else:
                seman[i, j] = semants[i,j]
                height[i, j] = height3[i, j]
                offset_y[i, j] = offset_y3[i, j]
                offset_x[i, j] = offset_x3[i, j]


    y_c, x_c = np.where(seman > score)

    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # h = height[y_c[i], x_c[i]] * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            # x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y ) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = nms(boxs, 0.5, usegpu=False, gpu_id=0)
        boxs = boxs[keep, :]
    return boxs


def parse_wider_offset(Y, C, score=0.1, down=4, nmsthre=0.5):
    seman = Y[0][0, :, :, 0]
    height = Y[1][0, :, :, 0]
    width = Y[1][0, :, :, 1]
    offset_y = Y[2][0, :, :, 0]
    offset_x = Y[2][0, :, :, 1]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = np.exp(width[y_c[i], x_c[i]]) * down
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            x1, y1 = min(x1, C.size_test[1]), min(y1, C.size_test[0])
            boxs.append([x1, y1, min(x1 + w, C.size_test[1]), min(y1 + h, C.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        # keep = nms(boxs, nmsthre, usegpu=False, gpu_id=0)
        # boxs = boxs[keep, :]
        boxs = soft_bbox_vote(boxs, thre=nmsthre)
    return boxs


def soft_bbox_vote(det, thre=0.35, score=0.05):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= thre)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= score)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets


def bbox_vote(det, thre):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    # det = det[np.where(det[:, 4] > 0.2)[0], :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= thre)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    return dets
