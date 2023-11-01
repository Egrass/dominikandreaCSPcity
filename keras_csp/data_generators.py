# import numpy as np
# import cv2
import random
from . import data_augment
from .bbox_transform import *


def calc_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    if scale == 'hw':
        scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1
    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
            seman_map[y1:y2, x1:x2, 1] = 1
            seman_map[c_y, c_x, 2] = 1

            if scale == 'h':
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1
            elif scale == 'w':
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1
            elif scale == 'hw':
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = 1
            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1

    if offset:
        return seman_map, scale_map, offset_map
    else:
        return seman_map, scale_map

def calc_gt_center_cls_offset(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])


    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1
    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
            seman_map[y1:y2, x1:x2, 1] = 1
            seman_map[c_y, c_x, 2] = 1


            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1

    if offset:
        return seman_map,offset_map



def calc_ms3_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))

    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        offset_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        offset_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    seman_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map1[:, :, 1] = 1

    seman_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map2[:, :, 1] = 1

    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            seman_map1[y1:y2, x1:x2, 1] = 0
            seman_map2[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))

            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1


                if offset:
                    offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map[c_y, c_x, 2] = 1
            elif h >= 112 / down*ratio and h <= 224 / down*ratio:
                seman_map1[y1:y2, x1:x2, 0] = np.maximum(seman_map1[y1:y2, x1:x2, 0], gau_map)
                seman_map1[y1:y2, x1:x2, 1] = 1
                seman_map1[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

                if offset:
                    offset_map1[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map1[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map1[c_y, c_x, 2] = 1
            else:
                seman_map2[y1:y2, x1:x2, 0] = np.maximum(seman_map2[y1:y2, x1:x2, 0], gau_map)
                seman_map2[y1:y2, x1:x2, 1] = 1
                seman_map2[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

                if offset:
                    offset_map2[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map2[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map2[c_y, c_x, 2] = 1

    if offset:
        return seman_map, scale_map, offset_map,seman_map1,scale_map1,offset_map1,seman_map2,scale_map2,offset_map2
    else:
        return seman_map, scale_map

def calc_ms3anchor_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))


    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    seman_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map1[:, :, 1] = 1

    seman_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map2[:, :, 1] = 1

    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            seman_map1[y1:y2, x1:x2, 1] = 0
            seman_map2[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))

            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1
                t = 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1])/t)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1


            elif h >= 112 / down*ratio and h <= 224 / down*ratio:
                seman_map1[y1:y2, x1:x2, 0] = np.maximum(seman_map1[y1:y2, x1:x2, 0], gau_map)
                seman_map1[y1:y2, x1:x2, 1] = 1
                seman_map1[c_y, c_x, 2] = 1
                t = 60/ down * ratio
                if scale == 'h':
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1])/t)
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1


            else:
                seman_map2[y1:y2, x1:x2, 0] = np.maximum(seman_map2[y1:y2, x1:x2, 0], gau_map)
                seman_map2[y1:y2, x1:x2, 1] = 1
                seman_map2[c_y, c_x, 2] = 1
                t = 120 / down * ratio
                if scale == 'h':
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1])/t)
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1



    if offset:
        return seman_map, scale_map,seman_map1,scale_map1,seman_map2,scale_map2
    else:
        return seman_map, scale_map

def calc_ms3anchor1_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))


    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    seman_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map1[:, :, 1] = 1

    seman_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map2[:, :, 1] = 1

    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            seman_map1[y1:y2, x1:x2, 1] = 0
            seman_map2[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))

            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1]))
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1


            elif h >= 112 / down*ratio and h <= 224 / down*ratio:
                seman_map1[y1:y2, x1:x2, 0] = np.maximum(seman_map1[y1:y2, x1:x2, 0], gau_map)
                seman_map1[y1:y2, x1:x2, 1] = 1
                seman_map1[c_y, c_x, 2] = 1
                #  t = -112 / down * ratio  + 30 / down * ratio
                if scale == 'h':
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1])/2)
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1
            else:
                seman_map2[y1:y2, x1:x2, 0] = np.maximum(seman_map2[y1:y2, x1:x2, 0], gau_map)
                seman_map2[y1:y2, x1:x2, 1] = 1
                seman_map2[c_y, c_x, 2] = 1
               # t = -224 / down * ratio+ 30 / down * ratio
                if scale == 'h':
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1])/4)
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1



    if offset:
        return seman_map, scale_map,seman_map1,scale_map1,seman_map2,scale_map2
    else:
        return seman_map, scale_map

def calc_ms3anchor1c_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 6))

    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 9))
    seman_map[:, :,  3:6] = 1


    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 3:6] = 0

    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))

            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 3] = 1
                seman_map[c_y, c_x, 6] = 1
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1]))
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 3] = 1


            elif h >= 112 / down*ratio and h <= 224 / down*ratio:
                seman_map[y1:y2, x1:x2, 1] = np.maximum(seman_map[y1:y2, x1:x2, 1], gau_map)
                seman_map[y1:y2, x1:x2, 4] = 1
                seman_map[c_y, c_x, 7] = 1
                #  t = -112 / down * ratio  + 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = np.log((gts[ind, 3] - gts[ind, 1])/2)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 4] = 1
            else:
                seman_map[y1:y2, x1:x2, 2] = np.maximum(seman_map[y1:y2, x1:x2, 2], gau_map)
                seman_map[y1:y2, x1:x2, 5] = 1
                seman_map[c_y, c_x, 8] = 1
               # t = -224 / down * ratio+ 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = np.log((gts[ind, 3] - gts[ind, 1])/4)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 5] = 1



    if offset:
        return seman_map, scale_map
    else:
        return seman_map, scale_map

def calc_ms3anchor1cls_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 6))
    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 9))
    seman_map[:, :,  3:6] = 1

    seman_mapc = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_mapc[:, :, 1] = 1

    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 3:6] = 0
            seman_mapc[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_mapc[y1:y2, x1:x2, 0] = np.maximum(seman_mapc[y1:y2, x1:x2, 0], gau_map)
            seman_mapc[y1:y2, x1:x2, 1] = 1
            seman_mapc[c_y, c_x, 2] = 1
            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 3] = 1
                seman_map[c_y, c_x, 6] = 1
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1]))
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 3] = 1


            elif h >= 112 / down*ratio and h <= 224 / down*ratio:
                seman_map[y1:y2, x1:x2, 1] = np.maximum(seman_map[y1:y2, x1:x2, 1], gau_map)
                seman_map[y1:y2, x1:x2, 4] = 1
                seman_map[c_y, c_x, 7] = 1
                #  t = -112 / down * ratio  + 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = np.log((gts[ind, 3] - gts[ind, 1])/2)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 4] = 1
            else:
                seman_map[y1:y2, x1:x2, 2] = np.maximum(seman_map[y1:y2, x1:x2, 2], gau_map)
                seman_map[y1:y2, x1:x2, 5] = 1
                seman_map[c_y, c_x, 8] = 1
               # t = -224 / down * ratio+ 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = np.log((gts[ind, 3] - gts[ind, 1])/4)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 5] = 1

            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1

    if offset:
        return seman_mapc, seman_map, scale_map,offset_map
    else:
        return seman_map, scale_map

def calc_ms4anchor1cls_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 8))
    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 12))
    seman_map[:, :,  4:8] = 1

    seman_mapc = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_mapc[:, :, 1] = 1

    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 4:8] = 0
            seman_mapc[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_mapc[y1:y2, x1:x2, 0] = np.maximum(seman_mapc[y1:y2, x1:x2, 0], gau_map)
            seman_mapc[y1:y2, x1:x2, 1] = 1
            seman_mapc[c_y, c_x, 2] = 1
            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 4] = 1
                seman_map[c_y, c_x, 8] = 1
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1]))
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 4] = 1


            elif h >= 112 / down*ratio and h <= 224 / down*ratio:
                seman_map[y1:y2, x1:x2, 1] = np.maximum(seman_map[y1:y2, x1:x2, 1], gau_map)
                seman_map[y1:y2, x1:x2, 5] = 1
                seman_map[c_y, c_x, 9] = 1
                #  t = -112 / down * ratio  + 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = np.log((gts[ind, 3] - gts[ind, 1])/2)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 5] = 1
            elif h >= 224 / down*ratio and h <= 448 / down*ratio:
                seman_map[y1:y2, x1:x2, 2] = np.maximum(seman_map[y1:y2, x1:x2, 1], gau_map)
                seman_map[y1:y2, x1:x2, 6] = 1
                seman_map[c_y, c_x, 10] = 1
                #  t = -112 / down * ratio  + 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = np.log((gts[ind, 3] - gts[ind, 1])/4)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 6] = 1
            else:
                seman_map[y1:y2, x1:x2, 3] = np.maximum(seman_map[y1:y2, x1:x2, 2], gau_map)
                seman_map[y1:y2, x1:x2, 7] = 1
                seman_map[c_y, c_x, 11] = 1
               # t = -224 / down * ratio+ 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 3] = np.log((gts[ind, 3] - gts[ind, 1])/8)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 7] = 1

            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1

    if offset:
        return seman_mapc, seman_map, scale_map,offset_map
    else:
        return seman_map, scale_map



def calc_ms2losscat_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 4))

    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 6))
    seman_map[:, :,  2:3] = 1


    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2,  2:3] = 0

    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))

            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 2] = 1
                seman_map[c_y, c_x, 4] = 1
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1]))
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = 1


            else:
                seman_map[y1:y2, x1:x2, 1] = np.maximum(seman_map[y1:y2, x1:x2, 1], gau_map)
                seman_map[y1:y2, x1:x2, 3] = 1
                seman_map[c_y, c_x, 5] = 1
                #  t = -112 / down * ratio  + 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = np.log((gts[ind, 3] - gts[ind, 1])/2)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 3] = 1




    if offset:
        return seman_map, scale_map
    else:
        return seman_map, scale_map


def calc_ms2loss_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 4))
    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 6))
    seman_map[:, :,  2:3] = 1

    seman_mapc = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_mapc[:, :, 1] = 1
    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2,  2:3] = 0
            seman_mapc[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_mapc[y1:y2, x1:x2, 0] = np.maximum(seman_mapc[y1:y2, x1:x2, 0], gau_map)
            seman_mapc[y1:y2, x1:x2, 1] = 1
            seman_mapc[c_y, c_x, 2] = 1
            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 2] = 1
                seman_map[c_y, c_x, 4] = 1
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1]))
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = 1


            else:
                seman_map[y1:y2, x1:x2, 1] = np.maximum(seman_map[y1:y2, x1:x2, 1], gau_map)
                seman_map[y1:y2, x1:x2, 3] = 1
                seman_map[c_y, c_x, 5] = 1
                #  t = -112 / down * ratio  + 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = np.log((gts[ind, 3] - gts[ind, 1])/2)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 3] = 1

            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1


    if offset:
        return seman_mapc,seman_map, scale_map,offset_map
    else:
        return seman_map, scale_map





def calc_ms3anchor1cc_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 6))
    offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 12))
    seman_map[:, :,  4:8] = 1


    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 4:8] = 0

    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))

            offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
            offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
            offset_map[c_y, c_x, 2] = 1
            h = y2 - y1 + 1

            seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
            seman_map[y1:y2, x1:x2, 4] = 1
            seman_map[c_y, c_x, 8] = 1

            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 1] = np.maximum(seman_map[y1:y2, x1:x2, 1], gau_map)
                seman_map[y1:y2, x1:x2, 5] = 1
                seman_map[c_y, c_x, 9] = 1
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log((gts[ind, 3] - gts[ind, 1]))
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 3] = 1


            elif h >= 112 / down*ratio and h <= 224 / down*ratio:
                seman_map[y1:y2, x1:x2, 2] = np.maximum(seman_map[y1:y2, x1:x2, 2], gau_map)
                seman_map[y1:y2, x1:x2, 6] = 1
                seman_map[c_y, c_x, 10] = 1
                #  t = -112 / down * ratio  + 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = np.log((gts[ind, 3] - gts[ind, 1])/2)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 4] = 1
            else:
                seman_map[y1:y2, x1:x2, 3] = np.maximum(seman_map[y1:y2, x1:x2, 3], gau_map)
                seman_map[y1:y2, x1:x2, 7] = 1
                seman_map[c_y, c_x, 11] = 1
               # t = -224 / down * ratio+ 30 / down * ratio
                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = np.log((gts[ind, 3] - gts[ind, 1])/4)
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 5] = 1



    if offset:
        return seman_map, scale_map,offset_map
    else:
        return seman_map, scale_map




def calc_ms2anchor_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))

    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    seman_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map1[:, :, 1] = 1



    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            seman_map1[y1:y2, x1:x2, 1] = 0

    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))

            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1


            else:
                seman_map1[y1:y2, x1:x2, 0] = np.maximum(seman_map1[y1:y2, x1:x2, 0], gau_map)
                seman_map1[y1:y2, x1:x2, 1] = 1
                seman_map1[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1




    if offset:
        return seman_map, scale_map,seman_map1,scale_map1
    else:
        return seman_map, scale_map



def calc_ms3i_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))

    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        offset_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        offset_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))

    seman_mapa = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_mapa[:, :, 1] = 1

    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    seman_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map1[:, :, 1] = 1

    seman_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map2[:, :, 1] = 1

    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            seman_map1[y1:y2, x1:x2, 1] = 0
            seman_map2[y1:y2, x1:x2, 1] = 0
            seman_mapa[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_mapa[y1:y2, x1:x2, 0] = np.maximum(seman_mapa[y1:y2, x1:x2, 0], gau_map)
            seman_mapa[y1:y2, x1:x2, 1] = 1
            seman_mapa[c_y, c_x, 2] = 1
            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1


                if offset:
                    offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map[c_y, c_x, 2] = 1
            elif h >= 112 / down*ratio and h <= 224 / down*ratio:
                seman_map1[y1:y2, x1:x2, 0] = np.maximum(seman_map1[y1:y2, x1:x2, 0], gau_map)
                seman_map1[y1:y2, x1:x2, 1] = 1
                seman_map1[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

                if offset:
                    offset_map1[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map1[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map1[c_y, c_x, 2] = 1
            else:
                seman_map2[y1:y2, x1:x2, 0] = np.maximum(seman_map2[y1:y2, x1:x2, 0], gau_map)
                seman_map2[y1:y2, x1:x2, 1] = 1
                seman_map2[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

                if offset:
                    offset_map2[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map2[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map2[c_y, c_x, 2] = 1

    if offset:
        return seman_map, scale_map, offset_map,seman_map1,scale_map1,offset_map1,seman_map2,scale_map2,offset_map2,seman_mapa
    else:
        return seman_map, scale_map



def calc_ms3ii_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))

    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))


    seman_mapa = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_mapa[:, :, 1] = 1

    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    seman_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map1[:, :, 1] = 1

    seman_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map2[:, :, 1] = 1

    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            seman_map1[y1:y2, x1:x2, 1] = 0
            seman_map2[y1:y2, x1:x2, 1] = 0
            seman_mapa[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_mapa[y1:y2, x1:x2, 0] = np.maximum(seman_mapa[y1:y2, x1:x2, 0], gau_map)
            seman_mapa[y1:y2, x1:x2, 1] = 1
            seman_mapa[c_y, c_x, 2] = 1
            h = y2 - y1 + 1
            # print h

            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1

            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1


            elif h >= 112 / down*ratio and h <= 224 / down*ratio:
                seman_map1[y1:y2, x1:x2, 0] = np.maximum(seman_map1[y1:y2, x1:x2, 0], gau_map)
                seman_map1[y1:y2, x1:x2, 1] = 1
                seman_map1[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

            else:
                seman_map2[y1:y2, x1:x2, 0] = np.maximum(seman_map2[y1:y2, x1:x2, 0], gau_map)
                seman_map2[y1:y2, x1:x2, 1] = 1
                seman_map2[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

    if offset:
        return seman_map, scale_map, offset_map,seman_map1,scale_map1,seman_map2,scale_map2,seman_mapa
    else:
        return seman_map, scale_map


def calc_ms2i_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))

    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        offset_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))


    seman_mapa = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_mapa[:, :, 1] = 1

    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    seman_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map1[:, :, 1] = 1


    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            seman_map1[y1:y2, x1:x2, 1] = 0
            seman_mapa[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_mapa[y1:y2, x1:x2, 0] = np.maximum(seman_mapa[y1:y2, x1:x2, 0], gau_map)
            seman_mapa[y1:y2, x1:x2, 1] = 1
            seman_mapa[c_y, c_x, 2] = 1
            h = y2 - y1 + 1
            # print h
            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1


                if offset:
                    offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map[c_y, c_x, 2] = 1
            else:
                seman_map1[y1:y2, x1:x2, 0] = np.maximum(seman_map1[y1:y2, x1:x2, 0], gau_map)
                seman_map1[y1:y2, x1:x2, 1] = 1
                seman_map1[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

                if offset:
                    offset_map1[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map1[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map1[c_y, c_x, 2] = 1

    if offset:
        return seman_map, scale_map, offset_map,seman_map1,scale_map1,offset_map1,seman_mapa
    else:
        return seman_map, scale_map


def calc_ms2ii_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))

    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))


    seman_mapa = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_mapa[:, :, 1] = 1

    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    seman_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map1[:, :, 1] = 1


    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            seman_map1[y1:y2, x1:x2, 1] = 0
            seman_mapa[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_mapa[y1:y2, x1:x2, 0] = np.maximum(seman_mapa[y1:y2, x1:x2, 0], gau_map)
            seman_mapa[y1:y2, x1:x2, 1] = 1
            seman_mapa[c_y, c_x, 2] = 1
            h = y2 - y1 + 1
            # print h
            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1

            if h < 112 / down *ratio:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1
            else:
                seman_map1[y1:y2, x1:x2, 0] = np.maximum(seman_map1[y1:y2, x1:x2, 0], gau_map)
                seman_map1[y1:y2, x1:x2, 1] = 1
                seman_map1[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

    if offset:
        return seman_map, scale_map, offset_map,seman_map1,scale_map1,seman_mapa
    else:
        return seman_map, scale_map



def calc_ms30i_gt_center(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    ratio = img_data['ratio']
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    scale_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))

    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        offset_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
        offset_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))

    seman_mapa = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_mapa[:, :, 1] = 1

    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    seman_map1 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map1[:, :, 1] = 1

    seman_map2 = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map2[:, :, 1] = 1

    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            seman_map1[y1:y2, x1:x2, 1] = 0
            seman_map2[y1:y2, x1:x2, 1] = 0
            seman_mapa[y1:y2, x1:x2, 1] = 0

    t = 112 / down * ratio
    t1 = 224 / down * ratio
    tt = 40 / down * ratio
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_mapa[y1:y2, x1:x2, 0] = np.maximum(seman_mapa[y1:y2, x1:x2, 0], gau_map)
            seman_mapa[y1:y2, x1:x2, 1] = 1
            seman_mapa[c_y, c_x, 2] = 1
            h = y2 - y1 + 1
            # print h
            if h < t:
                seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
                seman_map[y1:y2, x1:x2, 1] = 1
                seman_map[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                    scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1


                if offset:
                    offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map[c_y, c_x, 2] = 1
            elif h >= t and h <= t1:
                seman_map1[y1:y2, x1:x2, 0] = np.maximum(seman_map1[y1:y2, x1:x2, 0], gau_map)
                seman_map1[y1:y2, x1:x2, 1] = 1
                seman_map1[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1] - t+tt)
                    scale_map1[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

                if offset:
                    offset_map1[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map1[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map1[c_y, c_x, 2] = 1
            else:
                seman_map2[y1:y2, x1:x2, 0] = np.maximum(seman_map2[y1:y2, x1:x2, 0], gau_map)
                seman_map2[y1:y2, x1:x2, 1] = 1
                seman_map2[c_y, c_x, 2] = 1

                if scale == 'h':
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1] - t1+tt)
                    scale_map2[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

                if offset:
                    offset_map2[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                    offset_map2[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                    offset_map2[c_y, c_x, 2] = 1

    if offset:
        return seman_map, scale_map, offset_map,seman_map1,scale_map1,offset_map1,seman_map2,scale_map2,offset_map2,seman_mapa
    else:
        return seman_map, scale_map



def calc_gt_center_rec(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    if scale == 'hw':
        scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    rec_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 1))

    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
            rec_map[y1:y2, x1:x2, 0] = 1
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            rec_map[y1:y2, x1:x2, 0] = 1

            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
            seman_map[y1:y2, x1:x2, 1] = 1
            seman_map[c_y, c_x, 2] = 1


            if scale == 'h':
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1

    if offset:
        return seman_map, scale_map, offset_map,rec_map
    else:
        return seman_map, scale_map


def calc_gt_center_rec45(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    if scale == 'hw':
        scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1

    rec_map = np.zeros((int(C.size_train[0] / 16), int(C.size_train[1] / 16), 1))
    if len(igs) > 0:
        igs8 = igs / 16
        for ind in range(len(igs8)):
            x1, y1, x2, y2 = int(igs8[ind, 0]), int(igs8[ind, 1]), int(np.ceil(igs8[ind, 2])), int(np.ceil(igs8[ind, 3]))
            rec_map[y1:y2, x1:x2, 0] = 1
    if len(gts) > 0:
        gts8 = gts / 16
        for ind in range(len(gts8)):
            x1, y1, x2, y2 = int(np.ceil(gts8[ind, 0])), int(np.ceil(gts8[ind, 1])), int(gts8[ind, 2]), int(gts8[ind, 3])
            rec_map[y1:y2, x1:x2, 0] = 1


    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])

            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
            seman_map[y1:y2, x1:x2, 1] = 1
            seman_map[c_y, c_x, 2] = 1


            if scale == 'h':
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1

            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1

    if offset:
        return seman_map, scale_map, offset_map,rec_map
    else:
        return seman_map, scale_map


def calc_gt_center_giou(C, img_data, r=2, down=4, scale='h', offset=True):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])

    xyxy_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 9), dtype=np.float32)

    for c in range(int(C.size_train[0] / down)):
        xyxy_map[c, :, 5] = c + 0.5
        xyxy_map[c, :, 7] = c + 0.5
    for rr in range(int(C.size_train[1] / down)):
        xyxy_map[:, rr, 6] = rr + 0.5
        xyxy_map[:, rr, 8] = rr + 0.5


    scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 2))
    if scale == 'hw':
        scale_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    if offset:
        offset_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map = np.zeros((int(C.size_train[0] / down), int(C.size_train[1] / down), 3))
    seman_map[:, :, 1] = 1
    if len(igs) > 0:
        igs = igs / down
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / down
        for ind in range(len(gts)):
            # x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(round(gts[ind, 3]))
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            c_x, c_y = int((gts[ind, 0] + gts[ind, 2]) / 2), int((gts[ind, 1] + gts[ind, 3]) / 2)
            dx = gaussian(x2 - x1)
            dy = gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            seman_map[y1:y2, x1:x2, 0] = np.maximum(seman_map[y1:y2, x1:x2, 0], gau_map)
            seman_map[y1:y2, x1:x2, 1] = 1
            seman_map[c_y, c_x, 2] = 1

            xyxy_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = x1
            xyxy_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = y1
            xyxy_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = x2
            xyxy_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 3] = y2
            xyxy_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 4] = 1

            if scale == 'h':
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1
            elif scale == 'w':
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 2] - gts[ind, 0])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = 1
            elif scale == 'hw':
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 1] = np.log(gts[ind, 2] - gts[ind, 0])
                scale_map[c_y - r:c_y + r + 1, c_x - r:c_x + r + 1, 2] = 1
            if offset:
                offset_map[c_y, c_x, 0] = (gts[ind, 1] + gts[ind, 3]) / 2 - c_y - 0.5
                offset_map[c_y, c_x, 1] = (gts[ind, 0] + gts[ind, 2]) / 2 - c_x - 0.5
                offset_map[c_y, c_x, 2] = 1

    if offset:
        return seman_map, scale_map, offset_map,xyxy_map
    else:
        return seman_map, scale_map


def calc_gt_top(C, img_data, r=2):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])

    scale_map = np.zeros((int(C.size_train[0] / 4), int(C.size_train[1] / 4), 2))
    seman_map = np.zeros((int(C.size_train[0] / 4), int(C.size_train[1] / 4), 3))
    seman_map[:, :, 1] = 1
    if len(igs) > 0:
        igs = igs / 4
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / 4
        for ind in range(len(gts)):
            x1, y1, x2, y2 = int(round(gts[ind, 0])), int(round(gts[ind, 1])), int(round(gts[ind, 2])), int(
                round(gts[ind, 3]))
            w = x2 - x1
            c_x = int((gts[ind, 0] + gts[ind, 2]) / 2)

            dx = gaussian(w)
            dy = gaussian(w)
            gau_map = np.multiply(dy, np.transpose(dx))

            ty = np.maximum(0, int(round(y1 - w / 2)))
            ot = ty - int(round(y1 - w / 2))
            seman_map[ty:ty + w - ot, x1:x2, 0] = np.maximum(seman_map[ty:ty + w - ot, x1:x2, 0], gau_map[ot:, :])
            seman_map[ty:ty + w - ot, x1:x2, 1] = 1
            seman_map[y1, c_x, 2] = 1

            scale_map[y1 - r:y1 + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
            scale_map[y1 - r:y1 + r + 1, c_x - r:c_x + r + 1, 1] = 1
    return seman_map, scale_map


def calc_gt_bottom(C, img_data, r=2):
    def gaussian(kernel):
        sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    gts = np.copy(img_data['bboxes'])
    igs = np.copy(img_data['ignoreareas'])
    scale_map = np.zeros((int(C.size_train[0] / 4), int(C.size_train[1] / 4), 2))
    seman_map = np.zeros((int(C.size_train[0] / 4), int(C.size_train[1] / 4), 3))
    seman_map[:, :, 1] = 1
    if len(igs) > 0:
        igs = igs / 4
        for ind in range(len(igs)):
            x1, y1, x2, y2 = int(igs[ind, 0]), int(igs[ind, 1]), int(np.ceil(igs[ind, 2])), int(np.ceil(igs[ind, 3]))
            seman_map[y1:y2, x1:x2, 1] = 0
    if len(gts) > 0:
        gts = gts / 4
        for ind in range(len(gts)):
            x1, y1, x2, y2 = int(np.ceil(gts[ind, 0])), int(np.ceil(gts[ind, 1])), int(gts[ind, 2]), int(gts[ind, 3])
            y2 = np.minimum(int(C.random_crop[0] / 4) - 1, y2)
            w = x2 - x1
            c_x = int((gts[ind, 0] + gts[ind, 2]) / 2)
            dx = gaussian(w)
            dy = gaussian(w)
            gau_map = np.multiply(dy, np.transpose(dx))

            by = np.minimum(int(C.random_crop[0] / 4) - 1, int(round(y2 + w / 2)))
            ob = int(round(y2 + w / 2)) - by
            seman_map[by - w + ob:by, x1:x2, 0] = np.maximum(seman_map[by - w + ob:by, x1:x2, 0], gau_map[:w - ob, :])
            seman_map[by - w + ob:by, x1:x2, 1] = 1
            seman_map[y2, c_x, 2] = 1

            scale_map[y2 - r:y2 + r + 1, c_x - r:c_x + r + 1, 0] = np.log(gts[ind, 3] - gts[ind, 1])
            scale_map[y2 - r:y2 + r + 1, c_x - r:c_x + r + 1, 1] = 1

    return seman_map, scale_map


def get_data(ped_data, C, batchsize=8):
    current_ped = 0
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
        if current_ped > len(ped_data) - batchsize:
            random.shuffle(ped_data)
            current_ped = 0
        for img_data in ped_data[current_ped:current_ped + batchsize]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=True)
                else:
                    if C.point == 'top':
                        y_seman, y_height = calc_gt_top(C, img_data)
                    elif C.point == 'bottom':
                        y_seman, y_height = calc_gt_bottom(C, img_data)
                    else:
                        y_seman, y_height = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=False)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
        current_ped += batchsize
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_cls(ped_data, C, batchsize=8):
    current_ped = 0
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
        if current_ped > len(ped_data) - batchsize:
            random.shuffle(ped_data)
            current_ped = 0
        for img_data in ped_data[current_ped:current_ped + batchsize]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=True)
                else:
                    if C.point == 'top':
                        y_seman, y_height = calc_gt_top(C, img_data)
                    elif C.point == 'bottom':
                        y_seman, y_height = calc_gt_bottom(C, img_data)
                    else:
                        y_seman, y_height = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=False)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
        current_ped += batchsize
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]


def get_data_ms3_cls(ped_data, C, batchsize=8):
    current_ped = 0
    while True:
        x_img_batch, y_seman_batchc,y_seman_batch, y_height_batch, y_offset_batch = [], [], [], [], []
        if current_ped > len(ped_data) - batchsize:
            random.shuffle(ped_data)
            current_ped = 0
        for img_data in ped_data[current_ped:current_ped + batchsize]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    #y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=True)
                    y_semanc, y_seman, y_height, y_offset = calc_ms3anchor1cls_gt_center(C, img_data, down=C.down,
                                                                                         r=C.radius, scale=C.scale,
                                                                                         offset=True)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batchc.append(np.expand_dims(y_semanc, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_offset_batch.append(np.expand_dims(y_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt:', e))

        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batchc = np.concatenate(y_seman_batchc, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_offset_batch = np.concatenate(y_offset_batch, axis=0)
        current_ped += batchsize
        yield np.copy(x_img_batch), [np.copy(y_seman_batchc),np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]




def get_data_hybrid(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augmentRGBF(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)
                else:
                    if C.point == 'top':
                        y_seman, y_height = calc_gt_top(C, img_data)
                    elif C.point == 'bottom':
                        y_seman, y_height = calc_gt_bottom(C, img_data)
                    else:
                        y_seman, y_height = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=False)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augmentRGBF(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)
                else:
                    if C.point == 'top':
                        y_seman, y_height = calc_gt_top(C, img_data)
                    elif C.point == 'bottom':
                        y_seman, y_height = calc_gt_bottom(C, img_data)
                    else:
                        y_seman, y_height = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=False)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]


def get_data_hybrid_h(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)
                else:
                    if C.point == 'top':
                        y_seman, y_height = calc_gt_top(C, img_data)
                    elif C.point == 'bottom':
                        y_seman, y_height = calc_gt_bottom(C, img_data)
                    else:
                        y_seman, y_height = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=False)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)
                else:
                    if C.point == 'top':
                        y_seman, y_height = calc_gt_top(C, img_data)
                    elif C.point == 'bottom':
                        y_seman, y_height = calc_gt_bottom(C, img_data)
                    else:
                        y_seman, y_height = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=False)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp

        yield np.copy(x_img_batch), [np.copy(y_height_batch)]




def get_data_hybrid_cls_offset(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_offset_batch = [], [], []
        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_offset = calc_gt_center_cls_offset(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))

                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman,  y_offset = calc_gt_center_cls_offset(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))

                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)

        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_offset_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch)]




def get_data_hybridms3(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, y_seman_batch1, y_height_batch1, y_offset_batch1, y_seman_batch2, y_height_batch2, y_offset_batch2 = [], [], [], [], [], [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_offset1, y_seman2, y_height2, y_offset2 = calc_ms3_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    y_offset_batch1.append(np.expand_dims(y_offset1, axis=0))
                    y_offset_batch2.append(np.expand_dims(y_offset2, axis=0))
            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_offset1, y_seman2, y_height2, y_offset2 = calc_ms3_gt_center(
                        C, img_data, down=C.down, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    y_offset_batch1.append(np.expand_dims(y_offset1, axis=0))
                    y_offset_batch2.append(np.expand_dims(y_offset2, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_seman_batch1 = np.concatenate(y_seman_batch1, axis=0)
        y_height_batch1 = np.concatenate(y_height_batch1, axis=0)
        y_seman_batch2 = np.concatenate(y_seman_batch2, axis=0)
        y_height_batch2 = np.concatenate(y_height_batch2, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
            y_offset_batch1 = np.concatenate(y_offset_batch1, axis=0)
            y_offset_batch2 = np.concatenate(y_offset_batch2, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch),
                                         np.copy(y_seman_batch1), np.copy(y_height_batch1), np.copy(y_offset_batch1),
                                         np.copy(y_seman_batch2), np.copy(y_height_batch2), np.copy(y_offset_batch2)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybridms3anchor(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_seman_batch1,y_height_batch1, y_seman_batch2, y_height_batch2 = [], [], [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_seman1, y_height1, y_seman2, y_height2 = calc_ms3anchor_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_seman1, y_height1, y_seman2, y_height2 = calc_ms3anchor_gt_center(
                        C, img_data, down=C.down, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))

            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_seman_batch1 = np.concatenate(y_seman_batch1, axis=0)
        y_height_batch1 = np.concatenate(y_height_batch1, axis=0)
        y_seman_batch2 = np.concatenate(y_seman_batch2, axis=0)
        y_height_batch2 = np.concatenate(y_height_batch2, axis=0)

        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch),
                                         np.copy(y_seman_batch1), np.copy(y_height_batch1),
                                         np.copy(y_seman_batch2), np.copy(y_height_batch2)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybridms3anchor1(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_seman_batch1,y_height_batch1, y_seman_batch2, y_height_batch2 = [], [], [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_seman1, y_height1, y_seman2, y_height2 = calc_ms3anchor1_gt_center(C, img_data, down=C.down, r=C.radius,scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_seman1, y_height1, y_seman2, y_height2 = calc_ms3anchor1_gt_center(
                        C, img_data, down=C.down, r=C.radius, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))

            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_seman_batch1 = np.concatenate(y_seman_batch1, axis=0)
        y_height_batch1 = np.concatenate(y_height_batch1, axis=0)
        y_seman_batch2 = np.concatenate(y_seman_batch2, axis=0)
        y_height_batch2 = np.concatenate(y_height_batch2, axis=0)

        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch),
                                         np.copy(y_seman_batch1), np.copy(y_height_batch1),
                                         np.copy(y_seman_batch2), np.copy(y_height_batch2)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybridms3anchor1c(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch = [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height  = calc_ms3anchor1c_gt_center(C, img_data, down=C.down, r=C.radius,scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))


            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height  = calc_ms3anchor1c_gt_center(C, img_data, down=C.down, r=C.radius, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))


            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)


        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]


def get_data_hybridms3anchor1cls(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batchc,y_seman_batch,  y_height_batch ,y_offset_batch = [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_semanc, y_seman, y_height, y_offset  = calc_ms3anchor1cls_gt_center(C, img_data, down=C.down, r=C.radius,scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batchc.append(np.expand_dims(y_semanc, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_offset_batch.append(np.expand_dims(y_offset, axis=0))


            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_semanc, y_seman, y_height  , y_offset  = calc_ms3anchor1cls_gt_center(C, img_data, down=C.down, r=C.radius, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batchc.append(np.expand_dims(y_semanc, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_offset_batch.append(np.expand_dims(y_offset, axis=0))


            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batchc = np.concatenate(y_seman_batchc, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_offset_batch = np.concatenate(y_offset_batch, axis=0)

        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batchc),np.copy(y_seman_batch), np.copy(y_height_batch),np.copy(y_offset_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]

def get_data_hybridms4anchorcls(ped_data, C, batchsize=8):
    current_ped = 0

    while True:
        x_img_batch, y_seman_batchc,y_seman_batch,  y_height_batch ,y_offset_batch = [], [], [], [], []

        if current_ped > len(ped_data) - batchsize:
            random.shuffle(ped_data)
            current_ped = 0
        for img_data in ped_data[current_ped:current_ped + batchsize]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_semanc, y_seman, y_height, y_offset  = calc_ms4anchor1cls_gt_center(C, img_data, down=C.down, r=C.radius,scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batchc.append(np.expand_dims(y_semanc, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_offset_batch.append(np.expand_dims(y_offset, axis=0))


            except Exception as e:
                print(('get_batch_gt:', e))

        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batchc = np.concatenate(y_seman_batchc, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_offset_batch = np.concatenate(y_offset_batch, axis=0)

        current_ped += batchsize
        yield np.copy(x_img_batch), [np.copy(y_seman_batchc),np.copy(y_seman_batch), np.copy(y_height_batch),np.copy(y_offset_batch)]




def get_data_hybridms2losscat(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch = [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height  = calc_ms2losscat_gt_center(C, img_data, down=C.down, r=C.radius,scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))


            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height  = calc_ms2losscat_gt_center(C, img_data, down=C.down, r=C.radius, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))


            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)


        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]


def get_data_hybridms2_cls(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batchc, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_semanc, y_seman, y_height, y_offset = calc_ms2loss_gt_center(C, img_data, down=C.down, r=C.radius,scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batchc.append(np.expand_dims(y_semanc, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_offset_batch.append(np.expand_dims(y_offset, axis=0))


            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_semanc, y_seman, y_height, y_offset = calc_ms2loss_gt_center(C, img_data, down=C.down, r=C.radius, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batchc.append(np.expand_dims(y_semanc, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_offset_batch.append(np.expand_dims(y_offset, axis=0))


            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batchc = np.concatenate(y_seman_batchc, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_offset_batch = np.concatenate(y_offset_batch, axis=0)


        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batchc),np.copy(y_seman_batch), np.copy(y_height_batch),np.copy(y_offset_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]




def get_data_hybridms3anchor1cc(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch , y_offset_batch= [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height , y_offset = calc_ms3anchor1cc_gt_center(C, img_data, down=C.down, r=C.radius,scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height , y_offset= calc_ms3anchor1cc_gt_center(C, img_data, down=C.down, r=C.radius, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_offset_batch = np.concatenate(y_offset_batch, axis=0)

        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch),np.copy(y_offset_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]



def get_data_hybridms2anchor(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_seman_batch1,y_height_batch1  = [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_seman1, y_height1 = calc_ms2anchor_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))


            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_seman1, y_height1 = calc_ms2anchor_gt_center(
                        C, img_data, down=C.down, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))


            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_seman_batch1 = np.concatenate(y_seman_batch1, axis=0)
        y_height_batch1 = np.concatenate(y_height_batch1, axis=0)


        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch),
                                         np.copy(y_seman_batch1), np.copy(y_height_batch1)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]



def get_data_hybridms3i(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, y_seman_batch1, y_height_batch1, y_offset_batch1, y_seman_batch2, y_height_batch2, y_offset_batch2,y_seman_batcha = [], [], [], [], [], [], [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_offset1, y_seman2, y_height2, y_offset2,y_semana = calc_ms3i_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    y_offset_batch1.append(np.expand_dims(y_offset1, axis=0))
                    y_offset_batch2.append(np.expand_dims(y_offset2, axis=0))
            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_offset1, y_seman2, y_height2, y_offset2,y_semana = calc_ms3i_gt_center(
                        C, img_data, down=C.down, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    y_offset_batch1.append(np.expand_dims(y_offset1, axis=0))
                    y_offset_batch2.append(np.expand_dims(y_offset2, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batcha = np.concatenate(y_seman_batcha, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_seman_batch1 = np.concatenate(y_seman_batch1, axis=0)
        y_height_batch1 = np.concatenate(y_height_batch1, axis=0)
        y_seman_batch2 = np.concatenate(y_seman_batch2, axis=0)
        y_height_batch2 = np.concatenate(y_height_batch2, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
            y_offset_batch1 = np.concatenate(y_offset_batch1, axis=0)
            y_offset_batch2 = np.concatenate(y_offset_batch2, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch),
                                         np.copy(y_seman_batch1), np.copy(y_height_batch1), np.copy(y_offset_batch1),
                                         np.copy(y_seman_batch2), np.copy(y_height_batch2), np.copy(y_offset_batch2),np.copy(y_seman_batcha)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]


def get_data_hybridms3ii(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, y_seman_batch1, y_height_batch1, y_seman_batch2, y_height_batch2,y_seman_batcha = [], [], [], [], [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_seman2, y_height2,y_semana = calc_ms3ii_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_seman2, y_height2,y_semana = calc_ms3ii_gt_center(
                        C, img_data, down=C.down, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batcha = np.concatenate(y_seman_batcha, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_seman_batch1 = np.concatenate(y_seman_batch1, axis=0)
        y_height_batch1 = np.concatenate(y_height_batch1, axis=0)
        y_seman_batch2 = np.concatenate(y_seman_batch2, axis=0)
        y_height_batch2 = np.concatenate(y_height_batch2, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)

        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch),
                                         np.copy(y_seman_batch1), np.copy(y_height_batch1),
                                         np.copy(y_seman_batch2), np.copy(y_height_batch2),np.copy(y_seman_batcha)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]


def get_data_hybridms2i(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, y_seman_batch1, y_height_batch1, y_offset_batch1,y_seman_batcha = [], [], [], [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_offset1, y_semana = calc_ms2i_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    y_offset_batch1.append(np.expand_dims(y_offset1, axis=0))
            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_offset1, y_semana = calc_ms2i_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    y_offset_batch1.append(np.expand_dims(y_offset1, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batcha = np.concatenate(y_seman_batcha, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_seman_batch1 = np.concatenate(y_seman_batch1, axis=0)
        y_height_batch1 = np.concatenate(y_height_batch1, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
            y_offset_batch1 = np.concatenate(y_offset_batch1, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch),
                                         np.copy(y_seman_batch1), np.copy(y_height_batch1), np.copy(y_offset_batch1),
                                         np.copy(y_seman_batcha)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]


def get_data_hybridms2ii(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, y_seman_batch1, y_height_batch1,y_seman_batcha = [], [], [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_semana = calc_ms2ii_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_semana = calc_ms2ii_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batcha = np.concatenate(y_seman_batcha, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_seman_batch1 = np.concatenate(y_seman_batch1, axis=0)
        y_height_batch1 = np.concatenate(y_height_batch1, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)

        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch),
                                         np.copy(y_seman_batch1), np.copy(y_height_batch1),
                                         np.copy(y_seman_batcha)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]



def get_data_hybridms3i0(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, y_seman_batch1, y_height_batch1, y_offset_batch1, y_seman_batch2, y_height_batch2, y_offset_batch2,y_seman_batcha = [], [], [], [], [], [], [], [], [], [], []

        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_offset1, y_seman2, y_height2, y_offset2,y_semana = calc_ms30i_gt_center(C, img_data, down=C.down, scale=C.scale, offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    y_offset_batch1.append(np.expand_dims(y_offset1, axis=0))
                    y_offset_batch2.append(np.expand_dims(y_offset2, axis=0))
            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment_ratio(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset, y_seman1, y_height1, y_offset1, y_seman2, y_height2, y_offset2,y_semana = calc_ms30i_gt_center(
                        C, img_data, down=C.down, scale=C.scale, offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                y_seman_batch1.append(np.expand_dims(y_seman1, axis=0))
                y_height_batch1.append(np.expand_dims(y_height1, axis=0))
                y_seman_batch2.append(np.expand_dims(y_seman2, axis=0))
                y_height_batch2.append(np.expand_dims(y_height2, axis=0))
                y_seman_batcha.append(np.expand_dims(y_semana, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    y_offset_batch1.append(np.expand_dims(y_offset1, axis=0))
                    y_offset_batch2.append(np.expand_dims(y_offset2, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batcha = np.concatenate(y_seman_batcha, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        y_seman_batch1 = np.concatenate(y_seman_batch1, axis=0)
        y_height_batch1 = np.concatenate(y_height_batch1, axis=0)
        y_seman_batch2 = np.concatenate(y_seman_batch2, axis=0)
        y_height_batch2 = np.concatenate(y_height_batch2, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
            y_offset_batch1 = np.concatenate(y_offset_batch1, axis=0)
            y_offset_batch2 = np.concatenate(y_offset_batch2, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch),
                                         np.copy(y_seman_batch1), np.copy(y_height_batch1), np.copy(y_offset_batch1),
                                         np.copy(y_seman_batch2), np.copy(y_height_batch2), np.copy(y_offset_batch2),np.copy(y_seman_batcha)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]



def get_data_hybrid_rec(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, rec_batch = [], [], [], [], []
        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset,rec_map = calc_gt_center_rec(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                rec_batch.append(np.expand_dims(rec_map, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset,rec_map = calc_gt_center_rec(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                rec_batch.append(np.expand_dims(rec_map, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        rec_batch = np.concatenate(rec_batch, axis=0)

        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch), np.copy(rec_batch)]

def get_data_hybrid_rec45(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch, rec_batch = [], [], [], [], []
        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset,rec_map = calc_gt_center_rec45(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                rec_batch.append(np.expand_dims(rec_map, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset,rec_map = calc_gt_center_rec45(C, img_data, down=C.down, scale=C.scale,
                                                                 offset=C.offset)


                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                rec_batch.append(np.expand_dims(rec_map, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        rec_batch = np.concatenate(rec_batch, axis=0)

        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch), np.copy(rec_batch), np.copy(rec_batch)]





def get_data_hybridxyxy(ped_data, emp_data, C, batchsize=8, hyratio=0.5):
    current_ped = 0
    current_emp = 0
    batchsize_ped = int(batchsize * hyratio)
    batchsize_emp = batchsize - batchsize_ped
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch , giou_offset_batch= [], [], [], [], []
        if current_ped > len(ped_data) - batchsize_ped:
            random.shuffle(ped_data)
            current_ped = 0
        if current_emp > len(emp_data) - batchsize_emp:
            random.shuffle(emp_data)
            current_emp = 0
        for img_data in ped_data[current_ped:current_ped + batchsize_ped]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset,giou_offset = calc_gt_center_giou(C, img_data, down=C.down, scale=C.scale,offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    giou_offset_batch.append(np.expand_dims(giou_offset, axis=0))

            except Exception as e:
                print(('get_batch_gt:', e))
        for img_data in emp_data[current_emp:current_emp + batchsize_emp]:
            try:
                img_data, x_img = data_augment.augment(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset,giou_offset = calc_gt_center_giou(C, img_data, down=C.down, scale=C.scale,offset=C.offset)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
                    giou_offset_batch.append(np.expand_dims(giou_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt_emp:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
            giou_offset_batch = np.concatenate(giou_offset_batch, axis=0)
        current_ped += batchsize_ped
        current_emp += batchsize_emp
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch), np.copy(giou_offset_batch)]




def get_data_wider(ped_data, C, batchsize=8):
    current_ped = 0
    while True:
        x_img_batch, y_seman_batch, y_height_batch, y_offset_batch = [], [], [], []
        if current_ped > len(ped_data) - batchsize:
            random.shuffle(ped_data)
            current_ped = 0
        for img_data in ped_data[current_ped:current_ped + batchsize]:
            try:
                img_data, x_img = data_augment.augment_wider(img_data, C)
                if C.offset:
                    y_seman, y_height, y_offset = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=True)
                else:
                    y_seman, y_height = calc_gt_center(C, img_data, down=C.down, scale=C.scale, offset=False)

                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img_batch.append(np.expand_dims(x_img, axis=0))
                y_seman_batch.append(np.expand_dims(y_seman, axis=0))
                y_height_batch.append(np.expand_dims(y_height, axis=0))
                if C.offset:
                    y_offset_batch.append(np.expand_dims(y_offset, axis=0))
            except Exception as e:
                print(('get_batch_gt:', e))
        x_img_batch = np.concatenate(x_img_batch, axis=0)
        y_seman_batch = np.concatenate(y_seman_batch, axis=0)
        y_height_batch = np.concatenate(y_height_batch, axis=0)
        if C.offset:
            y_offset_batch = np.concatenate(y_offset_batch, axis=0)
        current_ped += batchsize
        if C.offset:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch), np.copy(y_offset_batch)]
        else:
            yield np.copy(x_img_batch), [np.copy(y_seman_batch), np.copy(y_height_batch)]
