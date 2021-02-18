import random
import cv2
import numpy as np
import torch


def ReadIndex(index_path, shuffle=False):
    img_list = []
    with open(index_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    if shuffle is True:
        random.shuffle(img_list)
    return img_list



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _preprocess_img(img):
    cvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cvImage = cvImage.transpose(2, 0, 1).astype(np.float32) / 255
    return torch.from_numpy(cvImage)

def _tunnel_preprocess_img(img):
    cvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    cvImage = clahe.apply(cvImage)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR)
    cvImage = cvImage.transpose(2, 0, 1).astype(np.float32) / 255
    return torch.from_numpy(cvImage)

def _tunnel_preprocess_gray(img):
    cvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    cvImage = clahe.apply(cvImage)

    cvImage = cvImage.astype(np.float32) / 255
    cvImage = np.expand_dims(cvImage,axis=0)
    return torch.from_numpy(cvImage)

def _preprocess_lab_connected(mask):
    cvImage = np.array(mask)
    cvImage[cvImage > 0] = 255
    cvImage = cvImage.astype(np.float32) / 255
    connected = _calculate_connected(cvImage)

    return [torch.from_numpy(cvImage), torch.from_numpy(connected)]


def _preprocess_lab(mask):
    cvImage = np.array(mask)
    cvImage[cvImage > 0] = 255
    cvImage = cvImage.astype(np.float32) / 255

    return [torch.from_numpy(cvImage),]

def _calculate_connected(cvImage):
    H, W = cvImage.shape
    connected = np.zeros((8, H, W), dtype=np.int)
    idx_list = np.where(cvImage > 0)
    for idx in zip(idx_list[0], idx_list[1]):
        for i in range(8):
            connected[i][idx[0]][idx[1]] = _pixel_eight_neighborhood(cvImage,idx, i)
    return connected.astype(np.float32)

def _pixel_eight_neighborhood(cvImage, idx, i):
    '''

    :param cvImage: mask numpy 0 and 1
    :param idx: (y,x)
    :param i:
    :return:

    0  1  2
    3  X  4
    5  6  7
    '''

    H, W = cvImage.shape

    loc_x = int(i%3-1 + idx[1])# w
    loc_y = int(i/3-1 + idx[0])# h
    if W>loc_x >= 0 and H>loc_y >= 0:
        if cvImage[loc_y, loc_x] == 1:
            return 1
        elif cvImage[loc_y, loc_x] == 0:
            return 0
        else:
            ValueError("mask error: not 1 or 0")
    else:
        return 0