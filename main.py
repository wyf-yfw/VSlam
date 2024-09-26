import cv2
from vslam import *

if __name__ == '__main__':
    d435i = Camera()
    FE = FeatureExtractor()
    while True:
        img, _ = d435i(GRAY=True)
        conv_img = FE.conv(img)
        cv2.imshow('img', img)
        cv2.imshow('conv', conv_img)
        cv2.waitKey(1)
