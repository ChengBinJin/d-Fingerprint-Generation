import os
import sys
import logging
import cv2
import csv
import numpy as np

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)

class FPGT(object):
    def __init__(self, file_name, is_print=False):
        self.base_path = '../../Data/BMP_320x280'

        self.minuThreshold = 20.
        self.width = 0
        self.height = 0
        self.resolution = 0
        self.quality = 0
        self.nMinutiae = 0
        self.minutiaes = []
        # minutiae location (x, y), minutiae direction (from 0 to 359), minutiae quality (from 0 to 100),
        # minutiae type (Ending or bifurcation)

        self._init_logger()
        self.read_minu(file_name)
        self.print_info(is_print)

    @staticmethod
    def _init_logger():
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler('../Data/fpgt.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    def read_minu(self, file_name):
        gt_name = os.path.join(self.base_path, file_name + '.MINU')
        if not os.path.isfile(gt_name):
            logger.info('Lost file_name: {}'.format(file_name + '.MINU'))
            return 0

        with open(gt_name, 'r') as f:
            rows = [row for row in csv.reader(f, delimiter='\t')]
            for i, row in enumerate(rows):
                if i == 0:
                    self.width = int(row[1])
                elif i == 1:
                    self.height = int(row[1])
                elif i == 2:
                    self.resolution = int(row[1])
                elif i == 3:
                    self.quality = int(row[1])
                elif i == 4:
                    self.nMinutiae = int(row[1])
                else:
                    if int(row[3]) < self.minuThreshold:
                        self.nMinutiae -= 1
                        continue
                    else:
                        self.minutiaes.append([int(row[0]), int(row[1]), int(row[2]), int(row[3]),
                                               0 if row[4].lower() == 'end' else 1])
            f.close()

    def print_info(self, is_print):
        if is_print:
            print('Image width: {}'.format(self.width))
            print('Image height: {}'.format(self.height))
            print('Image resolution: {}'.format(self.resolution))
            print('Image quality: {}'.format(self.quality))
            print('Image nMinutia: {}'.format(self.nMinutiae))
            for idx in range(self.nMinutiae):
                print('index: {}, x: {}, y: {}, direction: {}, quality: {}, type: {}'.format(
                    str(idx).zfill(3), self.minutiaes[idx][0], self.minutiaes[idx][1], self.minutiaes[idx][2],
                    self.minutiaes[idx][3], 'Ending' if self.minutiaes[idx][4] == 0 else 'Bifurcation'))

def key_control(idx, is_view):
    asc_code = cv2.waitKey(0) & 0xFF
    if asc_code == 27:
        sys.exit('Esc clicked!')
    elif asc_code == ord('v'):
        is_view = not is_view
    elif asc_code == ord('p'):
        idx -= 1
        if idx < 0:
            idx = 0
    else:
        idx += 1

    return idx, is_view

def fancyShow(img, fpgt, resize_ratio=1.):
    showImg = img.copy()

    # Ending red circel
    # Bifurcation blue square
    redColor = (0, 0, 255)
    blueColor = (255, 0, 0)
    radius = int(np.round(3. * resize_ratio))
    thickness = int(np.round(1. * resize_ratio))
    length = int(np.round(6. * resize_ratio))

    for idx in range(fpgt.nMinutiae):
        x, y, direc, _, minutiae_type = fpgt.minutiaes[idx]
        x, y = int(np.round(resize_ratio * x)), int(np.round(resize_ratio * y))
        radian = direc * 2. * np.pi / 360.
        if minutiae_type == 0:  # ending
            cv2.circle(showImg, (x, y), radius,redColor, thickness)
            cv2.line(showImg, (x, y), (int(np.round(x + np.cos(radian) * length)),
                                       int(np.round(y + np.sin(radian) * length))), redColor, thickness)
        elif minutiae_type == 1:  # bifurcation
            cv2.rectangle(showImg, (x-radius, y-radius), (x+radius, y+radius), blueColor, thickness)
            cv2.line(showImg, (x, y), (int(np.round(x + np.cos(radian) * length)),
                                       int(np.round(y + np.sin(radian) * length))), blueColor, thickness)

    return showImg
