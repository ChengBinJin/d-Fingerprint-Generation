import os
import sys
import csv
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--resize_ratio', dest='resize_ratio', type=float, default=2., help='resize ratio to easy check')
parser.add_argument('--is_train_data', dest='is_train_data', type=bool, default=True, help='show traind or test data')
parser.add_argument('--interval', dest='interval', type=int, default=0, help='interval between two frames')
args = parser.parse_args()


class FPGT(object):
    def __init__(self, file_name, is_print=True):
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

        self.read_minu(file_name)
        self.print_info(is_print)

    def read_minu(self, file_name):
        with open(os.path.join(self.base_path, file_name + '.MINU'), 'r') as f:
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

def fancyShow(img, fpgt, resize_ratio):
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

IS_VIEW = True
def main(basePath, data_file):
    imgNames = []
    with open(data_file, 'r') as f:
        for imgName in csv.reader(f, delimiter='\n'):
            imgNames.extend(imgName)

    for i, imgName in enumerate(imgNames):
        cv2.namedWindow(imgName, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(imgName, 100, 100)

        img = cv2.imread(os.path.join(basePath, imgName + '.bmp'))
        fpgt = FPGT(imgName)

        img = cv2.resize(img, None, fx=args.resize_ratio, fy=args.resize_ratio, interpolation=cv2.INTER_CUBIC)
        showImg = fancyShow(img, fpgt, args.resize_ratio)

        cv2.imshow(imgName, showImg)
        if cv2.waitKey(args.interval) & 0xFF == 27:
            cv2.destroyAllWindows()
            sys.exit('Esc clicked!')
        cv2.destroyAllWindows()

        # global IS_VIEW
        # if IS_VIEW:
        #     cv2.imshow('showImg', showImg)  # Display minutiaes
        # else:
        #     cv2.imshow('showImg', img)

        # asc_code = cv2.waitKey(0) & 0xFF
        # if asc_code == 27:
        #     sys.exit('Esc clicked!')
        # elif asc_code == ord('v'):
        #     IS_VIEW = not IS_VIEW


if __name__ == '__main__':
    base_path = '../../Data/BMP_320x280'
    train_file = '../data/train_images.txt'
    test_file = '../data/test_images.txt'

    if args.is_train_data:
        main(base_path, train_file)
    else:
        main(base_path, test_file)
