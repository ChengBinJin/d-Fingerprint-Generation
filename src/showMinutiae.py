import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--resize_ratio', dest='resize_ratio', type=float, default=2., help='resize ratio to easy check')
parser.add_argument('--is_train', dest='is_train', action='store_true', default=False,
                    help='show train or test data')
parser.add_argument('--is_print', dest='is_print', action='store_true', default=False,
                    help='show minutiae data or not')
args = parser.parse_args()


IS_VIEW = True
def main(basePath, data_file):
    # read showing file names
    imgNames = []
    with open(data_file, 'r') as f:
        for imgName in csv.reader(f, delimiter='\n'):
            imgNames.extend(imgName)

    idx = 0
    while True:
        imgName = imgNames[idx]
        winName = 'ID:' + str(idx).zfill(4) + ' ' + imgName + '.bmp'
        cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(winName, 100, 0)

        img = cv2.imread(os.path.join(basePath, imgName + '.bmp'))
        fpgt = FPGT(imgName, is_print=args.is_print)

        img = cv2.resize(img, None, fx=args.resize_ratio, fy=args.resize_ratio, interpolation=cv2.INTER_CUBIC)
        showImg = fancyShow(img, fpgt, args.resize_ratio)

        global IS_VIEW
        if IS_VIEW:
            cv2.imshow(winName, showImg)  # Display minutiaes
        else:
            cv2.imshow(winName, img)  # hide minutiaes

        idx, IS_VIEW = key_control(idx, IS_VIEW)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    base_path = '../../Data/BMP_320x280'
    train_file = '../data/train_images.txt'
    test_file = '../data/test_images.txt'

    print('hello world!')

    if args.is_train:
        main(base_path, train_file)
    else:
        main(base_path, test_file)

