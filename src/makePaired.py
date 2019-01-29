import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--is_write', dest='is_write', action='store_true', default=False,
                    help='write data to disk or not')
parser.add_argument('--is_train', dest='is_train', action='store_true', default=False,
                    help='write train or test data as paired data')
parser.add_argument('--is_show', dest='is_show', action='store_true', default=False,
                    help='show original and minutiae map')
parser.add_argument('--resize_ratio', dest='resize_ratio', type=float, default=2.,
                    help='resize ratio to easy check')
parser.add_argument('--is_print', dest='is_print', action='store_true', default=False,
                    help='print minutiae information or not')
parser.add_argument('--interval', dest='interval', type=int, default=0, help='interval time between two imgs')
args = parser.parse_args()


def encode_minumap(minutia_list, height=320, width=280, channel=3, bs=5, resize_ratio=1.):
    minumap = np.zeros((height, width, channel), dtype=np.uint8)
    scale = 255. / 180.

    for i, minutiae in enumerate(minutia_list):
        minu_y, minu_x = minutiae[0:2]
        minu_direc = minutiae[2]
        minu_type = 'Ending' if minutiae[4] == 0 else 'Bifurcation'

        minumap[minu_x-bs:minu_x+bs, minu_y-bs:minu_y+bs, 0] = (np.floor(minu_direc / 2.) + 1) * scale
        minumap[minu_x-bs:minu_x+bs, minu_y-bs:minu_y+bs, 1] = 128 if minu_type == 'Ending' else 255

    resizedMinumap = minumap.copy()
    if resize_ratio != 1.:
        resizedMinumap = cv2.resize(minumap, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)

    return minumap, resizedMinumap

def imshow(img, fpgt, reMinumap):
    winName = 'Show'
    cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(winName, 0, 0)

    reImg = cv2.resize(img, None, fx=args.resize_ratio, fy=args.resize_ratio, interpolation=cv2.INTER_CUBIC)
    minuimg = fancyShow(reImg, fpgt, resize_ratio=args.resize_ratio)  # draw minutiae points

    showImg = np.hstack([reImg, minuimg, reMinumap])

    cv2.imshow(winName, showImg)
    if cv2.waitKey(args.interval) & 0xFF == 27:
        sys.exit('Esc clicked!')

def imwrite(img, minumap, basePath, imgName):
    data = 'train' if args.is_train else 'test'
    save_folder = os.path.join(basePath, 'paired/{}'.format(data))

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    newImg = np.hstack([minumap, img])
    cv2.imwrite(os.path.join(save_folder, imgName + '.png'), newImg[:, :, ::-1])


def main(basePath, data_file):
    # read showing file names
    imgNames = []
    with open(data_file, 'r') as f:
        for imgName in csv.reader(f, delimiter='\n'):
            imgNames.extend(imgName)

    init_logger('fpgt.log') # initialize logger()

    # for idx, imgName in enumerate(imgNames):
    for idx in range(0, len(imgNames)):
        imgName = imgNames[idx]
        print('index: {}, imgName: {}'.format(str(idx).zfill(5), imgName))

        img = cv2.imread(os.path.join(basePath, imgName + '.bmp'))
        fpgt = FPGT(imgName)
        if fpgt.nMinutiae != 0:  # just do following operators ther are '.MINU' file
            minumap, reMinumap = encode_minumap(fpgt.minutiaes, resize_ratio=args.resize_ratio)

            if args.is_show:
                imshow(img, fpgt, reMinumap)

            if args.is_write:
                imwrite(img, minumap, basePath, imgName)  # save as paired img


if __name__ == '__main__':
    base_path = '../../Data/BMP_320x280'
    train_file = '../data/train_images.txt'
    test_file = '../data/test_images.txt'

    if args.is_train:
        main(base_path, train_file)
    else:
        main(base_path, test_file)
