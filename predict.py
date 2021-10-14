import os

from tqdm import tqdm
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
import numpy as np

from model import *
from data import *

model = unet()
model.load_weights('unet_bn_kaggle_crack.hdf5')


def predict_single_image(input_fpath, output_fpath):
    im = imread(input_fpath, as_gray=True)
    orig_shape = im.shape

    im = resize(im, (512, 512, 1))
    im = np.array([im])

    pred = model.predict(im)[0][:, :, 1]  # 0: background, 1: crack
    pred = resize(pred, orig_shape)

    imsave(output_fpath, pred)


def predict_directory(input_dir, output_dir):
    fname_list = os.listdir(input_dir)
    for fname in tqdm(fname_list):
        input_fpath = os.path.join(input_dir, fname)
        output_fpath = os.path.join(output_dir, fname)
        predict_single_image(input_fpath, output_fpath)


if __name__ == '__main__':
    predict_directory('data/kaggle_crack_segmentation_dataset/test/images', 'predictions/unet_bn_kaggle_crack/kaggle_crack_segmentation')
    predict_directory('data/BigEye_v1_noboundary_onlycrack/img', 'predictions/unet_bn_kaggle_crack/BigEye_v1_noboundary_onlycrack')
