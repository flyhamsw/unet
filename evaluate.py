from data import trainGenerator
from model import unet

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

model = unet()
model.load_weights('unet_bn_kaggle_crack.hdf5')

if __name__ == '__main__':
    testGene = trainGenerator(1, "data/kaggle_crack_segmentation_dataset/test/", "images", "masks", data_gen_args)
    model.evaluate(testGene, steps=1695)
