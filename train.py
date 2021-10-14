from model import *
from data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(1,'data/BigEye_v1_noboundary_onlycrack','img','label_255_dilated_disk3',data_gen_args,save_to_dir = None)

model = unet()

# Load pretrained model
model.load_weights('unet_bn_kaggle_crack.hdf5')

model_checkpoint = ModelCheckpoint('unet_bn_kaggle_crack+BigEye_v1_noboundary_onlycrack.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=101, epochs=10, callbacks=[model_checkpoint])
