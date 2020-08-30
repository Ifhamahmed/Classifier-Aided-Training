from comet_ml import Experiment
import matplotlib

experiment = Experiment("m487mKQwNTqFF4z7aZRX3Xv19", project_name="UNET_CS", log_env_gpu=True)
matplotlib.use("Agg")

# import packages
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from sklearn.metrics import classification_report
# from imutils import paths
from keras.optimizers import Adam
from dataLoader import Dataloader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
from Models import UNET
import tensorflow as tf
from keras import backend as K

def weighted_crossentropy(class_weights):
    weights = class_weights
    def w_cce(y_true, y_pred):
        def _to_tensor(x, dtype):
            """Convert the input `x` to a tensor of type `dtype`.
            # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
            # Returns
            A tensor.
            """
            x = tf.convert_to_tensor(x)
            if x.dtype != dtype:
                x = tf.cast(x, dtype)
            return x

        y_pred /= tf.reduce_sum(y_pred, axis=len(y_pred.get_shape()) - 1,
                            keepdims=True)
        w = _to_tensor(weights, 'float32')
        shape = tf.shape(y_pred)
        tmp = tf.reshape(w, (1, 1, 1, shape[3]))
        w = tf.tile(tmp, (shape[0], shape[1], shape[2], 1))

        _EPSILON = K.epsilon()
        epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        clipped_y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # compute crossentropy
        cross_entropy = y_true * tf.math.log(clipped_y_pred)

        # apply weights
        weighted_cce = tf.math.multiply(cross_entropy, w)
        return - tf.reduce_sum(weighted_cce, axis=len(y_pred.get_shape()) - 1)
    return w_cce


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Input Path of dataset")
ap.add_argument("-m", "--model", required=True, help="Path of location to save model")
ap.add_argument("-p", "--plot", required=True, help="Path of location to save plot")
args = vars(ap.parse_args())

# get dataset path
dataset_path = args["dataset"]

# import colour palette
#par = 0
#weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, par, 1, par, 1, 1, 1, 1, par, par, 1, 1, par, par, par, par, par])
df = pd.read_csv('classes.csv', ",", header=None)
palette = np.array(df.values, dtype=np.uint8)
# palette = palette[np.nonzero(weights)[0], :]
num_of_Classes = palette.shape[0]

# initialize data and label paths
print("[INFO] loading images........")

train_frame_path = os.path.sep.join([args["dataset"], "train_frames/train"])
train_mask_path = os.path.sep.join([args["dataset"], "train_masks/train"])

val_frame_path = os.path.sep.join([args["dataset"], "val_frames/val"])
val_mask_path = os.path.sep.join([args["dataset"], "val_masks/val"])

input_size = [640, 640]
# instantiate datagenerator class
train_set = Dataloader(image_paths=train_frame_path,
                       mask_paths=train_mask_path,
                       image_size=input_size,
                       numclasses=num_of_Classes,
                       channels=[3, 3],
                       palette=palette,
                       seed=47)

val_set = Dataloader(image_paths=val_frame_path,
                     mask_paths=val_mask_path,
                     image_size=input_size,
                     numclasses=num_of_Classes,
                     channels=[3, 3],
                     palette=palette,
                     seed=47)

# build model
model = UNET.build((640, 640, 3), num_of_Classes, pretrained_weights='weights.h5')
model.summary()

# learning parameters
BS = 2
INIT_LR = 0.00001
EPOCHS = 30

# initialize data generators
traingen = train_set.data_gen(should_augment=True, batch_size=BS)
valgen = val_set.data_gen(should_augment=False, batch_size=BS)

# initialise variables
No_of_train_images = len(os.listdir(dataset_path + '/train_frames/train'))
No_of_val_images = len(os.listdir(dataset_path + '/val_frames/val'))
print("Number of Training Images = {}".format(No_of_train_images))

weights_path = os.getcwd() + '/output/weights_CS_UNET.h5'
opt = Adam(lr=INIT_LR)

par = 0
weights = np.array([par, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, par, 1, par, 1, 1, 1, 1, par, par, 1, 1, par, par, par, par, par])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

# early stopping parameters and output
checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
csvlogger = CSVLogger('./log.out', append=True, separator=';')
# earlystopping = EarlyStopping(monitor='val_loss', verbose=1, min_delta=0.001, patience=10, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_delta=0.00001,
                              cooldown=0, min_lr=0.000001)

callbacks_list = [checkpoint, csvlogger, reduce_lr]

# train network
print("[INFO] Training network.........")
H = model.fit_generator(traingen, epochs=EPOCHS,
                        steps_per_epoch=No_of_train_images // BS,
                        validation_data=valgen,
                        validation_steps=No_of_val_images // BS,
                        callbacks=callbacks_list)

# evaluate Network
print("[INFO] Evaluating network.........")

# plot the training loss and accuracy
N = np.arange(0, len(H.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (UNET_H_Cityscapes)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"] + '/UNET_plot_CS.png')

# save the model
print("[INFO] Saving the model...")
model.save_weights(args["model"] + '/' + 'UNET_CS_weights.h5')
print('[INFO] Process Finished!!')
experiment.end()
