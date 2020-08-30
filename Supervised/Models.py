import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.regularizers import *
from keras import layers

############################################ UNET with GC##############################################################

class UNET:
    @staticmethod
    def build(input_shape, classes=2, pretrained_weights=None):
        inputshape = input_shape
        chanDim = -1

        if K.image_data_format() == "channels first":
            inputshape = (input_shape[2], input_shape[0], input_shape[1])
            chanDim = 1

        inputs = Input(inputshape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=chanDim)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=chanDim)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        drop7 = Dropout(0.5)(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2))(drop7))
        merge8 = concatenate([conv2, up8], axis=chanDim)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=chanDim)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(classes, (1, 1), activation='softmax')(conv9)

        model = Model(input=inputs, output=conv10, name='UNET')

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

############################################ UNET Extended ##############################################################

class UNET_Extended:
    @staticmethod
    def build(input_shape, classes=2, pretrained_weights=None):
        inputshape = input_shape
        chanDim = -1

        if K.image_data_format() == "channels first":
            inputshape = (input_shape[2], input_shape[0], input_shape[1])
            chanDim = 1

        inputs = Input(inputshape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=chanDim)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=chanDim)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=chanDim)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=chanDim)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(classes, (1, 1), activation='softmax')(conv9)

        model = Model(input=inputs, output=conv10, name='UNET')

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model


############################################ UNET ##############################################################


class UNET_ORIG:
    @staticmethod
    def build(input_shape, classes=2, pretrained_weights=None):
        inputshape = input_shape
        chanDim = -1

        if K.image_data_format() == "channels first":
            inputshape = (input_shape[2], input_shape[0], input_shape[1])
            chanDim = 1

        inputs = Input(inputshape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=chanDim)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=chanDim)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        drop7 = Dropout(0.5)(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2))(drop7))
        merge8 = concatenate([conv2, up8], axis=chanDim)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=chanDim)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(classes, (1, 1), activation='softmax')(conv9)

        model = Model(input=inputs, output=conv10, name='UNET')

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model


class UNET_Heavy:
    @staticmethod
    def build(input_shape, classes=2, pretrained_weights=None):
        inputshape = input_shape
        chanDim = -1

        if K.image_data_format() == "channels first":
            inputshape = (input_shape[2], input_shape[0], input_shape[1])
            chanDim = 1

        inputs = Input(inputshape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=chanDim)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=chanDim)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        drop7 = Dropout(0.5)(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2))(drop7))
        merge8 = concatenate([conv2, up8], axis=chanDim)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=chanDim)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(classes, (1, 1), activation='softmax')(conv9)

        model = Model(input=inputs, output=conv10, name='UNET')

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

class ResNet50:
    def _identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def _conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def build(self, input_shape, numOfClasses, pretrained_weights=None):
        inputshape = input_shape
        chandim = -1

        if K.image_data_format() == "channels first":
            inputshape = (input_shape[2], input_shape[0], input_shape[1])
            chandim = 1

        input = Input(shape=inputshape)

        x = ZeroPadding2D((3, 3))(input)
        x = Conv2D(64, 7, strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=chandim, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self._conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self._conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='d')
        x = Dropout(0.5)(x)

        x = self._conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self._conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(numOfClasses, activation='sigmoid')(x)

        model = Model(inputs=input, outputs=x, name='ResNet50')

        if pretrained_weights:
            model.load_weights(pretrained_weights)

        return model

############################################# FCN Segmentor #################################################
class FCN:
    @staticmethod
    def build(inputshape, numOfClasses, pretrained_weights=None):
        weight_path = 'vgg_weights.h5'
        input_shape = inputshape
        IMAGE_ORDERING = "channels_last"

        if K.image_data_format() == 'channels first':
            input_shape = (input_shape[2], input_shape[0], input_shape[1])
            IMAGE_ORDERING = "channels_first"

        inputs = Input(shape=input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
            inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
        f1 = x

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
        f2 = x

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
        pool3 = x

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(
            x)  # (None, 14, 14, 512)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(
            pool4)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(
            x)  # (None, 7, 7, 512)

        vgg = Model(inputs, pool5)
        vgg.load_weights(weight_path)

        n = 4096
        o = (Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
        conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

        # 4 times upsamping for pool4 layer
        conv7_4 = Conv2DTranspose(numOfClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False,
                                  data_format=IMAGE_ORDERING)(conv7)
        # (None, 224, 224, 10)
        # 2 times upsampling for pool411
        pool411 = (
            Conv2D(numOfClasses, (1, 1), activation='relu', padding='same', name="pool4_11",
                   data_format=IMAGE_ORDERING))(
            pool4)
        pool411_2 = (
            Conv2DTranspose(numOfClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False,
                            data_format=IMAGE_ORDERING))(
            pool411)

        pool311 = (
            Conv2D(numOfClasses, (1, 1), activation='relu', padding='same', name="pool3_11",
                   data_format=IMAGE_ORDERING))(
            pool3)

        o = Add(name="add")([pool411_2, pool311, conv7_4])
        o = Conv2DTranspose(numOfClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False,
                            data_format=IMAGE_ORDERING)(o)
        o = (Activation('softmax'))(o)

        model = Model(input=inputs, output=o, name='FCN-8')

        if pretrained_weights:
            model.load_weights(pretrained_weights)

        return model

############################################ Dilated Network #################################################

class DilatedNet:
    @staticmethod
    def build(input_shape, classes, pretrained_weights=None):
        inputshape = input_shape

        inputs = Input(inputshape)

        h = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(inputs)
        h = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(h)
        h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
        h = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(h)
        h = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(h)
        h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
        h = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(h)
        h = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(h)
        h = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(h)
        h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
        h = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(h)
        h = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(h)
        h = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(h)
        h = ZeroPadding2D((2, 2))(h)
        h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
        h = ZeroPadding2D((2, 2))(h)
        h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
        h = ZeroPadding2D((2, 2))(h)
        h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
        h = ZeroPadding2D((12, 12))(h)
        h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
        h = Dropout(0.5, name='drop6')(h)
        h = Conv2D(4096, (1, 1), activation='relu', name='fc7', padding='same')(h)
        h = Dropout(0.5, name='drop7')(h)
        h = Conv2D(classes, (1, 1), name='final', padding='same')(h)
        h = Conv2D(classes, (3, 3), activation='relu', name='ctx_conv1_1', padding='same')(h)
        h = Conv2D(classes, (3, 3), activation='relu', name='ctx_conv1_2', padding='same')(h)
        h = ZeroPadding2D(padding=(2, 2))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
        h = ZeroPadding2D(padding=(4, 4))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
        h = ZeroPadding2D(padding=(8, 8))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
        h = ZeroPadding2D(padding=(16, 16))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
        h = ZeroPadding2D(padding=(32, 32))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(32, 32), activation='relu', name='ctx_conv6_1')(h)
        h = ZeroPadding2D(padding=(64, 64))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(64, 64), activation='relu', name='ctx_conv7_1')(h)
        # h = ZeroPadding2D(padding=(1, 1))(h)
        h = Conv2D(classes, (3, 3), activation='relu', name='ctx_fc1', padding='same')(h)
        h = Conv2D(classes, (1, 1), name='ctx_final', padding='same')(h)

        # the following two layers pretend to be a Deconvolution with grouping layer.
        # never managed to implement it in Keras
        # since it's just a gaussian upsampling trainable=False is recommended
        h = UpSampling2D(size=(8, 8))(h)
        output = Conv2D(classes, kernel_size=(16, 16), activation='softmax', trainable=False, use_bias=False, name='ctx_upsample', padding='same')(h)

        model = Model(input=inputs, output=output, name='dilation_cityscapes')

        if pretrained_weights is not None:
            model.load_weights(pretrained_weights)

        return model

# class DilatedNet:
#     @staticmethod
#     def build(input_shape, classes, pretrained_weights=None):
#         inputshape = input_shape
#         chanDim = -1
#
#         if K.image_data_format() == "channels first":
#             inputshape = (input_shape[2], input_shape[0], input_shape[1])
#             chanDim = 1
#
#         inputs = Input(inputshape)
#         x = Conv2D(64, 3, 3, activation='relu', name='conv_1_1', padding='same')(inputs)
#         x = Conv2D(64, 3, 3, activation='relu', name='conv_1_2', padding='same')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2))(x)
#
#         x = Conv2D(128, 3, 3, activation='relu', name='conv_2_1', padding='same')(x)
#         x = Conv2D(128, 3, 3, activation='relu', name='conv_2_2')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2))(x)
#
#         x = Conv2D(256, 3, 3, activation='relu', name='conv_3_1', padding='same')(x)
#         x = Conv2D(256, 3, 3, activation='relu', name='conv_3_2', padding='same')(x)
#         x = Conv2D(256, 3, 3, activation='relu', name='conv_3_3', padding='same')(x)
#         x = MaxPooling2D((2, 2), strides=(2, 2))(x)
#
#         x = Conv2D(512, 3, 3, activation='relu', name='conv_4_1', padding='same')(x)
#         x = Conv2D(512, 3, 3, activation='relu', name='conv_4_2', padding='same')(x)
#         x = Conv2D(512, 3, 3, activation='relu', name='conv_4_3', padding='same')(x)
#
#         x = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='aconv_1_1')(x)
#         x = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='aconv_1_2')(x)
#         x = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='aconv_1_3')(x)
#
#         x = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='aconv_1_4')(x)
#         x = Dropout(0.5)(x)
#         x = Conv2D(4096, 1, 1, activation='relu', padding='same')(x)
#         x = Dropout(0.5)(x)
#         x = Conv2D(classes, 1, 1, activation='linear', padding='same')(x)





