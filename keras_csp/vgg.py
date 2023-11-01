# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import *
from keras import backend as K
from .keras_layer_L2Normalization import L2Normalization
import numpy as np
import keras, math
from keras.models import Model
from keras.models import Sequential

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# focal loss like
def prior_probability_onecls(num_class=1, probability=0.01):
	def f(shape, dtype=keras.backend.floatx()):
		assert(shape[0] % num_class == 0)
		# set bias to -log((1 - p)/p) for foregound
		result = np.ones(shape, dtype=dtype) * -math.log((1 - probability) / probability)
		# set bias to -log(p/(1 - p)) for background
		return result
	return f


def VGG16(input_tensor=None,include_top=True,classes=1, trainable=True):

    img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=trainable)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', trainable=trainable)(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1', trainable=trainable)(x)
        x = Dense(4096, activation='relu', name='fc2', trainable=trainable)(x)
        #x = Dense(classes, activation='softmax', name='predictions', trainable=trainable)(x)
        x = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal',
                          bias_initializer=prior_probability_onecls(probability=0.01), trainable=trainable,
                          name='predictionsig')(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(trainable=trainable)(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(trainable=trainable)(x)

    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, dila=(1, 1), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=dila, padding='same',
                      name=conv_name_base + '2b', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def nn_p2h5123(img_input=None, offset=True, num_scale=1, trainable=False):
    bn_axis = 3
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)


    feat = Convolution2D(512, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',trainable=trainable)(x)
    feat = BatchNormalization(axis=bn_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)


    x_regr = Convolution2D(num_scale, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)


    return [x_regr]




def nn_p3p4p5(img_input=None, offset=True, num_scale=1, trainable=False):
    bn_axis = 3
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)
    P3_up = Conv2D(256, (3, 3), activation='relu', padding='same', name='P3_up_conv4', kernel_initializer='glorot_normal', bias_initializer='zeros',use_bias=True, trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)


    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(x)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='P4_up_conv4',kernel_initializer='glorot_normal', bias_initializer='zeros',use_bias=True,trainable=trainable)(x)
    P4_up = Deconvolution2D(512, kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(x4)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(x)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='P5_up_conv4', kernel_initializer='glorot_normal',bias_initializer='zeros', use_bias=True, trainable=trainable)(x)
    P5_up = Deconvolution2D(512, kernel_size=8, strides=4, padding='same',kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(x5)

    print('P3_up: ', P3_up._keras_shape[1:])
    print('P4_up: ', P4_up._keras_shape[1:])
    print('P5_up: ', P5_up._keras_shape[1:])
    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P3_up, P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',trainable=trainable)(conc)
    feat = BatchNormalization(axis=bn_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)

    x_regr = Convolution2D(num_scale, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    if offset:
        x_offset = Convolution2D(2, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                                 name='offset_regr', trainable=trainable)(feat)

        return [x_class, x_regr, x_offset]
    else:
        return [x_class, x_regr]

def nn_p4p5(img_input=None, offset=True, num_scale=1, trainable=False):
    bn_axis = 3
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)


    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(x)
    P4_up = Deconvolution2D(512, kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(x)
    P5_up = Deconvolution2D(512, kernel_size=8, strides=4, padding='same',kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(x)


    print('P4_up: ', P4_up._keras_shape[1:])
    print('P5_up: ', P5_up._keras_shape[1:])
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',trainable=trainable)(conc)
    feat = BatchNormalization(axis=bn_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)

    x_regr = Convolution2D(num_scale, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    if offset:
        x_offset = Convolution2D(2, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                                 name='offset_regr', trainable=trainable)(feat)

        return [x_class, x_regr, x_offset]
    else:
        return [x_class, x_regr]


def nn_p3p4p5df(img_input=None, offset=True, num_scale=1, trainable=False):
    bn_axis = 3
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)
    P3_up = Conv2D(256, (3, 3), activation='relu', padding='same', name='P3_up_conv4', kernel_initializer='glorot_normal', bias_initializer='zeros',use_bias=True, trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)


    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(x)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='P4_up_conv4',kernel_initializer='glorot_normal', bias_initializer='zeros',use_bias=True,trainable=trainable)(x)
    P4_up = Deconvolution2D(512, kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(x4)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(x)
    P5_up = Deconvolution2D(512, kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(x)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='P5_up_conv1', kernel_initializer='glorot_normal', trainable=trainable)(P5_up)





    print('P3_up: ', P3_up._keras_shape[1:])
    print('P4_up: ', P4_up._keras_shape[1:])
    print('P5_up: ', P5_up._keras_shape[1:])
    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P3_up, P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',trainable=trainable)(conc)
    feat = BatchNormalization(axis=bn_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)

    x_regr = Convolution2D(num_scale, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    if offset:
        x_offset = Convolution2D(2, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                                 name='offset_regr', trainable=trainable)(feat)

        return [x_class, x_regr, x_offset]
    else:
        return [x_class, x_regr]



def nn_p3p4p5r(img_input=None, offset=True, num_scale=1, trainable=False):
    bn_axis = 3
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)
    P3_up = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(x)
    P4_up = Deconvolution2D(512, kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(x)

    P5_up = Deconvolution2D(512, kernel_size=8, strides=4, padding='same',kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(x)

    print('P3_up: ', P3_up._keras_shape[1:])
    print('P4_up: ', P4_up._keras_shape[1:])
    print('P5_up: ', P5_up._keras_shape[1:])

    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P3_up, P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',trainable=trainable)(conc)
    feat = BatchNormalization(axis=bn_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)

    x_regr = Convolution2D(num_scale, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    if offset:
        x_offset = Convolution2D(2, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                                 name='offset_regr', trainable=trainable)(feat)

        return [x_class, x_regr, x_offset]
    else:
        return [x_class, x_regr]



def nn_p3p4p5_1(img_input=None, offset=True, num_scale=1, trainable=False):
    bn_axis = 3
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)
    P3_up = Conv2D(256, (1, 1), activation='relu', name='P3_up_conv4', kernel_initializer='glorot_normal', bias_initializer='zeros',use_bias=True, trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)


    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(x)
    x4 = Conv2D(512, (1, 1), activation='relu', name='P4_up_conv4',kernel_initializer='glorot_normal', bias_initializer='zeros',use_bias=True,trainable=trainable)(x)
    P4_up = Deconvolution2D(512, kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(x4)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(x)
    x5 = Conv2D(512, (1, 1), activation='relu', name='P5_up_conv4', kernel_initializer='glorot_normal',bias_initializer='zeros', use_bias=True, trainable=trainable)(x)
    P5_up = Deconvolution2D(512, kernel_size=8, strides=4, padding='same',kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(x5)

    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P3_up, P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',trainable=trainable)(conc)
    feat = BatchNormalization(axis=bn_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)

    x_regr = Convolution2D(num_scale, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    if offset:
        x_offset = Convolution2D(2, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                                 name='offset_regr', trainable=trainable)(feat)

        return [x_class, x_regr, x_offset]
    else:
        return [x_class, x_regr]


def nn_p3p4p5_seg(img_input=None, offset=True, num_scale=1, trainable=False):
    bn_axis = 3
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)
    P3_up = Conv2D(256, (3, 3), activation='relu', padding='same', name='P3_up_conv4', kernel_initializer='glorot_normal', bias_initializer='zeros',use_bias=True, trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)


    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(x)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='P4_up_conv4',kernel_initializer='glorot_normal', bias_initializer='zeros',use_bias=True,trainable=trainable)(x)
    P4_up = Deconvolution2D(512, kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(x4)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(x)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='P5_up_conv4', kernel_initializer='glorot_normal',bias_initializer='zeros', use_bias=True, trainable=trainable)(x)
    P5_up = Deconvolution2D(512, kernel_size=8, strides=4, padding='same',kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(x5)

    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P3_up, P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',trainable=trainable)(conc)
    feat = BatchNormalization(axis=bn_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_seg = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='seg', trainable=trainable)(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)

    x_regr = Convolution2D(num_scale, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    if offset:
        x_offset = Convolution2D(2, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                                 name='offset_regr', trainable=trainable)(feat)

        return [x_class, x_regr, x_offset,x_seg]
    else:
        return [x_class, x_regr]





def VGG16_SEG(input_tensor=None,include_top=True,classes=1, trainable=True):

    img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=trainable)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=trainable)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(x)
    seg = Conv2D(1, (1, 1), activation='relu', padding='same', name='seg',  kernel_initializer='glorot_normal',
                 bias_initializer=prior_probability_onecls(probability=0.01),
                 trainable=trainable)(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', trainable=trainable)(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1', trainable=trainable)(x)
        x = Dense(4096, activation='relu', name='fc2', trainable=trainable)(x)
        #x = Dense(classes, activation='softmax', name='predictions', trainable=trainable)(x)
        x = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal',
                          bias_initializer=prior_probability_onecls(probability=0.01), trainable=trainable,
                          name='predictionsig')(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(trainable=trainable)(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(trainable=trainable)(x)

    return [x,seg]



def VGGCAM(nb_classes, num_input_channels=1024):
    """
    Build Convolution Neural Network
    args : nb_classes (int) number of classes
    returns : model (keras NN) the Neural Net model
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))#p1

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))#p2

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))#p3

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))#p4

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))

    # Add another conv layer with ReLU + GAP
    model.add(Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same"))
    model.add(AveragePooling2D((14, 14)))
    model.add(Flatten())
    # Add the W layer
    model.add(Dense(nb_classes, activation='softmax'))

    model.name = "VGGCAM"

    return model




