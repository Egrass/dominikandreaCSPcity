from keras.layers import *
from keras import backend as K
from keras_layer_L2Normalization import L2Normalization
import numpy as np
import keras, math
from custom_layers import Scale


def nn_p3p4p5(img_input=None, offset=True, num_scale=1, trainable=False):

    growth_rate = 32
    nb_filter = 64
    reduction = 0.5
    dropout_rate = 0.0
    weight_decay = 1e-4,
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    concat_axis = 3

    # From architecture for ImageNet (Table 1 in the paper)
    nb_layers = [6,12,32,32] # For DenseNet-161

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False, trainable=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x, nb_filter = dense_block(x, 2, nb_layers[0], nb_filter, growth_rate, trainable,dropout_rate=dropout_rate, weight_decay=weight_decay)
    x,stage2 = transition_block(x, 2, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)# Add transition_block
    nb_filter = int(nb_filter * compression)
    print('stage2: ', stage2._keras_shape[1:])

    x, nb_filter = dense_block(x, 3, nb_layers[1], nb_filter, growth_rate, trainable,dropout_rate=dropout_rate, weight_decay=weight_decay)
    x,stage3 = transition_block(x, 3, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)# Add transition_block
    nb_filter = int(nb_filter * compression)
    print('stage3: ', stage3._keras_shape[1:])


    x, nb_filter = dense_block(x, 4, nb_layers[2], nb_filter, growth_rate, trainable, dropout_rate=dropout_rate, weight_decay=weight_decay)
    x ,stage4 = transition_block2(x, 4, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)# Add transition_block
    nb_filter = int(nb_filter * compression)
    print('stage4: ', stage4._keras_shape[1:])

    #stage 5
    x, nb_filter = dense_block2(x, 5, nb_layers[-1], nb_filter, growth_rate, trainable, dila=(2, 2),dropout_rate=dropout_rate, weight_decay=weight_decay)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(5)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(5)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(5)+'_blk')(x)
    print('stage5: ', x._keras_shape[1:])


    P3_up = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',
                            kernel_initializer='glorot_normal', name='P3up', trainable=trainable)(stage3)
    # print('P3_up: ', P3_up._keras_shape[1:])
    P4_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(stage4)
    # print('P4_up: ', P4_up._keras_shape[1:])
    P5_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(x)
    # print('P5_up: ', P5_up._keras_shape[1:])

    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P3_up, P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',
                         trainable=trainable)(conc)
    feat = BatchNormalization(axis=concat_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)
    x_regr = Convolution2D(1, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    x_offset = Convolution2D(2, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                                 name='offset_regr', trainable=trainable)(feat)
    return [x_class, x_regr, x_offset]



def conv_block(x, stage, branch, nb_filter, trainable, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def conv_block2(x, stage, branch, nb_filter, trainable,  dila=(1,1),dropout_rate=None, weight_decay=1e-4):

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False, trainable=trainable)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    #x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, (3, 3),padding='same', dilation_rate=dila, name=conv_name_base+'_x2', use_bias=False, trainable=trainable)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)
    stage = 'stage' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    stage = x
    x = Convolution2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x,stage

def transition_block2(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)
    stage = 'stage' + str(stage)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    stage = x
    x = Convolution2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((3, 3), padding='same',strides=(1, 1), name=pool_name_base)(x)

    return x,stage

def dense_block(x, stage, nb_layers, nb_filter, growth_rate, trainable, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = Concatenate(axis=concat_axis)([concat_feat, x])
        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def dense_block2(x, stage, nb_layers, nb_filter, growth_rate, trainable,dila=(1,1),  dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1

        x = conv_block2(concat_feat, stage, branch, growth_rate, trainable,dila,dropout_rate, weight_decay)

        concat_feat = Concatenate(axis=concat_axis)([concat_feat, x])
        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter



def prior_probability_onecls(num_class=1, probability=0.01):
	def f(shape, dtype=keras.backend.floatx()):
		assert(shape[0] % num_class == 0)
		# set bias to -log((1 - p)/p) for foregound
		result = np.ones(shape, dtype=dtype) * -math.log((1 - probability) / probability)
		# set bias to -log(p/(1 - p)) for background
		return result
	return f
