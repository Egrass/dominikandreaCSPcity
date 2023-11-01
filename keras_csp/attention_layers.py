from keras.engine import Layer, InputSpec
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import keras.backend as K
from keras.layers import Activation, Conv2D
import tensorflow as tf
from keras.layers import *

class CAM(Layer):
    '''Custom Layer for DenseNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9,  gamma_init='zero', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(CAM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (1,)

        # Tensorflow >= 1.0.0 compatibility
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))

        self.trainable_weights = [self.gamma]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = x.get_shape().as_list()
        _, h, w, filters = input_shape
        if h ==None:
            #h = 84
            #h = 120
            h = 160
            h = 256
        if w == None:
            # w = 112
            #w = 160
            w = 320
            w = 512

        vec_a = K.reshape(x, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma * aaTa+ x
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(CAM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PAM(Layer):
    '''Custom Layer for DenseNet used for BatchNormalization.

    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, gamma_init='zero', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(PAM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (1,)

        # Tensorflow >= 1.0.0 compatibility
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))

        self.trainable_weights = [self.gamma]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = x.get_shape().as_list()
        _, h, w, filters = input_shape
        if h ==None:
          h = 84
          # h = 120
        if w == None:
           w = 112
        #w = 160

        b = Conv2D(filters // 8, 1, trainable=True,use_bias=False, kernel_initializer='he_normal')(x)
        c = Conv2D(filters // 8, 1, trainable=True,use_bias=False, kernel_initializer='he_normal')(x)
        d = Conv2D(filters, 1, trainable=True,use_bias=False, kernel_initializer='he_normal')(x)

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma * bcTd + x
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(PAM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
