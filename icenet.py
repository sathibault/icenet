from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, MaxPooling2D, Dense, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D, Flatten
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential

from keras import backend as K

def relu(x):
    return K.relu(x)

def _conv_block(inputs, filters, kernel, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='valid', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu)(x)

def _bottleneck(inputs, filters, kernel, t, s):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = Conv2D(tchannel, (1,1))(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu)(x)

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='valid')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    return x

def make_net(ishape, k):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    inputs = Input(shape=ishape)
    x = _conv_block(inputs, 5, (5,5), (2,2))
    x = _bottleneck(x, 10, (3,3), t=1, s=1)
    x = _bottleneck(x, 20, (3,3), t=2, s=2)
    x = _bottleneck(x, 20, (3,3), t=2, s=1)
    x = _bottleneck(x, 10, (3,3), t=1, s=3)

    x = Dropout(0.2)(x)
    x = Conv2D(k, (5,5))(x)
    #outputs = Reshape((k,))(x)
    outputs = Flatten()(x);
    model = Model(inputs, outputs)
    return model

def make_tutorial(augmentation, ishape, k):
    if augmentation:
        return Sequential([
                augmentation,
                Input(shape=ishape),
                Conv2D(16, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Conv2D(32, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Conv2D(64, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Dropout(0.2),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(k)
                ])
    else:
        return Sequential([
                Input(shape=ishape),
                Conv2D(16, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Conv2D(32, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Conv2D(64, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Dropout(0.2),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(k)
                ])

if __name__ == '__main__':
    model = make_net((80,80,3), 1)
    print(model.summary())
