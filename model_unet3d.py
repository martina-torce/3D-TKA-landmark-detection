import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate

def unet(input_shape):
    inputs = Input(shape=input_shape)

    # Contracting Path
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    # Bottleneck
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)

    # Expanding Path
    up4 = UpSampling3D(size=(2, 2, 2))(conv3)
    concat4 = concatenate([conv2, up4], axis=-1)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(concat4)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)

    up5 = UpSampling3D(size=(2, 2, 2))(conv4)
    concat5 = concatenate([conv1, up5], axis=-1)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(concat5)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)

    # Output layer
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='unet')

    return model