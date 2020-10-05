import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate


def create_model(input_shape, C, is_train=True, ki='he_normal', kr=None):
    def _cbr(_x, filters, strides, name):
        with tf.name_scope(name):
            _x = Conv2D(filters, (3, 3), strides=strides, activation=None, padding='same', use_bias=False, kernel_initializer=ki, kernel_regularizer=kr)(_x)
            _x = BatchNormalization(fused=True)(_x)
            _x = ReLU()(_x)
        return _x

    def _block(_x, n_layers, filters, name):
        with tf.name_scope(name):
            for i in range(1, n_layers + 1):
                _x = _cbr(_x, filters=filters, strides=2 if i == 1 else 1, name=f"conv_{i}")
        return _x

    def _upsample(_x, filters, target_size, name):
        with tf.name_scope(name):
            _x = tf.image.resize(_x, size=target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            _x = _cbr(_x, filters=filters, strides=1, name=f"conv")
        return _x

    if is_train:
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = input_layer
    else:
        raise NotImplementedError

    x1 = _block(x, n_layers=4, filters=C, name='Block1')
    x2 = _block(x1, n_layers=6, filters=2 * C, name='Block2')
    x3 = _block(x2, n_layers=6, filters=4 * C, name='Block3')

    with tf.name_scope('Upsample'):
        x1 = _cbr(x1, filters=2 * C, strides=1, name='Up1')
        x2 = _upsample(x2, filters=2 * C, target_size=x1.shape[1:3], name='Up2')
        x3 = _upsample(x3, filters=2 * C, target_size=x1.shape[1:3], name='Up3')

    with tf.name_scope('Head'):
        x = Concatenate(axis=-1, name='Concat')([x1, x2, x3])

        cls_map = Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='cls', kernel_initializer='glorot_normal', kernel_regularizer=kr)(x)
        hwl_map = Conv2D(3, kernel_size=(1, 1), padding='same', activation=None, name='hwl', kernel_initializer=ki, kernel_regularizer=kr)(x)
        xyz_map = Conv2D(3, kernel_size=(1, 1), padding='same', activation=None, name='xyz', kernel_initializer=ki, kernel_regularizer=kr)(x)
        angle_map = Conv2D(2, kernel_size=(1, 1), padding='same', activation=None, name='angle', kernel_initializer=ki, kernel_regularizer=kr)(x)
        output_map = [cls_map, hwl_map, xyz_map, angle_map]

    return tf.keras.models.Model([input_layer], output_map)
