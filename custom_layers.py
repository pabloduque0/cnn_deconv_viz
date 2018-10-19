from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.layers import MaxPool2D

class MaxPoolingWithArgmax2D(MaxPool2D):

    def __init__(self, **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        output = super(MaxPoolingWithArgmax2D, self).call(inputs)
        argmax = K.gradients(K.sum(output), inputs)

        zero = tf.constant(0, dtype=tf.float32)
        argmax = tf.not_equal(argmax[0], zero)
        argmax = tf.where(argmax)

        return [output, argmax[0]]

    def build(self, input_shape):
        return super(MaxPoolingWithArgmax2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        pooling_shape = super(MaxPoolingWithArgmax2D, self).compute_output_shape(input_shape)
        argmax_shape = input_shape
        return [pooling_shape, argmax_shape]

def unpooling2D(x, **kwargs):

    if 'argmax' not in kwargs:
        raise ValueError('argmax is needed for unpooling layer')

    argmax = kwargs['argmax']
    output_shape = list(K.int_shape(x))
    output_shape[1] *= 2
    output_shape[2] *= 2
    output = K.zeros(tuple(output_shape[1:]))

    height = K.shape(output)[0]
    width = K.shape(output)[1]
    channels = K.shape(output)[2]
    # build the indices for a SparseTensor addition like http://stackoverflow.com/a/34686952/3524844

    t1 = tf.to_float(tf.range(channels))
    t1 = K.tile(t1, [(width // 2) * (height // 2)])
    t1 = K.reshape(t1, [-1, channels])
    print(K.eval(t1), len(K.eval(t1)), len(K.eval(t1[0])))
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = K.reshape(t1, [channels, height // 2, width // 2, 1])

    t2 = tf.squeeze(argmax)
    t2 = tf.stack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

    t = tf.concat([t2, t1], 3)
    indices = K.reshape(t, [(height // 2) * (width // 2) * channels, 3])
    indices = tf.to_int64(indices)
    # Get the values for max_unpooling (used in addition with argmax location)
    x1 = tf.squeeze(x)
    x1 = K.reshape(x1, [-1, channels])
    x1 = tf.transpose(x1, perm=[1, 0])
    values = K.reshape(x1, [-1])
    # perform addition
    delta = tf.SparseTensor(indices, values, tf.to_int64(K.shape(output)))
    final_output = K.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

    return final_output


def get_ordered_argmax(argmax, poolsize):

    if len(argmax.shape) != 3:
        raise ValueError('get_ordered_armax expects a tf.squeezed tensor input')

    rows, columns, channels = K.int_shape(argmax)
    for chan in channels:
        for i, j in zip(range(0, rows, poolsize[0]), range(0, columns, poolsize[1])):
            pass





def call(self, inputs, output_shape=None):
    updates, mask = inputs[0], inputs[1]
    with K.tf.variable_scope(self.name):
        mask = K.cast(mask, 'int32')
        input_shape = K.tf.shape(updates, out_type='int32')
        #  calculation new shape
        if output_shape is None:
            output_shape = (
            input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
        self.output_shape1 = output_shape

        # calculation indices for batch, height, width and feature maps
        one_like_mask = K.ones_like(mask, dtype='int32')
        batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = K.tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = K.tf.size(updates)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values = K.reshape(updates, [updates_size])
        ret = K.tf.scatter_nd(indices, values, output_shape)
        return ret