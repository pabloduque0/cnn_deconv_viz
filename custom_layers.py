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
        return output, argmax

    def build(self, input_shape):
        return super(MaxPoolingWithArgmax2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return super(MaxPoolingWithArgmax2D, self).compute_output_shape(input_shape)


def unpooling2D(x, **kwargs):

    if 'argmax' not in kwargs:
        raise ValueError('Argmax is needed for unpooling layer')

    argmax = kwargs['argmax']
    output_shape = [_shape // 2 for _shape in K.int_shape(x)]
    output = K.zeros(*output_shape)

    height = K.shape(output)[0]
    width = K.shape(output)[1]
    channels = K.shape(output)[2]
    # build the indices for a SparseTensor addition like http://stackoverflow.com/a/34686952/3524844

    t1 = tf.to_int64(tf.range(channels))
    t1 = K.tile(t1, [(width // 2) * (height // 2)])
    t1 = K.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = K.reshape(t1, [channels, height // 2, width // 2, 1])

    t2 = K.squeeze(argmax)
    t2 = tf.pack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

    t = tf.concat(3, [t2, t1])
    indices = K.reshape(t, [(height // 2) * (width // 2) * channels, 3])
    # Get the values for max_unpooling (used in addition with argmax location)
    x1 = K.squeeze(x)
    x1 = K.reshape(x1, [-1, channels])
    x1 = K.transpose(x1, perm=[1, 0])
    values = K.reshape(x1, [-1])
    # perform addition
    delta = K.SparseTensor(indices, values, K.to_int64(K.shape(output)))
    return K.expand_dims(K.sparse_tensor_to_dense(K.sparse_reorder(delta)), 0)


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