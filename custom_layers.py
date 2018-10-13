from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf


class Maxpooling_with_argmaxCPU(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Maxpooling_with_argmaxCPU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1:], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Maxpooling_with_argmaxCPU, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        input_shape = K.shape(x)

        output_shape = [_shape // 2 for _shape in K.int_shape(x)]

        output = K.zeros(*output_shape)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



def unpooling2D(x, **kwargs):

    if 'argmax' not in kwargs:
        raise ValueError('Argmax is needed for unpooling layer')

    argmax = kwargs['argmax']

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

