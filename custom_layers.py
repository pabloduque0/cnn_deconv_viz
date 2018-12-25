from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.layers import MaxPool2D, UpSampling2D, Layer
import numpy as np
from numpy.lib.stride_tricks import as_strided

class MaxPoolingWithArgmax2D(MaxPool2D):

    def __init__(self, **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)

    def call(self, inputs):
        output = super(MaxPoolingWithArgmax2D, self).call(inputs)
        sum_grads = K.gradients(K.sum(output), inputs)

        zero = tf.constant(0, dtype=tf.float32)
        argmax = tf.not_equal(sum_grads[0], zero)
        argmax_mask = tf.cast(argmax, tf.float32)

        return [output, argmax_mask]

    def build(self, input_shape):
        return super(MaxPoolingWithArgmax2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        pooling_shape = super(MaxPoolingWithArgmax2D, self).compute_output_shape(input_shape)
        argmax_shape = input_shape
        return [pooling_shape, argmax_shape]


def unpooling_with_argmax2D(x, poolsize, argmax):

    """
    Channels last only
    :param x:
    :param poolsize:
    :return:
    """

    unpooled_layer = K.repeat_elements(x, poolsize[0], axis=1)
    unpooled_layer = K.repeat_elements(unpooled_layer, poolsize[1], axis=2)

    x_copy = tf.identity(unpooled_layer)
    for i in range(2):
        x_copy = tf.reduce_mean(x_copy, axis=1)

    _, indices = tf.nn.top_k(x_copy, K.int_shape(argmax)[-1])
    selected_maps = get_selected_maps(unpooled_layer, indices, tf.shape(argmax))

    unpooled_with_values = argmax * selected_maps
    return unpooled_with_values


def get_selected_maps(tensor_input, indices, new_shape):

    all_indices = tf.where(tf.equal(tensor_input, tensor_input))
    to_take = tf.tile(tf.Variable([True]), [new_shape[-1]])
    to_leave = tf.tile(tf.Variable([False]), [tf.shape(tensor_input)[-1]-new_shape[-1]])

    one_img_mask = tf.concat([to_take, to_leave], axis=0)
    multiples = tf.cast(tf.shape(all_indices)[0] / tf.shape(one_img_mask)[0], "int64")
    full_mask = tf.tile(one_img_mask, [multiples])
    cropped_indexes = tf.boolean_mask(all_indices, full_mask)

    modified_idxs = cropped_indexes[:, :3]
    selected_maps = tf.cast(tf.reshape(indices, (-1,)), "int64")
    repetitions = tf.cast(tf.shape(modified_idxs)[0] / tf.shape(selected_maps)[0], "int64")
    selected_maps = tf.tile(selected_maps, [repetitions])
    modified_idxs = tf.concat([modified_idxs, tf.expand_dims(selected_maps, -1)], -1)

    final_maps = tf.gather_nd(tensor_input, modified_idxs)
    final_maps = tf.reshape(final_maps, new_shape)

    return final_maps


def reverse_upconcat(tensor_input, height_factor, width_factor):
    input_shape = K.int_shape(tensor_input)
    output = tf.image.resize_nearest_neighbor(
        tensor_input,
        [int(input_shape[1] * height_factor), int(input_shape[2] * width_factor)],
        align_corners=False,
        name=None
    )
    return output

def reverse_upconcat_output_shape(input_shape, height_factor, width_factor):
    input_shape = list(input_shape)
    new_shape = (input_shape[1]*height_factor, input_shape[2]*width_factor, input_shape[3])
    return new_shape


def unpoolingMask2D_output_shape(input_shape):

    """
    Channels last only. (2, 2) unpooling only
    :param input_shape:
    :param poolsize:
    :return:
    """
    output_shape = [*input_shape]

    if len(output_shape) == 3:
        output_shape[0] *= 2
        output_shape[1] *= 2
    elif len(output_shape) == 4:
        output_shape[1] *= 2
        output_shape[2] *= 2

    return [tuple(output_shape)]



def get_ordered_argmax(argmax, poolsize):

    _, rows, columns, channels = K.int_shape(argmax)

    ordered_indices = None
    for chan in range(channels):
        for i in range(0, rows, poolsize[0]):
            for j in range(0, columns, poolsize[1]):
                pool_section = argmax[i:i + poolsize[0], j:j + poolsize[1]]

                flat_argmax = tf.cast(tf.argmax(K.flatten(pool_section)), tf.int32)

                # convert indices into 2D coordinates
                argmax_row = flat_argmax // tf.shape(pool_section)[1] + i
                argmax_col = flat_argmax % tf.shape(pool_section)[1] + j

                # stack and return 2D coordinates
                argmax = tf.Variable([tf.stack([argmax_row, argmax_col], axis=0)])

                if ordered_indices is None:
                    ordered_indices = argmax
                else:
                    ordered_indices = K.concatenate([ordered_indices, argmax], axis=0)

    return ordered_indices


def get_ordered_argmax_fast(input_data, poolsize):

    input_data = K.eval(input_data)
    batch, rows, columns, channels = input_data.shape

    poolsize = np.array(poolsize)

    arr_in = np.ascontiguousarray(input_data)
    new_shape = tuple(np.array(input_data.shape) // poolsize) + tuple(poolsize)
    new_strides = tuple(input_data.strides * poolsize) + input_data.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)
    reshape_to = list(arr_out.shape)
    reshape_to[-1], reshape_to[-2] = np.cumprod(poolsize)[-1], 1
    reshaped = np.reshape(arr_out, reshape_to)
    argmax = np.argmax(reshaped, axis=3)

    argmax_row = argmax // poolsize[1]
    argmax_col = argmax % poolsize[1]

    section_indexes = np.concatenate([np.expand_dims(np.ravel(argmax_row), 1),
                                      np.expand_dims(np.ravel(argmax_col), 1)], axis=1)

    section_indexes[:, 0] += np.repeat(np.arange(0,
                                                 input_data.shape[0],
                                                 poolsize[0]),
                                       input_data.shape[0] // poolsize[0])

    section_indexes[:, 1] += np.tile(np.arange(0,
                                               input_data.shape[1],
                                               poolsize[1]),
                                     input_data.shape[1] // poolsize[1])

    return tf.Variable(section_indexes)

def get_ordered_argmax_mid(input_data, poolsize):

    data_blocks_rows = tf.split(input_data, poolsize, 0)

    ordered_indices = None
    for block in data_blocks:

        flat_argmax = tf.cast(tf.argmax(K.flatten(block)), tf.int32)

        # convert indexes into 2D coordinates
        argmax_row = flat_argmax // tf.shape(pool_section)[1] + i
        argmax_col = flat_argmax % tf.shape(pool_section)[1] + j

        # stack and return 2D coordinates
        argmax = tf.Variable([tf.stack([argmax_row, argmax_col], axis=0)])

        if ordered_indices is None:
            ordered_indices = argmax
        else:
            ordered_indices = K.concatenate([ordered_indices, argmax], axis=0)



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


"""
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

"""