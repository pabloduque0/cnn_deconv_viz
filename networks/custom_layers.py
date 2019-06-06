from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.layers import MaxPool2D, Cropping3D
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


class CropResize(Cropping3D):

    def __init__(self, height_factor, width_factor, **kwargs):
        self.height_factor = height_factor
        self.width_factor = width_factor
        super(Cropping3D, self).__init__(**kwargs)


    def call(self, inputs):
        output = super(Cropping3D, self).call(inputs)
        output = K.resize_images(output, self.height_factor, self.width_factor,
                                 "channels_last", "nearest")
        return output

    def build(self, input_shape):
        return super(Cropping3D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        cropping_shape = super(Cropping3D, self).compute_output_shape(input_shape)
        resized_shape = [cropping_shape[0],
                        tf.cast(cropping_shape[1]*self.height_factor, "int"),
                        tf.cast(cropping_shape[2]*self.width_factor, "int"),
                        cropping_shape[3]]
        return resized_shape

