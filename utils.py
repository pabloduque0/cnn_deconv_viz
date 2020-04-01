import numpy as np

def get_coordinates(data_list):

    coord_data_list = []

    for data_set in data_list:

        assert data_set.ndim == 4
        imgs, rows, columns, channels = data_set.shape
        grid_rows, grid_columns = np.mgrid[0:rows,0:columns]
        grid_rows = np.expand_dims(np.repeat(np.expand_dims(grid_rows, 0), imgs, axis=0), -1) / (rows - 1)
        grid_columns = np.expand_dims(np.repeat(np.expand_dims(grid_columns, 0), imgs, axis=0), -1) / (columns - 1)

        coord_data_list.append(np.concatenate([data_set, grid_rows, grid_columns], axis=-1))


    return coord_data_list


def custom_split(data, subject_slices, indices):

    new_indices = indices - np.arange(len(indices))
    test_data = np.empty((1, *data.shape[1:]))
    validation_data = data[new_indices[0] * subject_slices:(new_indices[0] + 1) * subject_slices]
    data = np.concatenate([data[:new_indices[0] * subject_slices], data[(new_indices[0] + 1) * subject_slices:]])

    for idx in new_indices[1:]:
        subject_imgs = data[idx * subject_slices:(idx + 1) * subject_slices]
        test_data = np.concatenate([test_data, subject_imgs])
        data = np.concatenate([data[:idx * subject_slices], data[(idx + 1) * subject_slices:]])

    return data, np.asanyarray(test_data), validation_data

