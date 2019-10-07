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
