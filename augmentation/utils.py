import numpy as np


def make_indices_groups(images, size_group=None, n_groups=None):

    if size_group is None and n_groups is None:
        raise Exception("Either size_group or n_groups must be passed to generate indices groups.")

    if n_groups is None:
        n_groups = len(images) // size_group

    indices_left = set(np.arange(len(images)))
    list_groups = []
    for i in range(n_groups):

        if i < n_groups - 1:
            group = np.random.choice(list(indices_left), size_group)
            indices_left = indices_left.difference(set(group))
        else:
            group = np.random.choice(list(indices_left), len(indices_left))

        list_groups.append(group)

    return list_groups

