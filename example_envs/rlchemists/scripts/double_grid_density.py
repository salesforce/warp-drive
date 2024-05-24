import numpy as np


def double_the_grid(en_array_file_name):
    original_array = np.load(en_array_file_name)

    shape = original_array.shape

    new_shape = (2 * shape[0] - 1, 2 * shape[1] - 1, 2 * shape[2] - 1)
    new_array = np.zeros(new_shape)
    new_array[::2, ::2, ::2] = original_array

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                # the odd index is the newly added while even is the old
                r_i = i % 2
                r_j = j % 2
                r_k = k % 2

                # original vertices of the array (all indices are even)
                if r_i == 0 and r_j == 0 and r_k == 0:
                    continue

                assert (new_array[i, j, k] == 0)
                # cell center (all indices are odds)
                if r_i + r_j + r_k == 3:
                    new_array[i, j, k] = (new_array[i - 1, j - 1, k - 1] + new_array[i + 1, j - 1, k - 1] + new_array[
                        i - 1, j + 1, k - 1] + new_array[i + 1, j + 1, k - 1] +
                                          new_array[i - 1, j - 1, k + 1] + new_array[i + 1, j - 1, k + 1] + new_array[
                                              i - 1, j + 1, k + 1] + new_array[i + 1, j + 1, k + 1]) / 8
                # surface center (two indices are odds)
                elif r_i + r_j + r_k == 2:
                    # on the i surface
                    if r_i == 0:
                        new_array[i, j, k] = (new_array[i, j - 1, k - 1] + new_array[i, j - 1, k + 1] + new_array[
                            i, j + 1, k - 1] + new_array[i, j + 1, k + 1]) / 4
                    # on the j surface
                    elif r_j == 0:
                        new_array[i, j, k] = (new_array[i - 1, j, k - 1] + new_array[i - 1, j, k + 1] + new_array[
                            i + 1, j, k - 1] + new_array[i + 1, j, k + 1]) / 4
                    # on the k surface
                    else:
                        new_array[i, j, k] = (new_array[i - 1, j - 1, k] + new_array[i - 1, j + 1, k] + new_array[
                            i + 1, j - 1, k] + new_array[i + 1, j + 1, k]) / 4
                # edge middle
                else:
                    if r_i == 1:
                        new_array[i, j, k] = (new_array[i - 1, j, k] + new_array[i + 1, j, k]) / 2
                    elif r_j == 1:
                        new_array[i, j, k] = (new_array[i, j - 1, k] + new_array[i, j + 1, k]) / 2
                    else:
                        new_array[i, j, k] = (new_array[i, j, k - 1] + new_array[i, j, k + 1]) / 2

    return new_array


if __name__ == "__main__":

    en_array_file = "../en_array/en_array.npy" # The original en array
    assert ".npy" in en_array_file
    new_array = double_the_grid(en_array_file)

    output_file = en_array_file.split(".npy")[0] + "_double_grids_check.npy"
    with open(output_file, 'wb') as output_file:
        np.save(output_file, new_array)
