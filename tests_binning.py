import numpy as np
np.set_printoptions(edgeitems=3, linewidth=120)

data = np.random.randint(0, 20, size=(15, 21))
print('Old dimensions:', data.shape)


def squeeeeze(data: np.array, target_col_count):
    assert data.ndim > 1

    line_count = len(data)
    target_length = 16
    batches = np.array_split(data, target_length, axis=1)
    data_squeezed = np.zeros(line_count)
    for array in batches:
        batch_avg = array.mean(axis=1)
        data_squeezed = np.vstack((data_squeezed, batch_avg))
    data_squeezed = data_squeezed[1:]
    data_squeezed = data_squeezed.T
    return data_squeezed


b = squeeeeze(data, 16)
print(b.shape)