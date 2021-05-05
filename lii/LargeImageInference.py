import numpy as np


def infer(image, window_shape, f, overlap=1):
    """
    :param image: a 3D or 4D numpy array, channel last, shape = (Z, Y, X, C) or (Z, Y, X)
                  if 3D array, a channel will be added.
    :param window_shape: a 3D tuple, a 3D array or an int (if same value for all axis)
    :param f: a function or class to apply,
    taking a single argument, a 5D numpy array as input, channel last, shape = (B, Z, Y, X, C)
    returning an array of same shape, with fixed number of channel.
    :param overlap: 3D tuple or int, values can be 1 or 2 (1 = no overlap, 2 = overlap)
    :return: a 3D or 4D numpy array
    """
    # parameters processing
    if isinstance(overlap, int):
        overlap = (overlap, overlap, overlap)

    window_shape = np.array(window_shape)
    strides = np.array(window_shape) // overlap

    # assertion
    assert len(image.shape) in [3, 4],\
        f'image shape {image.shape} invalid'
    assert len(window_shape) == 3,\
        f'window shape {window_shape} invalid'
    assert (np.array(overlap) > 0).all() and (np.array(overlap) <= 2).all(),\
        f"overlap {overlap} values invalid"
    assert (strides == np.array(window_shape) / overlap).all(),\
        f"window_shape {window_shape} cannot be divided by overlap {overlap}"
    assert ((strides % overlap) == [0, 0, 0]).all(),\
        "strides cannot be divided by overlap"

    # add a color dim if necessary
    n_dim_in = len(image.shape)
    if n_dim_in == 3:
        image = np.expand_dims(image, -1)

    image_shape = np.array(image.shape)[0:3]
    # pad image (image_p = image with padding)
    image_p_shape = np.ceil(image_shape / strides).astype(np.int32) * strides
    pad = image_p_shape - image_shape
    # image_p = np.pad(image, ((0, pad[0]), (0, pad[1]), (0, pad[2]), (0, 0)), mode='reflect')
    image_p = np.pad(image,
                     ((strides[0], strides[0]+pad[0]),
                      (strides[1], strides[1]+pad[1]),
                      (strides[2], strides[2]+pad[2]),
                      (0, 0)),
                     mode='reflect')
    image_p_shape = image_p.shape

    # loop and infer
    result = None
    # center border
    cb = strides - (strides // overlap)
    for z in range(0, image_p_shape[0]-strides[0], strides[0]):
        for y in range(0, image_p_shape[1]-strides[1], strides[1]):
            for x in range(0, image_p_shape[2]-strides[2], strides[2]):
                # patch prediction
                p = image_p[z:z+window_shape[0], y:y+window_shape[1], x:x+window_shape[2]]
                if not (np.array(p.shape)[0:3] == window_shape).all():
                    raise LookupError

                p = f(np.expand_dims(p, 0))[0]

                if result is None:
                    result = np.zeros((image_p_shape[0], image_p_shape[1], image_p_shape[2], p.shape[-1]), dtype=p.dtype)

                # keep center
                p = p[cb[0]:cb[0]+strides[0], cb[1]:cb[1]+strides[1], cb[2]:cb[2]+strides[2], :]
                result[z+cb[0]:z+cb[0]+strides[0], y+cb[1]:y+cb[1]+strides[1], x+cb[2]:x+cb[2]+strides[2], :] = p

    # remove padding
    result = result[strides[0]:strides[0]+image_shape[0],
                    strides[1]:strides[1]+image_shape[1],
                    strides[2]:strides[2]+image_shape[2], :]
    return result


def infer2d(image, window_shape, f, overlap=1):
    """
    :param image: a 2D or 3D numpy array, channel last, shape = (Y, X, C) or (Y, X)
    :param window_shape: a 2D numpy array
    :param f: see infer()
    :param overlap: see infer()
    :return:
    """
    assert len(image.shape) in [2, 3], \
        f"image shape {image.shape} is incorrect (length is {len(image.shape)} but should be 2 or 3."
    assert len(window_shape) == 2, \
        f"window_shape {window_shape} is incorrect (length is {len(window_shape)} but should be 2."

    image = np.expand_dims(image, 0)
    window_shape = (1,) + window_shape

    if isinstance(overlap, int):
        overlap = (1, overlap, overlap)
    if len(overlap) == 2:
        overlap = (1,) + overlap

    return infer(image, window_shape, f, overlap)[0]
