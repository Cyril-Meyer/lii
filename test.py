import sys
import os
import numpy as np
import tensorflow as tf
from lii import LargeImageInference as lii

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


model_2d = tf.keras.Sequential()
model_2d.add(tf.keras.layers.InputLayer((None, None, 1)))

model_3d = tf.keras.Sequential()
model_3d.add(tf.keras.layers.InputLayer((None, None, None, 1)))
model_3d.add(tf.keras.layers.Lambda(lambda x: x))


def apply_copy(x):
    return np.copy(x)


def apply_copy_border_0(x):
    x = np.copy(x)
    x[0:2, 0, 0] = 0
    x[0, 0:2, 0] = 0
    x[0, 0, 0:2] = 0
    return x


def apply_model_2d(x):
    x = x[0]
    return np.expand_dims(model_2d.predict(x), 0)


array_2d_01 = np.arange(65*65).reshape((65, 65))
array_2d_02 = np.arange(63*63).reshape((63, 63))

array_3d_01 = np.arange(32*32*31).reshape((32, 32, 31))
array_3d_02 = np.arange(32*32*31).reshape((32, 32, 31, 1))
array_3d_03 = np.arange(31*31*31).reshape((31, 31, 31, 1))
array_3d_04 = np.arange(33*33*33).reshape((33, 33, 33, 1))
array_3d_05 = np.arange(24*1394*1832).reshape((24, 1394, 1832, 1))
array_3d_06 = np.arange(80*1536*1280).reshape((80, 1536, 1280, 1))
array_3d_07 = np.arange(16*16*16).reshape((16, 16, 16, 1))
array_3d_08 = np.arange(64*128*256).reshape((64, 128, 256, 1))

apply = apply_copy

for i in range(3):
    arr = np.rollaxis(array_3d_08, i)
    for j in range(3):
        overlap = [1, 1, 1]
        overlap[j] = 2
        assert((lii.fast_infer(arr, (32, 32, 32), apply, overlap) == arr).all())

assert((lii.infer(array_3d_05, (16, 512, 512), apply, (2, 2, 2)) == array_3d_05).all())
assert((lii.fast_infer(array_3d_06, (16, 1536, 320), apply, (2, 1, 1)) == array_3d_06).all())
assert((lii.infer(array_3d_06, (16, 1536, 320), apply, (2, 1, 1)) == array_3d_06).all())
assert((lii.fast_infer(array_3d_07, (4, 4, 4), apply, (2, 1, 1)) == array_3d_07).all())
assert((lii.infer(array_3d_07, (4, 4, 4), apply, (2, 1, 1)) == array_3d_07).all())

apply = apply_copy_border_0
assert((lii.infer(array_3d_05, (16, 512, 512), apply, (2, 2, 2)) == array_3d_05).all())
assert((lii.infer(array_3d_06, (16, 1536, 1280), apply, (2, 1, 1)) == array_3d_06).all())

apply = apply_copy
assert((lii.infer2d(array_2d_01, (16, 16), apply, 1)[:, :, 0] == array_2d_01).all())
assert((lii.infer2d(array_2d_01, (16, 16), apply, 2)[:, :, 0] == array_2d_01).all())
assert((lii.infer2d(array_2d_02, (16, 16), apply, 1)[:, :, 0] == array_2d_02).all())
assert((lii.infer2d(array_2d_02, (16, 16), apply, 2)[:, :, 0] == array_2d_02).all())

assert((lii.infer2d(array_2d_01, (32, 32), apply, 1)[:, :, 0] == array_2d_01).all())
assert((lii.infer2d(array_2d_01, (32, 32), apply, 2)[:, :, 0] == array_2d_01).all())
assert((lii.infer2d(array_2d_02, (32, 32), apply, 1)[:, :, 0] == array_2d_02).all())
assert((lii.infer2d(array_2d_02, (32, 32), apply, 2)[:, :, 0] == array_2d_02).all())

assert((lii.infer(array_3d_01, (8, 8, 8), apply, 2)[:, :, :, 0] == array_3d_01).all())
assert((lii.infer(array_3d_02, (8, 8, 8), apply, (1, 2, 2)) == array_3d_02).all())
assert((lii.infer(array_3d_02, (16, 16, 16), apply, (1, 2, 2)) == array_3d_02).all())

assert((lii.infer(array_3d_03, (16, 16, 16), apply, (1, 1, 1)) == array_3d_03).all())
assert((lii.infer(array_3d_03, (16, 16, 16), apply, (1, 1, 2)) == array_3d_03).all())
assert((lii.infer(array_3d_03, (16, 16, 16), apply, (1, 2, 1)) == array_3d_03).all())
assert((lii.infer(array_3d_03, (16, 16, 16), apply, (1, 2, 2)) == array_3d_03).all())
assert((lii.infer(array_3d_03, (16, 16, 16), apply, (2, 1, 1)) == array_3d_03).all())
assert((lii.infer(array_3d_03, (16, 16, 16), apply, (2, 1, 2)) == array_3d_03).all())
assert((lii.infer(array_3d_03, (16, 16, 16), apply, (2, 2, 1)) == array_3d_03).all())
assert((lii.infer(array_3d_03, (16, 16, 16), apply, (2, 2, 2)) == array_3d_03).all())

assert((lii.infer(array_3d_04, (8, 16, 16), apply, (1, 1, 1)) == array_3d_04).all())
assert((lii.infer(array_3d_04, (16, 8, 16), apply, (1, 1, 2)) == array_3d_04).all())
assert((lii.infer(array_3d_04, (16, 16, 8), apply, (1, 2, 1)) == array_3d_04).all())
assert((lii.infer(array_3d_04, (4, 16, 16), apply, (1, 2, 2)) == array_3d_04).all())
assert((lii.infer(array_3d_04, (16, 4, 16), apply, (2, 1, 1)) == array_3d_04).all())
assert((lii.infer(array_3d_04, (16, 16, 4), apply, (2, 1, 2)) == array_3d_04).all())
assert((lii.infer(array_3d_04, (8, 16, 8), apply, (2, 2, 1)) == array_3d_04).all())
assert((lii.infer(array_3d_04, (16, 4, 16), apply, (2, 2, 2)) == array_3d_04).all())

assert((lii.infer(array_3d_05, (16, 512, 512), apply, (2, 2, 2)) == array_3d_05).all())

apply = apply_model_2d
assert((lii.infer2d(array_2d_01, (16, 16), apply, 2)[:, :, 0] == array_2d_01).all())
assert((lii.infer2d(array_2d_02, (16, 16), apply, 1)[:, :, 0] == array_2d_02).all())

assert((lii.infer(array_3d_03, (1, 8, 8), apply, (1, 2, 2)) == array_3d_03).all())

apply = model_3d.predict
assert((lii.infer(array_3d_01, (8, 8, 8), apply, 2)[:, :, :, 0] == array_3d_01).all())
assert((lii.infer(array_3d_02, (8, 8, 8), apply, (1, 2, 2)) == array_3d_02).all())
assert((lii.infer(array_3d_02, (16, 16, 16), apply, (1, 2, 2)) == array_3d_02).all())

print("All tests passed.")
