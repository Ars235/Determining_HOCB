# from functions import root_dir
import os
import cv2
import numpy as np

root_dir = 'd:'


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result, rot_mat


def get_boundary_pixels(mask, radius):
    # boundary pixels[i,j] = True if point [i,j] has black neighours in radius
    boundary_pixels = np.zeros(np.shape(mask))
    for i in range(np.shape(mask)[0]):
        if not i % 200:
            print('i: ', i)
        for j in range(np.shape(mask)[1]):
            left_edge0 = max(i - radius, 0)
            left_edge1 = max(j - radius, 0)
            right_edge0 = min(i + radius, np.shape(mask)[0] - 1)
            right_edge1 = min(j + radius, np.shape(mask)[1] - 1)
            # boundary_pixels[i][j] = 1 - np.prod(mask[left_edge0:right_edge0, left_edge1:right_edge1])
            boundary_pixels[i][j] = np.any(np.any(mask[left_edge0:right_edge0, left_edge1:right_edge1]) == 0)
    return boundary_pixels


# mask0_dir = root_dir + '/data/masks/mask-id1.bmp'
# mask1_dir = root_dir + '/data/masks/mask-id2.bmp'
mask0_dir = root_dir + '/data/masks/mask-id1.jpg'
mask1_dir = root_dir + '/data/masks/mask-id2.jpg'
mask0 = cv2.imread(mask0_dir, 0)  # read mask as one-channel image
mask1 = cv2.imread(mask1_dir, 0)
# print('mask1: ', mask1.shape)
mask1, rot_mat1 = rotate_image(mask1, -2)
mask0 = mask0 / 255  # normalize to [0,1]
mask1 = mask1 / 255
# mask0[np.where(mask0 > 0)[0]] = 1
# mask1[np.where(mask1 > 0)[0]] = 1

# correct pixels
for i in range(np.shape(mask1)[0]):
    for j in range(np.shape(mask1)[1]):
        if mask1[i, j] > 0.5:
            mask1[i, j] = 1
        else:
            mask1[i, j] = 0

for i in range(np.shape(mask0)[0]):
    for j in range(np.shape(mask0)[1]):
        if mask0[i, j] > 0.5:
            mask0[i, j] = 1
        else:
            mask0[i, j] = 0

# count1 = np.sum(np.where(mask1 > 0)[0])
# print('count: ', count1)
# print('count0: ', np.shape(mask1)[0]*np.shape(mask1)[1]-count1)

unique, counts = np.unique(mask0, return_counts=True)
dd = dict(zip(unique, counts))
print('mask0: ', dd)

unique, counts = np.unique(mask1, return_counts=True)
dd = dict(zip(unique, counts))
print('mask1 : ', dd)

mask1_rot = cv2.rotate(mask1, cv2.ROTATE_90_COUNTERCLOCKWISE)
# print('mask0 shape: ', mask0.shape)
# print('max mask0 value: ', np.max(mask0))
# print('min mask0 value: ', np.min(mask0))


# boundary_pixels0 = get_boundary_pixels(mask0, radius=10)
# boundary_pixels1 = get_boundary_pixels(mask1_rot, radius=10)
# np.savez('boundary', boundary_pixels0=boundary_pixels0, boundary_pixels1=boundary_pixels1)

# cv2.imshow('image', cv2.resize(boundary_pixels0, (960, 540)))
# cv2.waitKey(0)
# exit(1)
