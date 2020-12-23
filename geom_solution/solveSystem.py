import numpy as np
import cv2
import os
import pandas as pd
import torch
from functions import train_dataset, h, calc_theta_fi, show_image_with_point, mat_block

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ))
image0_path = root_dir + '/data/snapshots-2016-02-26/img-2016-02-26T13-23-27devID1.jpg'
image1_path = root_dir + '/data/snapshots-2016-02-26/img-2016-02-26T13-23-27devID2.jpg'
frame = cv2.imread(image0_path, 0)
frame1 = cv2.imread(image1_path, 0)
frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
h0 = h[0]
print('h0: ', h0)
n = 15

center_x = np.shape(frame)[0] / 2
center_y = np.shape(frame)[1] / 2
print('center_x: ', center_x)
print('center_y: ', center_y)
center = np.array([center_x, center_y])

with open('test.npy', 'rb') as f:
    points = np.load(f)

point1 = np.zeros((n, 2))
point2 = np.zeros((n, 2))
fi1 = np.zeros(n)
fi2 = np.zeros(n)
theta1 = np.zeros(n)
theta2 = np.zeros(n)


# def right_block():
#     block = np.zeros((4, 3))
#     block[0, 0] = 1
#     block[1, 1] = 1
#     block[2, 2] = 1
#     return block

def right_block():
    block = np.zeros((4, 3))
    block[0, 0] = 1
    block[1, 1] = 1
    block[2, 2] = 1
    # block[3, 0] = 1
    return block


def right_side(hh):
    block = np.zeros(4)
    block[0] = 0
    block[1] = 0
    block[2] = 0
    block[3] = hh
    return block


def matrices(points, h0):
    for k in range(n):
        point1[k, 0] = points[k]
        point1[k, 1] = points[n + k]
        point2[k, 0] = points[2 * n + k]
        point2[k, 1] = points[3 * n + k]
        fi1[k], theta1[k] = calc_theta_fi(center, point1[k, :])
        fi2[k], theta2[k] = calc_theta_fi(center, point2[k, :])

    b = np.zeros(4 * n)
    A = np.zeros((4 * n, 2 * n + 3))
    for k in range(n):
        A[4*k:4*k+4, 2 * k:2 * k + 2] = mat_block(theta1[k], fi1[k], theta2[k], fi2[k])
        A[4 * k:4 * k + 4, 2 * n: 2 * n + 3] = np.squeeze(right_block())
    for k in range(n):
        b[4 * k:4 * k + 4] = right_side(h0)
    return A, b


A, b = matrices(points, h0)

print('fi1: ', fi1[0])
print('fi2: ', fi2[0])
print('fi1: ', fi1[2])
print('fi2: ', fi2[2])
show_image_with_point(frame, point1[0])
show_image_with_point(frame1, point2[0])
show_image_with_point(frame, point1[2])
show_image_with_point(frame1, point2[2])

# print('shape A: ', A.shape)
# print('shape b: ', b.shape)

pinvA = np.linalg.pinv(A)
# print('pinvA shape: ', pinvA.shape)
solution = pinvA.dot(b)

dx = solution[-3]
dy = solution[-2]
dz = solution[-1]

print('dx: ', dx)
print('dy: ', dy)
print('dz: ', dz)
# print('a: ', a)

r1 = np.zeros(n)
r2 = np.zeros(n)
for k in range(n):
    r1[k] = solution[2 * k]
    r2[k] = solution[2 * k + 1]
H1 = np.multiply(r1, np.cos(theta1))
H2 = np.multiply(r2, np.cos(theta2))
height = 0.5*(np.mean(H1) + np.mean(H2))
print('H: ', height)


# check = A.dot(solution) - b
# print('check: ', check)
