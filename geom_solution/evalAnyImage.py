import numpy as np
from functions import show_image_with_point, calc_theta_fi, mat_block
import os
import cv2

dx = 100.5254378669789
dy = -139.45637249575182
dz = -49.236899901591315


n = 15

# loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=1)
# _, data, _ = next(iter(loader))
# points = torch.squeeze(data).numpy() * 1920

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ))
image0_path = root_dir + '/data/snapshots-2016-02-26/img-2016-02-26T13-23-27devID1.jpg'
image1_path = root_dir + '/data/snapshots-2016-02-26/img-2016-02-26T13-23-27devID2.jpg'

frame = cv2.imread(image0_path, 0)
frame1 = cv2.imread(image1_path, 0)
frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)

with open('test.npy', 'rb') as f:
    points = np.load(f)

center = np.array([960, 960])

point1 = np.zeros((n, 2))
point2 = np.zeros((n, 2))
fi1 = np.zeros(n)
fi2 = np.zeros(n)
theta1 = np.zeros(n)
theta2 = np.zeros(n)


def right_block():
    block = np.zeros((4, 1))
    block[3, 0] = -1
    print('rb: ', block)
    return block


def right_side_block(dx1, dy1, dz1):
    block = np.zeros((4, 1))
    block[0, 0] = -dx1
    block[1, 0] = -dy1
    block[2, 0] = -dz1
    return block


for k in range(n):
    point1[k, 0] = points[k]
    point1[k, 1] = points[n + k]
    point2[k, 0] = points[2 * n + k]
    point2[k, 1] = points[3 * n + k]
    fi1[k], theta1[k] = calc_theta_fi(center, point1[k, :])
    fi2[k], theta2[k] = calc_theta_fi(center, point2[k, :])

n = 15
b = np.zeros(4 * n)
A = np.zeros((4 * n, 2 * n + 1))
for k in range(n):
    A[4 * k:4 * k + 4, 2 * k:2 * k + 2] = mat_block(theta1[k], fi1[k], theta2[k], fi2[k])
    A[4 * k:4 * k + 4, 2 * n] = np.squeeze(right_block())
for k in range(n):
    b[4 * k:4 * k + 4] = np.squeeze(right_side_block(dx, dy, dz))

print('A: ', A)
print('b: ', b)

# show_image_with_point(frame, point1[0, :])
# show_image_with_point(frame1, point2[0, :])
# show_image_with_point(frame, point1[1, :])
# show_image_with_point(frame1, point2[1, :])
# show_image_with_point(frame, point31)
# show_image_with_point(frame1, point32)
# show_image_with_point(frame, point41)
# show_image_with_point(frame1, point42)

print('shape A: ', A.shape)
print('shape b: ', b.shape)

Apinv = np.linalg.pinv(A)
solution = Apinv.dot(b)

print('solution: ', solution)

check = A.dot(solution) - b
print('check: ', check)
print('H: ', solution[-1])
