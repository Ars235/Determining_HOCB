from solveSystem import matrices
from functions import shuffle, train_dataset, get_good_batch, h_max
import torch
import numpy as np

n = 15

batch_size = 512
print('shuffle: ', shuffle)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size)

h_label_norm, stat_data = get_good_batch(train_dataloader)
p = stat_data.cpu().numpy()
h_label_norm = np.squeeze(h_label_norm.cpu().numpy())

# np.savez('points', h_label_norm=h_label_norm, points=points)
# data = np.load('points.npz')
# h_label_norm = np.squeeze(data['h_label_norm'])
# p = data['points']

h = h_label_norm * h_max
p = p*1920
batch_size = np.shape(h)[0]
print('batch size out: ', batch_size)
# print('p: ', p)
# print('h: ', h)

A = np.zeros((4 * n * batch_size, 2 * n + 3))
b = np.zeros(4 * n * batch_size)

# print('A shape: ', A.shape)
# print('b shape: ', b.shape)

for k in range(batch_size):
    points = p[k, :]
    Ak, bk = matrices(points, h[k])
    A[4 * k * n: 4 * (k + 1) * n, :] = Ak
    b[4 * k * n: 4 * (k + 1) * n] = bk

# print('shape A: ', A.shape)
# print('shape b: ', b.shape)

print('start solving: ')

pinvA = np.linalg.pinv(A)
solution = pinvA.dot(b)

dx = solution[-3]
dy = solution[-2]
dz = solution[-1]

print('dx: ', dx)
print('dy: ', dy)
print('dz: ', dz)
