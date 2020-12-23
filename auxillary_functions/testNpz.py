import numpy as np
import cv2

boundary = np.load('boundary.npz')
# print('boundary0: ', boundary['boundary_pixels0'])
# print('boundary1: ', boundary['boundary_pixels1'])
boundary_pixels0 = boundary['boundary_pixels0']
boundary_pixels1 = boundary['boundary_pixels1']

# cv2.imshow('image', cv2.resize(boundary_pixels0, (960, 540)))
# cv2.waitKey(0)
#
# cv2.imshow('image', cv2.resize(boundary_pixels1, (960, 540)))
# cv2.waitKey(0)

# x = range(1, 6)
# for n in x:
#   print(n)