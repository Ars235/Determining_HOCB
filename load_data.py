import numpy as np
import pandas as pd
import torch
import os
import cv2
import math
import datetime
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result, rot_mat


class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, nfeatures, train_path='/app/data', data_file='/app/prep_data.csv'):
        print('Dataset initializing...')
        print(f'Train path {train_path}, Data file {data_file}')
        self.files = pd.read_csv(data_file)

        # Print example to be sure
        print('Example: ', self.files.iloc[23]['image1'])

        self.nfeatures = nfeatures
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        i = idx // 2
        j = (idx % 2) + 1

        file_name = self.files.iloc[i][f'image{j}']

        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        # image = image[150:1650, 400:1650]
        warped, M = rotate_image(image, -90)

        sift = self.sift

        # extract keypoints of the image pair using SIFT
        kp1, descs1 = sift.detectAndCompute(image, None)
        kp2, descs2 = sift.detectAndCompute(warped, None)

        # limit the number of keypoints
        kp1_num = min(self.nfeatures, len(kp1))
        kp2_num = min(self.nfeatures, len(kp2))
        kp1 = kp1[:kp1_num]
        kp2 = kp2[:kp2_num]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])
        kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2])

        # skip this image pair if no keypoints detected in image
        if len(kp1) < 1 or len(kp2) < 1:
            return {
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image,
                'image1': warped,
                'file_name': file_name
            }

        # confidence of each key point
        scores1_np = np.array([kp.response for kp in kp1])
        scores2_np = np.array([kp.response for kp in kp2])

        kp1_np = kp1_np[:kp1_num, :]
        kp2_np = kp2_np[:kp2_num, :]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]

        matched = self.matcher.match(descs1, descs2)
        kp1_projected = cv2.warpAffine(kp1_np.reshape((1, -1, 2)), M, kp1_np.shape[1::-1])[0, :, :]
        dists = cdist(kp1_projected, kp2_np)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.)
        descs2 = np.transpose(descs2 / 256.)

        image = torch.from_numpy(image / 255.).double()[None].cuda()
        warped = torch.from_numpy(warped / 255.).double()[None].cuda()
        print(all_matches)
        return {
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': image,
            'image1': warped,
            'all_matches': list(all_matches),
            'file_name': file_name
        }
