from typing import Tuple, List, Type, Dict, Any
from torch.utils.data import Dataset
import pandas as pd
import os
from models.matching import Matching
from models.utils import (frame2tensor, make_matching_plot_fast)
import cv2
import matplotlib.cm as cm
import torch
import ntpath
import numpy as np
import getpass
from prepareBoundary import mask0, mask1

username = getpass.getuser()
if username == 'arsfi':
    shuffle = False
    force_cpu = True
else:
    shuffle = True  # if server, then shuffle
    force_cpu = False

device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))

root_dir = 'd:'
print(root_dir)
print('root dir: ', root_dir)

# path_to_target_file = root_dir + '/cbh_meters.csv'
# path_to_target_file = root_dir + '/not_nan_target.csv'
path_to_target_file = root_dir + '/paired_target.csv'
# path_to_target_file = root_dir + '/paired_targetU2000NotNan.csv'

df_target = pd.read_csv(path_to_target_file)
h = df_target['target_value']
h_max = np.max(h)
print('hmax: ', h_max)
print('h_min: ', np.min(h))
img0_paths = df_target['image1']
img1_paths = df_target['image2']
# h = np.log(h)
pic_half_size = 1920 / 2
pixels_norm_factor = 2 * pic_half_size

print('number of images: ', h.shape)

boundary = np.load('d:/SuperGlueDir/GluePretr/boundary.npz')
boundary_pixels0 = boundary['boundary_pixels0']
boundary_pixels1 = boundary['boundary_pixels1']
mask0 = (torch.Tensor(mask0)).to(device)
mask1 = (torch.Tensor(mask1)).to(device)


# calculate percentiles with step=statistics_step
def calc_percentiles(x, statistics_step):
    thresholds = np.arange(statistics_step, 100 - statistics_step, statistics_step)
    perc_array = np.zeros(thresholds.shape)
    for k, perc in enumerate(thresholds):
        perc_array[k] = np.percentile(x, perc)
    return perc_array


class vngoDataset(Dataset):
    def __init__(self, img0_paths1, img1_paths1, h1, n):
        self.img0_paths = img0_paths1
        self.img1_paths = img1_paths1
        self.h = h1
        self.n = n
        print('n: ', self.n)

    def __len__(self):
        return len(self.img0_paths)

    def __getitem__(self, index):
        name0 = self.img0_paths[index]
        name1 = self.img1_paths[index]
        name0 = root_dir + name0[4:]
        name1 = root_dir + name1[4:]
        print(name1, name0)

        # select self.n points with the highest confidence value
        keyp0_valid, keyp1_valid, conf_valid, frame0_brightness, frame1_brightness, not_skip_train, out = getKpts(
            matching, name0, name1, self.n)

        x0_valid = keyp0_valid[:, 0]
        y0_valid = keyp0_valid[:, 1]
        x1_valid = keyp1_valid[:, 0]
        y1_valid = keyp1_valid[:, 1]

        if self.n != -1:
            stat_data = torch.zeros((self.n, 16))
        else:
            stat_data = torch.zeros((x0_valid.shape[0], 16))

        for k in range(x0_valid.shape[0]):
            if not_skip_train:
                fi0, theta0 = calc_theta_fi_torch(pic_half_size, torch.Tensor([x0_valid[k], y0_valid[k]]))
                fi1, theta1 = calc_theta_fi_torch(pic_half_size, torch.Tensor([x1_valid[k], y1_valid[k]]))

                fi0 = torch.Tensor([fi0]).to(device)
                fi1 = torch.Tensor([fi1]).to(device)
                theta0 = torch.Tensor([theta0]).to(device)
                theta1 = torch.Tensor([theta1]).to(device)

                frame0_brightnessk = torch.Tensor([frame0_brightness[k]]).to(device)
                frame1_brightnessk = torch.Tensor([frame1_brightness[k]]).to(device)

                x0_validk = (x0_valid[k] / pixels_norm_factor)
                y0_validk = (y0_valid[k] / pixels_norm_factor)
                x1_validk = (x1_valid[k] / pixels_norm_factor)
                y1_validk = (y1_valid[k] / pixels_norm_factor)

                x0_validk = torch.unsqueeze(x0_validk, dim=0)
                x1_validk = torch.unsqueeze(x1_validk, dim=0)
                y0_validk = torch.unsqueeze(y0_validk, dim=0)
                y1_validk = torch.unsqueeze(y1_validk, dim=0)

                stat_data[k, :] = torch.cat((torch.cos(fi0), torch.cos(fi1), torch.cos(theta0), torch.cos(theta1),
                                             torch.sin(fi0), torch.sin(fi1), torch.sin(theta0), torch.sin(theta1),
                                             x0_validk, y0_validk, x1_validk, y1_validk,
                                             torch.abs(x0_validk - x1_validk), torch.abs(y0_validk - y1_validk),
                                             frame0_brightnessk, frame1_brightnessk))

            else:
                # print('x0 valid shape: ', x0_valid.shape)
                stat_data[k, :] = torch.zeros((1, 16))

        # normalize height and keypoint coordinates
        h_norm = self.h[index] / h_max
        stat_data = stat_data.to(device)
        h_norm = (torch.Tensor([h_norm])).to(device)
        not_skip_train = (torch.Tensor([not_skip_train])).to(device)

        return h_norm, stat_data, not_skip_train, out, name0, name1


def get_good_batch(train_dataloader1):
    h_label, stat_data2, not_skip_train, out, name0, name1 = next(iter(train_dataloader1))
    # print('stat data: ', stat_data2)
    not_skip_train = torch.squeeze(not_skip_train, dim=1)
    # print('not skip: ', not_skip_train)
    indices = torch.squeeze(not_skip_train.nonzero(), dim=1)
    # print('indices: ', indices)
    indices = list(map(int, indices.tolist()))
    # print('indices: ', indices)
    h_selected = h_label[indices]
    out_selected = out[indices]
    name0 = list(name0)
    name1 = list(name1)
    name0_selected = [name0[i] for i in indices]
    name1_selected = [name1[i] for i in indices]
    data_selected = stat_data2[indices]

    # indices = torch.squeeze(torch.nonzero(h_selected < 1500))
    # indices = list(map(int, indices.tolist()))
    # h_selected = h_selected[indices]
    # data_selected = h_selected[indices]

    print('batch out shape: ', h_selected.shape[0])
    return h_selected, data_selected, out_selected, name0_selected, name1_selected


config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        # 'weights': 'outdoor',
        'weights': 'indoor',
        'sinkhorn_iterations': 100,
        # 'match_threshold': 0.25,
        'match_threshold': 0.2,
    }
}


def delete_boundary_kpts(kpts0, kpts1, conf, boundary_pixels_img0, n):
    array_len = np.shape(kpts0)[0]
    index = 0
    while index < array_len:
        v0 = kpts0[index]
        if boundary_pixels_img0[int(v0[0]), int(v0[1])]:
            kpts0 = np.delete(kpts0, index, axis=0)
            kpts1 = np.delete(kpts1, index, axis=0)
            conf = np.delete(conf, index, axis=0)
            index -= 1
            array_len -= 1
        index += 1
    return kpts0, kpts1, conf


# get n Keypoints from 2 images if possible
def getKpts(matching, image0_path, image1_path, n=-1):
    torch.set_grad_enabled(False)
    keys = ['keypoints', 'scores', 'descriptors']

    # make image0 first, image 1 second
    if image0_path[-5] == '2':
        image0_path, image1_path = image1_path, image0_path

    print(image0_path)
    frame0 = cv2.imread(image0_path, 0)
    print(type(frame0), type(mask0))
    frame0 = (torch.Tensor(frame0)).to(device)
    frame0 = torch.mul(frame0, mask0)  # apply mask
    frame_tensor0 = (frame0 / 255.).float()[None, None]
    last_data = matching.superpoint({'image': frame_tensor0})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor0

    frame1 = cv2.imread(image1_path, 0)
    frame1 = (torch.Tensor(frame1)).to(device)
    frame1 = torch.mul(frame1, mask1)  # apply mask
    frame1 = torch.rot90(frame1)  # rotate image to make images 0,1 similar
    frame_tensor1 = (frame1 / 255.).float()[None, None]

    pred = matching({**last_data, 'image1': frame_tensor1})
    kpts0 = last_data['keypoints0'][0]
    kpts1 = pred['keypoints1'][0]
    matches = pred['matches0'][0]
    confidence = pred['matching_scores0'][0]

    # valid_data
    valid = matches > -1

    valid_kpts0 = kpts0[valid].to(device)
    valid_kpts1 = kpts1[matches[valid]].to(device)
    valid_conf = confidence[valid].to(device)

    # print('ind shape before deletion: ', valid_conf.shape)
    # img_center = torch.Tensor([pic_half_size,pic_half_size])
    if len(valid_conf.size()) > 0:
        # delete out-of-circle points
        center = torch.Tensor([pic_half_size, pic_half_size]).to(device)
        len0 = torch.norm(valid_kpts0 - center, dim=1)
        len1 = torch.norm(valid_kpts1 - center, dim=1)
        indices0 = (len0 < pic_half_size).nonzero()
        indices1 = (len1 < pic_half_size).nonzero()
        indices = torch.unique(torch.cat((indices0, indices1)))
        valid_kpts0 = torch.squeeze(valid_kpts0[indices])
        valid_kpts1 = torch.squeeze(valid_kpts1[indices])
        valid_conf = torch.squeeze(valid_conf[indices])

    if len(valid_conf.size()) > 0:
        frame0_brightness = torch.zeros(valid_conf.shape[0])
        frame1_brightness = torch.zeros(valid_conf.shape[0])
        for k in range(valid_conf.shape[0]):
            frame0_brightness[k] = frame0[int(valid_kpts0[k, 0]), int(valid_kpts0[k, 1])] / 255
            frame1_brightness[k] = frame1[int(valid_kpts1[k, 0]), int(valid_kpts1[k, 1])] / 255

        if n != -1:
            if valid_conf.shape[0] < n:
                valid_kpts0 = torch.zeros((n, 2))
                valid_kpts1 = torch.zeros((n, 2))
                valid_conf = torch.zeros(n)
                frame0_brightness = torch.zeros(n)
                frame1_brightness = torch.zeros(n)
                not_skip = 0
                # print('here0')
            else:
                valid_conf, arr1inds = torch.sort(valid_conf, descending=True)
                valid_kpts0 = valid_kpts0[arr1inds, :]
                valid_kpts1 = valid_kpts1[arr1inds, :]
                frame0_brightness = frame0_brightness[arr1inds]
                frame1_brightness = frame1_brightness[arr1inds]
                valid_kpts0 = valid_kpts0[0:n]
                valid_kpts1 = valid_kpts1[0:n]
                valid_conf = valid_conf[0:n]
                frame0_brightness = frame0_brightness[0:n]
                frame1_brightness = frame1_brightness[0:n]
                not_skip = 1
                # print('here1')
        # if n==-1, keep all points
        else:
            not_skip = 1
            # print('here2')
    else:
        valid_kpts0 = torch.zeros((1, 2))
        valid_kpts1 = torch.zeros((1, 2))
        valid_conf = torch.zeros(2)
        frame0_brightness = torch.zeros(2)
        frame1_brightness = torch.zeros(2)
        not_skip = 0
        # print('here3')

    # save image
    if not_skip:
        valid_kpts0_pic = valid_kpts0.cpu().numpy()
        valid_kpts1_pic = valid_kpts1.cpu().numpy()
        valid_conf_pic = valid_conf.cpu().numpy()
        frame0_pic = frame0.cpu().numpy()
        frame1_pic = frame1.cpu().numpy()
        kpts0_pic = kpts0.cpu().numpy()
        kpts1_pic = kpts1.cpu().numpy()

        color = cm.jet(valid_conf_pic)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0_pic), len(kpts1_pic)),
            'Matches: {}'.format(len(valid_kpts0_pic))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
        ]
        out = make_matching_plot_fast(
            frame0_pic, frame1_pic, kpts0_pic, kpts1_pic, valid_kpts0_pic, valid_kpts1_pic, color, text,
            path=None, show_keypoints=True, small_text=small_text)
    else:
        out = np.zeros((1920, 3850, 3))
    return valid_kpts0, valid_kpts1, valid_conf, frame0_brightness, frame1_brightness, not_skip, out


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))


def calc_theta_fi_torch(pic_half_size, point):
    center = torch.Tensor([pic_half_size, pic_half_size])
    point_centered = point - center
    point_norm = torch.norm(point_centered)
    point_resized_to_pic = point_centered / pic_half_size
    abs_resized_to_pic = point_norm / pic_half_size
    if abs_resized_to_pic > 1:
        # print('point is out of circle')
        # exit(1)
        abs_resized_to_pic = torch.Tensor([1]).to(device)  # заглушка, разобраться
    theta = torch.asin(abs_resized_to_pic)
    point_resized_norm = torch.norm(point_resized_to_pic)
    if point_resized_norm != 0:
        fi = torch.atan2(point_resized_to_pic[1], point_resized_to_pic[0])
    else:
        # print('fi=0')
        fi = 0
    return fi, theta


# def calc_theta_fi(center, point):
#     point_centered = point - center
#     point_norm = np.linalg.norm(point_centered)
#     pic_half_size = center[0]
#     point_resized_to_pic = point_centered / pic_half_size
#     abs_resized_to_pic = point_norm / pic_half_size
#     # if abs_resized_to_pic > 1:
#     #     print('point is out of circle')
#     theta = np.arcsin(abs_resized_to_pic)
#     fi = np.arctan2(point_resized_to_pic[1], point_resized_to_pic[0])
#     return fi, theta


def show_image_with_point(frame, point):
    image1 = cv2.circle(frame, (int(960), int(960)), radius=20, color=0, thickness=-1)
    image1 = cv2.circle(image1, (int(point[0]), int(point[1])), radius=20, color=0, thickness=-1)
    image1 = cv2.resize(image1, (960, 540))
    cv2.imshow('w1', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mat_block(theta1, fi1, theta2, fi2):
    block = np.zeros((4, 2))
    block[0, 0] = np.sin(theta1) * np.cos(fi1)
    block[0, 1] = -np.sin(theta2) * np.cos(fi2)
    block[1, 0] = np.sin(theta1) * np.sin(fi1)
    block[1, 1] = -np.sin(theta2) * np.sin(fi2)
    block[2, 0] = np.cos(theta1)
    block[2, 1] = -np.cos(theta2)
    block[3, 0] = np.cos(theta1)
    return block


matching = Matching(config).eval().to(device)


def set_title(img, title):
    height, width, ch = img.shape
    new_width, new_height = int(width + width / 20), int(height + height / 8)

    # Crate a new canvas with new width and height.
    canvas = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125

    # New replace the center of canvas with original image
    padding_top, padding_left = 200, 10
    if padding_top + height < new_height and padding_left + width < new_width:
        canvas[padding_top:padding_top + height, padding_left:padding_left + width] = img
    # else:
    #     print ("The Given padding exceeds the limits.")

    color = (255, 0, 255)
    img = cv2.putText(canvas.copy(), title, (int(0.25 * width), 100), cv2.FONT_HERSHEY_COMPLEX, 3, color, 4)

    return img
