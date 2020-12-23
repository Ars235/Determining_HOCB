import ntpath
import matplotlib.pyplot as plt
from functions import train_dataset, h_max, shuffle, pixels_norm_factor
import numpy as np
import torch

path_to_statistics_folder = './statistics/'
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, batch_size=1)

for h_norm, stat_data, skip_train, dist_valid, name0, conf_valid in train_dataloader:
    # print('stat_data size: ', stat_data.shape)
    stat_data = stat_data * pixels_norm_factor
    dist_valid = dist_valid * pixels_norm_factor
    mean_dist = stat_data[:, -2]
    std_dist = stat_data[:, -1]
    # print('mean: ', mean_dist)
    # print('std: ', std_dist)

    h = h_norm * h_max  # unnormalize
    name0 = ntpath.basename(name0[0])
    name0 = name0[:-4]

    plt.clf()
    plt.hist(x=dist_valid, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('L2-distance bettwen corresponding Keypoints')
    plt.ylabel('Frequency')
    title_str = 'H=' + str(round(h.item(), 1)) + ' mean=' + str(round(mean_dist.item(), 1)) + ' std=' + str(round(std_dist.item(), 1))
    plt.title(title_str)
    plt.savefig(path_to_statistics_folder + name0 + "_dist_valid.png")

    plt.clf()
    plt.hist(x=conf_valid, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title(title_str)
    plt.savefig(path_to_statistics_folder + name0 + "_conf_valid.png")

    # dist_valid = (torch.squeeze(dist_valid)).numpy()
    # statistics_step = 10  # calculate percentiles
    # perc_array = calc_percentiles(dist_valid, statistics_step)
    # print('percentiles: ', stat_data[:, 1:-2])

    # exit(1)
