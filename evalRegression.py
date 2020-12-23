import torch
from functions import device, shuffle, h_max, get_good_batch, mean_absolute_percentage_error
from functions import vngoDataset, img0_paths, img1_paths, h, set_title
import numpy as np
import ntpath
import cv2
from tqdm import tqdm

fcnn = torch.load('./regression_model.pth')
# print(fcnn)
fcnn = fcnn.to(device)

# for param in fcnn.parameters():
#     print('param: ', param)

n_examples = 64
n_points = 15
test_dataset = vngoDataset(img0_paths, img1_paths, h, n=n_points)
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=shuffle, batch_size=1)
h_averaged = torch.zeros(n_examples)
h_labels = torch.zeros(n_examples)
# print('h labels: ', h_labels)
fcnn.eval()
loss_fun = mean_absolute_percentage_error

for p in tqdm(range(n_examples)):
    h_label_norm, stat_data, out, name0, name1 = get_good_batch(test_dataloader)
    if h_label_norm.shape[0] == 0:
        continue
    else:
        h_label_norm = h_label_norm.to(device)
        h_label = h_label_norm * h_max
        stat_data = stat_data.double().to(device)

        n_p = stat_data.shape[1]  # number of detected pairs of points
        batch_size_out = stat_data.shape[0]
        h_pred_all_points = torch.zeros(n_p).to(device)

        for k in range(n_p):
            stat_data1 = stat_data[:, k, :]  # select k-th point
            with torch.set_grad_enabled(False):
                h_norm_pred = fcnn(stat_data1)
                h_norm_pred = torch.squeeze(h_norm_pred)
                h_label_norm = torch.squeeze(h_label_norm)
                h_label = torch.squeeze(h_label)
                loss_val = loss_fun(y_true=h_label_norm, y_pred=h_norm_pred)
                h_pred = h_norm_pred * h_max
                # print('h pred: ', h_pred)
                # print('loss_val (mape): ', loss_val)
                # print('L1 (meters): ', torch.nn.L1Loss()(h_pred, h_label))
                h_pred_all_points[k] = h_pred

        h_labels[p] = h_label.item()
        h_mean = (torch.mean(h_pred_all_points))
        h_averaged[p] = h_mean

        out_img = out[0, :]
        out_path = str('./res/' + ntpath.basename(name0[0]))
        out_img = out_img.cpu().numpy()
        title = 'h_pred = ' + str(round(h_mean.item(), 1)) + ' h_label = ' + str(round(h_label.item(), 1))
        out_img = set_title(out_img, title)
        cv2.imwrite(out_path, out_img)

h_labels = h_labels[h_labels.nonzero()]  # delete zero elements
h_averaged = h_averaged[h_averaged.nonzero()]

print('h averaged: ', h_averaged)
print('h label: ', h_labels)
print('loss_val (mape final): ', loss_fun(y_true=h_labels, y_pred=h_averaged))

h_labels = h_labels.cpu().numpy()
h_averaged = h_averaged.cpu().numpy()
np.savez('predictions', labels=h_labels, pred=h_averaged)
