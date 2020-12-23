import numpy as np
from functions import mean_absolute_percentage_error
import torch

data = np.load('predictions.npz')
h_label = data['labels']
h_pred = data['pred']
indices_under_1500 = h_label < 1500
indices_under_1300 = h_label < 1300

h_pred = torch.Tensor(h_pred)
h_label = torch.Tensor(h_label)

h_pred_under_1500 = h_pred[indices_under_1500]
h_label_under_1500 = h_label[indices_under_1500]
h_pred_under_1300 = h_pred[indices_under_1300]
h_label_under_1300 = h_label[indices_under_1300]

print('all data: ', h_pred.shape[0])
print('under 1500 data: ', h_pred_under_1500.shape[0])
print('under 1300 data: ', h_pred_under_1300.shape[0])

print('mape (all data): ', mean_absolute_percentage_error(h_label, h_pred))
print('mape (under 1500): ', mean_absolute_percentage_error(h_label_under_1500, h_pred_under_1500))
print('mape (under 1300): ', mean_absolute_percentage_error(h_label_under_1300, h_pred_under_1300))

print('label under 1300: ', h_label_under_1300)
print('pred under 1300: ', h_pred_under_1300)
