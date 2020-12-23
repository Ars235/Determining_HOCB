import torch
from typing import Tuple, List, Type, Dict, Any
from functions import device, shuffle, h_max, get_good_batch, get_lr, mean_absolute_percentage_error
from functions import img1_paths, img0_paths, h, vngoDataset
import numpy as np

batch_size_train = 64
print('shuffle: ', shuffle)
train_dataset = vngoDataset(img0_paths, img1_paths, h, n=1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size_train)

# get_good_batch(train_dataloader)

# loader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, batch_size=1)
# _, data, _ = next(iter(loader))
input_size = 16
print('input size: ', input_size)

act = torch.nn.ReLU

fcnn = torch.nn.Sequential(
    torch.nn.Linear(input_size, 16),
    act(),
    torch.nn.Linear(16, 13),
    act(),
    torch.nn.Linear(13, 10),
    act(),
    torch.nn.Linear(10, 8),
    act(),
    torch.nn.Linear(8, 5),
    act(),
    torch.nn.Linear(5, 1)
)

print('Total number of trainable parameters',
      sum(p.numel() for p in fcnn.parameters() if p.requires_grad))

# fcnn = torch.nn.Sequential(
#     torch.nn.Linear(input_size, 1),
#     torch.nn.Sigmoid(),
# )

# fcnn = torch.load('./regression_model.pth')
# print('fcnn loaded')

fcnn = fcnn.to(device)
n_epochs = 20
initial_lr = 0.01
# loss_fun = mean_absolute_percentage_error
# loss_fun = torch.nn.L1Loss()
loss_fun = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fcnn.parameters(), lr=initial_lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
batches_in_epoch = 20

# start training
for epoch in range(1, n_epochs):
    epoch_loss = []
    fcnn.double().train()
    for k in range(batches_in_epoch):
        # h_label_norm, stat_data = get_good_batch(train_dataloader)
        h_label_norm, stat_data, _, _, _ = get_good_batch(train_dataloader)
        # print('stat data: ', stat_data)
        stat_data = stat_data.double().to(device)
        # print('got batch: ', h_label_norm)
        h_label_norm = h_label_norm.to(device)
        with torch.set_grad_enabled(True):
            fcnn.zero_grad()
            # print('stat data: ', stat_data.shape)
            h_norm_pred = fcnn(stat_data)
            h_norm_pred = torch.squeeze(h_norm_pred)
            h_label_norm = torch.squeeze(h_label_norm)
            # loss_val = loss_fun(y_true=h_label_norm, y_pred=h_norm_pred)
            loss_val = loss_fun(h_label_norm, h_norm_pred)
            loss_l2 = torch.nn.L1Loss()(h_label_norm * h_max, h_norm_pred * h_max)
            # print('loss_val (mape): ', loss_val)
            # print('L1 (meters): ', loss_l1)
            print('L2 (meters): ', torch.sqrt(loss_l2))
            print('mape: ', mean_absolute_percentage_error(y_pred=h_norm_pred * h_max, y_true=h_label_norm * h_max))
            epoch_loss.append(loss_l2)
            loss_val.backward()

            # print('label: ', torch.exp(h_label_norm*h_max))
            # print('pred: ', torch.exp(h_norm_pred * h_max))

            # for param in fcnn.parameters():
            #     print('param: ', param)
            # for param in fcnn.parameters():
            #     print('param grad: ', param.grad)

            optimizer.step()
            lr_scheduler.step(loss_val)
            print('lr: ', get_lr(optimizer))
            torch.save(fcnn, './regression_model.pth')

    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    print('epoch_loss: ', epoch_loss)
