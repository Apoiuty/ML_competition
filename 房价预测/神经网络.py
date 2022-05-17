import math
import time

import torch.nn as nn
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import ray
from ray import tune


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class LinerModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, 1)
        )

    def forward(self, data):
        return self.net(data)


class HousePriceData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X)


np.random.seed(42)
all_features, y = torch.load('../input/housing-price/house_price.pkl')
X = all_features.iloc[:len(y), :]
X, y = X.values, y.values
y = y[:, np.newaxis]
std = StandardScaler()
X = std.fit_transform(X)
final_test = all_features.iloc[len(y):, :]
X = torch.tensor(X, device='cuda:0', dtype=torch.float32)
y = torch.tensor(y, device='cuda:0', dtype=torch.float32)

k_fold = KFold(n_splits=5, shuffle=True)

cv_loss = 0
epoch = 500
for train_index, test_index in k_fold.split(X):
    print('cv')
    time.sleep(2)
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    model = LinerModel(X.shape[1]).cuda()
    loss = RMSLELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.1)
    train_data = HousePriceData(X_train, y_train)
    dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    best_epoch = 0
    best_test_loss = math.inf
    for i in range(epoch):
        delta = 0
        for x, y_ in dataloader:
            y_pred = model(x)
            optim.zero_grad()
            l = loss(y_pred, y_)
            delta += l.item()
            l.backward()
            optim.step()
        lr_scheduler.step()
        if i % 10 == 0:
            print(f'epoch{i} train_loss: {delta:.6f}')
        test_loss = loss(model(X_test), y_test).item()
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = i
            print(f'step{i} test_loss {best_test_loss:.6f}')

    cv_loss += loss(model(X_test), y_test).item()
print(f'cv10 loss:{cv_loss / 5}')
