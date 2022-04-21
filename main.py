# -*- coding: utf-8 -*-
'''
@Time    : 2022/4/17 12:21
@Author  : LYZ
@FileName: main.py
@Software: PyCharm
'''
import os
import pytorch_ssim
from BCDNet import UNet
from tqdm import tqdm
from torch import nn
import torch
from Eval import pre

noisy_data_path = "dataset/noisy/"
origin_data_path = "dataset/ground_truth/"
NoisyFiles = os.listdir(noisy_data_path)
OriginFiles = os.listdir(origin_data_path)
NoisyFiles_len = len(NoisyFiles)
device = "cuda:0"
lr = 0.000001

epochs = 1000
model_path = "BCD-Model.pth"
white_level = 16383
black_level = 1024

if __name__ == "__main__":
    net = UNet().to(device)
    net.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    net.train()

    for epoch in range(epochs):

        net.to(device)

        running_loss = 0.0
        for i in tqdm(range(NoisyFiles_len)):
            X, X_height, X_width = pre(input_path=noisy_data_path + str(i) + "_noise.dng")
            Y, Y_height, Y_width = pre(input_path=origin_data_path + str(i) + "_gt.dng")

            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()

            Y_HAT = net(X)

            l = 1 - pytorch_ssim.ssim(Y_HAT, Y)

            l.backward()
            optimizer.step()

            running_loss += l.item()
        print("Epoch{}\tloss {}".format(epoch, running_loss / NoisyFiles_len))

        print()
        torch.save(net.state_dict(), 'models/CBD-L1-' + str(epoch) + '.pth')
