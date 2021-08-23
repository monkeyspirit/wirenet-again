import torch
import torch.nn as nn
import torch.nn.functional as F


class MalConv_ForADV(nn.Module):
    # trained to minimize cross-entropy loss
    # criterion = nn.CrossEntropyLoss()
    def __init__(self, out_size=2, channels=128, window_size=512, embd_size=8):
        super(MalConv_ForADV, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        self.embd.requires_grad_(True)
        self.window_size = window_size

        self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)

        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)

    def forward(self, x):
        # print('Input:', x.shape)
        x = torch.transpose(x, -1, -2)
        # print('Post transpose:', x.shape)
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))
        # print('Post sigm cnn:', cnn_value.shape)
        # print('Post sigm gating_weight:', gating_weight.shape)
        x = cnn_value * gating_weight
        # print('Post xor:', x.shape)
        x = self.pooling(x)
        # print('Post pooling:', x.shape)

        # Flatten
        x = x.view(x.size(0), -1)
        # print('Post flatten:', x.shape)
        x = F.relu(self.fc_1(x))
        # print('Post ReLu fc1:', x.shape)
        x = self.fc_2(x)
        # print('Post fc2:', x.shape)

        return x
