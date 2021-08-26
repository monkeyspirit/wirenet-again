import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import sys
import struct
import os
from tqdm import tqdm

from MalConv_ForADV import MalConv_ForADV

kernel_size = 512
eps = 0.8
target = 0  # benign
loop_num = 10


def fgsm_attack():
    bytez = open('taquila-bad-classified.exe', "rb").read()

    # Create malconv
    malconv = MalConv_ForADV(channels=256, window_size=512, embd_size=8)
    weights = torch.load('malconv/malconv.checkpoint', map_location='cpu')
    malconv.load_state_dict(weights['model_state_dict'])
    # malconv.eval()


    # Compute payload size
    payload_size = (kernel_size + (kernel_size - np.mod(len(bytez), kernel_size))) * 8 + 1
    print('payload: ', payload_size)

    # Creat embedding matrix
    embed = malconv.embd
    m = embed(torch.arange(0, 256)) # M

    # Make label from target
    label = torch.tensor([target], dtype=torch.long)

    perturbation = np.random.randint(0, 256, payload_size, dtype=np.uint8)

    # Make input file x as numpy array
    x = np.frombuffer(bytez, dtype=np.uint8)
    inp = torch.from_numpy(np.copy(x))[np.newaxis, :].float()
    inp_adv = inp.requires_grad_()
    embd_x = embed(inp_adv.long()).detach()
    embd_x.requires_grad = True
    outputs = malconv(embd_x)
    results = F.softmax(outputs, dim=1)
    print('Prediction benign: {:.4}'.format(results.detach().numpy()[0][0]))


if __name__ == '__main__':
    fgsm_attack()
