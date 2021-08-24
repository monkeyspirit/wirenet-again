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


def reconstruction(x, y):
    """
    reconstruction restore original bytes from embedding matrix.
    Args:
        x torch.Tensor:
            x is word embedding
        y torch.Tensor:
            y is embedding matrix
    Returns:
        torch.Tensor:
    """
    x_size = x.size()[0]
    y_size = y.size()[0]
    # print(x_size, y_size)

    z = torch.zeros(x_size)

    for i in tqdm(range(x_size)):
        dist = torch.zeros(256)

        for j in range(y_size):
            dist[j] = torch.dist(x[i], y[j])  # computation of euclidean distance

        z[i] = dist.argmin()

    return z


def fgsm_attack():
    bytez = open('output/AddString_5_Tequila.exe', "rb").read()

    # Create malconv
    malconv = MalConv_ForADV(channels=256, window_size=512, embd_size=8)
    weights = torch.load('malconv/malconv.checkpoint', map_location='cpu')
    malconv.load_state_dict(weights['model_state_dict'])
    # malconv.eval()


    # Compute payload size
    payload_size = (kernel_size + (kernel_size - np.mod(len(bytez), kernel_size))) * 8 + 1
    print('payload: ',payload_size)

    # Creat embedding matrix
    embed = malconv.embd
    m = embed(torch.arange(0, 256)) # M

    # Make label from target
    label = torch.tensor([target], dtype=torch.long)

    perturbation = np.random.randint(0, 256, payload_size, dtype=np.uint8)

    # Make input file x as numpy array
    x = np.frombuffer(bytez, dtype=np.uint8)

    loss_values = []

    for i in range(loop_num):
        print('[{}]'.format(str(i + 1)))

        # Make input of malconv
        inp = torch.from_numpy(np.concatenate([x, perturbation])[np.newaxis, :]).float()
        inp_adv = inp.requires_grad_()
        embd_x = embed(inp_adv.long()).detach() # Z
        embd_x.requires_grad = True
        # embd_x.retain_grad()

        outputs = malconv(embd_x)
        results = F.softmax(outputs, dim=1)

        r = results.detach().numpy()[0]
        print('Benign: {:.5g}'.format(r[0]), ', Malicious: {:.5g}'.format(r[1]))

        # Compute loss
        loss = nn.CrossEntropyLoss()(results, label)
        # ----
        if i == 0:
            loss_values.append(loss.item())


        print('Loss: {:.5g}'.format(loss.item()))


        # Make a decision on evasion rates
        if results[0][0] > 0.5:
            print('Evasion rates: {:.5g}'.format(results[0][0].item()), '\n')
            aes_name = 'mod_5.exe'

            with open(aes_name, 'wb') as out:
                aes = np.concatenate([x, perturbation.astype(np.uint8)])

                for s in aes:
                    out.write(struct.pack('B', int(s)))

            print(aes_name, ' has been created.')

            return

        # Update
        loss.backward()

        grad = embd_x.grad
        grad_sign = grad.detach().sign()[0][-payload_size:]  # extract only grad_sign of perturbation

        # Change types to numpy to prevent Error: Leaf variable was used in an inplace operation
        perturbation = embed(torch.from_numpy(perturbation).long())

        # Compute perturbation
        perturbation = (perturbation - eps * grad_sign).detach().numpy()

        embd_x = embd_x.detach().numpy()
        embd_x[0][-payload_size:] = perturbation  # update perturbation

        # ----
        # if the loss value is the same we are in a local minimum, so we regenerate the
        # perturbation vector
        if i != 0 and loss_values[-1] == loss.item():
            print('Regeneration of the perturbation')
            perturbation = np.random.randint(0, 256, payload_size, dtype=np.uint8)
        else:
            if i != 0:
                loss_values.append(loss.item())

            print('Reconstruction phase:')
            perturbation = reconstruction(torch.from_numpy(perturbation), m).detach().numpy()
            print('sum of perturbation: ', perturbation.sum(), '\n')  # for debug

            # Generate perturbation file
            with open('perturb.bin', 'wb') as out:
                for s in perturbation:
                    out.write(struct.pack('B', int(s)))

    print('Adversarial Examples is not found.')


if __name__ == '__main__':
    fgsm_attack()
