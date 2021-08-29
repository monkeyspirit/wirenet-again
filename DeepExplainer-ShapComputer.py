from os import walk

import numpy as np

import shap

import torch
import torch.nn
import torch.nn.functional as F


prefix = '.'


class MalConv_ForADV(torch.nn.Module):
  # trained to minimize cross-entropy loss
  # criterion = nn.CrossEntropyLoss()
  def __init__(self, out_size=2, channels=128, window_size=512, embd_size=8):
    super(MalConv_ForADV, self).__init__()
    self.embd = torch.nn.Embedding(257, embd_size, padding_idx=0)
    self.embd.requires_grad_(True)
    self.window_size = window_size

    self.conv_1 = torch.nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
    self.conv_2 = torch.nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)

    self.pooling = torch.nn.AdaptiveMaxPool1d(1)

    self.fc_1 = torch.nn.Linear(channels, channels)
    self.fc_2 = torch.nn.Linear(channels, out_size)

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
    x = F.softmax(x, dim=1)
    return x



model = MalConv_ForADV(channels=256, window_size=512, embd_size=8)
weights = torch.load(f'{prefix}/malconv/malconv.checkpoint', map_location='cpu')
model.load_state_dict(weights['model_state_dict'])



malwares = []
for dirpath, dirnames, filenames in walk('windows-zoo'):
    malwares.extend(filenames)
    break
malwares = ['windows-zoo/'+malware for malware in malwares]


corretto, sbagliato = 0, 0
malware_bins = None
padding_char = 0


lista_dims = []
for i, mal_path in enumerate(malwares):
  with open(mal_path, 'rb') as handle:
    sample = handle.read()
  sample = np.frombuffer(sample, dtype=np.uint8)
  sample = torch.from_numpy(np.copy(sample))[np.newaxis, :]
  sample_embed = model.embd(sample.long()).detach()
  lista_dims.append(sample_embed.shape[1])

max_dim = min(max(lista_dims), 2**10)


with open('tequila-bad-classified.exe', 'rb') as handle:
  b_tequila_buffer = handle.read()
bytez = np.frombuffer(b_tequila_buffer, dtype=np.uint8)
if len(bytez) <= max_dim:
  sample = np.ones(max_dim, dtype=np.uint8)*padding_char
  sample[:len(bytez)] = bytez
else:
  sample = bytez[-max_dim:]
b_tequila = torch.from_numpy(np.copy(sample))[np.newaxis, :]
b_tequila_embed = model.embd(b_tequila.long()).detach()



for mal_path in malwares:
  with open(mal_path, 'rb') as handle:
    sample = handle.read()

  bytez = np.frombuffer(sample, dtype=np.uint8)
  sample = None
  if len(bytez) <= max_dim:
     sample= np.ones(max_dim, dtype=np.uint8)*padding_char
     sample[:len(bytez)] = bytez
  else:
     sample = bytez[-max_dim:]
  sample = torch.from_numpy(np.copy(sample))[np.newaxis, :]
  sample_embed = model.embd(sample.long()).detach()

  # res = F.softmax(model(sample_embed), dim=1).detach()
  res = model(sample_embed).detach()
  if res[0][1] > 0.5:
    corretto += 1
    if malware_bins != None:
      malware_bins = torch.cat((malware_bins, sample_embed), 0)
    else:
      malware_bins = torch.Tensor(sample_embed)
  else:
    sbagliato += 1

print(f'Corretti: {corretto}\tSbaliati: {sbagliato}')


explainer = shap.DeepExplainer(model=model, data=malware_bins)

shap_values = explainer.shap_values(b_tequila_embed)
print(shap_values)

import pickle
with open('end_deep_shap.pickle', 'wb') as handle:
  pickle.dump(shap_values, handle)

with open('end_deep_explainer.pickle', 'wb') as handle:
  pickle.dump(explainer, handle)
