import torch
import torch.nn.functional as F
import numpy as np
from MalConv import MalConv



class MalConvModel(object):
    def __init__(self, model_path, thresh=0.5, name='malconv'):
        self.model = MalConv(channels=256, window_size=512, embd_size=8).train()
        self.model.eval()
        weights = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(weights['model_state_dict'])
        self.thresh = thresh
        self.__name__ = name

    def predict(self, bytez):

        _inp = torch.from_numpy(np.frombuffer(bytez, dtype=np.uint8)[np.newaxis, :])
        outputs = F.softmax(self.model(_inp), dim=-1)

        return outputs.detach().numpy()[0, 1]


