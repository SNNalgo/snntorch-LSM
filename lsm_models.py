import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

import snntorch as snn

class LSM(nn.Module):
    def __init__(self, N, in_sz, Win, Wlsm, alpha=0.9, beta=0.9, th=20):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, N)
        self.fc1.weight = nn.Parameter(torch.from_numpy(Win))
        self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th)
        self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(Wlsm))
    def forward(self, x):
        num_steps = x.size(0)
        spk, syn, mem = self.lsm.init_rsynaptic()
        spk_rec = []
        for step in range(num_steps):
            curr = self.fc1(x[step])
            spk, syn, mem = self.lsm(curr, spk, syn, mem)
            spk_rec.append(spk)
        spk_rec_out = torch.stack(spk_rec)
        return spk_rec_out