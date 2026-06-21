import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader
from numpy.linalg import matrix_rank

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import time

from lsm_weight_definitions import initWeights1
from lsm_models import LSM

if __name__ == "__main__":

    #Load dataset (Using NMNIST here)
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=3000),
                                          transforms.ToFrame(sensor_size=sensor_size,time_window=1000)])

    #trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

    #cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
    cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

    batch_size = 256
    #trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

    #Set device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    data, targets = next(iter(testloader))
    flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(device)
    print('data shape: ', data.shape)
    print('flat data shape: ', flat_data.shape)

    in_sz = flat_data.shape[-1]

    #Set neuron parameters
    tauV = 16.0
    tauI = 16.0
    th = 20
    curr_prefac = np.float32(1/tauI)
    alpha = np.float32(np.exp(-1/tauI))
    beta = np.float32(1 - 1/tauV)
    Nz = 10
    NMNIST_stim = 14
    AIS = np.logspace(np.log10(1.5), np.log10(1500), 30)
    in_conn = 0.3
    w_ins = AIS/(NMNIST_stim*in_conn)
    lyapunov_exp = []
    out_rank = []
    perturb_frac = 1.0
    
    lambda_range = np.arange(-20, 20, 0.1)
    ratio_vals = (np.exp(lambda_range) - 1)/lambda_range
    n_vals = ratio_vals.shape[0]
    
    #Run with no_grad for LSM
    with torch.no_grad():
        start_time = time.time()
        for i in range(w_ins.shape[0]):
            print("completed fraction: ", i/w_ins.shape[0])
            
            w_in = w_ins[i]
            Win, Wlsm = initWeights1(w_in, 2, in_conn, in_sz, Nz=Nz)
            N = Wlsm.shape[0]
            lsm_net = LSM(N, in_sz, np.float32(curr_prefac*Win), np.float32(curr_prefac*Wlsm), alpha=alpha, beta=beta, th=th).to(device)
            num_partitions = 3
            lsm_net.eval()
            
            spk_rec = lsm_net(flat_data)
            spk_rate = torch.mean(spk_rec, dim=0)
            spk_rate_np = spk_rate.cpu().numpy()
            #_, S, _ = np.linalg.svd(spk_rate_np)
            #out_rank.append(matrix_rank(spk_rate_np, tol=1e-3*np.max(S))) # with rtol
            
            perturbation = torch.normal(mean=perturb_frac*NMNIST_stim/in_sz, std=perturb_frac*NMNIST_stim/in_sz, size=flat_data.shape).to(device)
            perturbation_rate = torch.mean(perturbation, dim=0)
            spk_perturbed = lsm_net(flat_data + perturbation)
            perturbed_rate = torch.mean(spk_perturbed, dim=0)
            
            del_in = (torch.sum(perturbation_rate**2, dim=-1)**0.5).cpu().numpy()
            del_out = (torch.sum((perturbed_rate-spk_rate)**2, dim=-1)**0.5).cpu().numpy()
            ratios = np.expand_dims(del_out/del_in, axis=1)
            ratios_tile = np.tile(ratios, (1, n_vals))
            print('ratios_tile.shape: ', ratios_tile.shape)
            print('ratio_vals.shape: ', ratio_vals.shape)
            ratio_diffs = np.abs(ratios_tile - ratio_vals)
            print('ratio_diffs.shape: ', ratio_diffs.shape)
            lambda_vals = lambda_range[np.argmin(ratio_diffs, axis=1)]
            print('lambda_vals.shape: ', lambda_vals.shape)
            print('AIS: ', AIS[i])
            print('Lyapunov exponent: ', np.mean(lambda_vals))
            
            #lyapunov_exp.append(np.log10(np.mean(del_out/del_in)))
            lyapunov_exp.append(np.mean(lambda_vals))
    
    #print('Output Spike-Rate Matrix Rank: ', out_rank)
    print('Lyapunov Exponents: ', lyapunov_exp)
    print('AIS: ', AIS.tolist())
    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Average Input Stimulus')
    ax1.set_ylabel('lyapunov exponent', color=color)
    ax1.semilogx(AIS, lyapunov_exp, color='k', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    
    #ax2 = ax1.twinx()
    #color = 'tab:blue'
    #ax2.set_ylabel('Output Spike-Rate Matrix Rank', color=color) 
    #ax2.semilogx(AIS, out_rank, color='b', linestyle='-')
    #ax2.tick_params(axis='y', labelcolor=color)
    #fig.tight_layout()
    plt.show()
    
    
    