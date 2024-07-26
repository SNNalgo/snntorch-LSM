import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn import linear_model
import time

from lsm_weight_definitions import initWeights_partitionV2
from lsm_models import LSM, LSM_partition

def simple_ensemble_lsm(in_conn, num_res=2, num_partitions=1, Nz=12):

    #Load dataset (Using NMNIST here)
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=3000),
                                          transforms.ToFrame(sensor_size=sensor_size,time_window=1000)])

    trainset = tonic.datasets.NMNIST(save_to='../data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='../data', transform=frame_transform, train=False)

    cached_trainset = DiskCachedDataset(trainset, cache_path='../cache/nmnist/train')
    cached_testset = DiskCachedDataset(testset, cache_path='../cache/nmnist/test')

    batch_size = 256
    trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

    #Set device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    data, targets = next(iter(trainloader))
    flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1))
    print(flat_data.shape)

    in_sz = flat_data.shape[-1]

    #Set neuron parameters
    tauV = 16.0
    tauI = 16.0
    th = 20
    curr_prefac = np.float32(1/tauI)
    alpha = np.float32(np.exp(-1/tauI))
    beta = np.float32(1 - 1/tauV)
    
    Nx=10
    Ny=10
    #Nz=12
    
    inh_fr = 0.5
    #in_conn = 0.15
    
    #num_partitions = 1
    w_lsm = 2
    w_in = (25*0.15)/in_conn
    '''
    window = 5
    inCh = data.shape[-3]
    inH  = data.shape[-2]
    inW  = data.shape[-1]
    print('possible connections per input neuron in receptive field: ', window*window*Nz)
    print('required connections per input neuron in receptive field: ', in_conn*Nx*Ny*Nz)
    print('#INFO: ensure that required connections is at most half of the possible connections (Note: Nz is scaled equally by partitions for both)')
    '''
    Wins_ens = []
    Wlsms = []
    for i in range(num_res):
        Wins, Wlsm = initWeights_partitionV2(w_in, w_lsm, in_conn, in_sz, num_partitions, inh_fr=inh_fr, Nx=Nx, Ny=Ny, Nz=Nz)
        for j in range(num_partitions):
            Wins[j] = np.float32(curr_prefac*Wins[j])
        Wins_ens.append(Wins)
        Wlsms.append(np.float32(curr_prefac*Wlsm))
    N = Wlsms[0].shape[0]
    lsm_nets = [LSM_partition(N, in_sz, Wins_ens[i], Wlsms[i], num_partitions, alpha=alpha, beta=beta, th=th).to(device) for i in range(num_res)]
    #Run with no_grad for LSM
    with torch.no_grad():
        start_time = time.time()
        for i, (data, targets) in enumerate(iter(trainloader)):
            if i%25 == 24:
                print("train batches completed: ", i)
            flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(device)
            spk_ens = [lsm_net(flat_data, device) for lsm_net in lsm_nets]
            spk_rec = torch.cat(spk_ens, 2)
            #print("ensemble output shape: ", spk_rec.shape)
            lsm_out = torch.mean(spk_rec, dim=0)
            if i==0:
                in_train = torch.mean(flat_data, dim=0).cpu().numpy()
                lsm_out_train = lsm_out.cpu().numpy()
                lsm_label_train = np.int32(targets.numpy())
            else:
                in_train = np.concatenate((in_train, torch.mean(flat_data, dim=0).cpu().numpy()), axis=0)
                lsm_out_train = np.concatenate((lsm_out_train, lsm_out.cpu().numpy()), axis=0)
                lsm_label_train = np.concatenate((lsm_label_train, np.int32(targets.numpy())), axis=0)
        end_time = time.time()

        print("running time of training epoch: ", end_time - start_time, "seconds")

        for i, (data, targets) in enumerate(iter(testloader)):
            if i%25 == 24:
                print("test batches completed: ", i)
            flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(device)
            spk_ens = [lsm_net(flat_data, device) for lsm_net in lsm_nets]
            spk_rec = torch.cat(spk_ens, 2)
            lsm_out = torch.mean(spk_rec, dim=0)
            if i==0:
                in_test = torch.mean(flat_data, dim=0).cpu().numpy()
                lsm_out_test = lsm_out.cpu().numpy()
                lsm_label_test = np.int32(targets.numpy())
            else:
                in_test = np.concatenate((in_test, torch.mean(flat_data, dim=0).cpu().numpy()), axis=0)
                lsm_out_test = np.concatenate((lsm_out_test, lsm_out.cpu().numpy()), axis=0)
                lsm_label_test = np.concatenate((lsm_label_test, np.int32(targets.numpy())), axis=0)

    print(lsm_out_train.shape)
    print(lsm_out_test.shape)

    print(in_train.shape)
    print(in_test.shape)

    print("mean in spiking (train) : ", np.mean(in_train))
    print("mean in spiking (test) : ", np.mean(in_test))

    print("mean LSM spiking (train) : ", np.mean(lsm_out_train))
    print("mean LSM spiking (test) : ", np.mean(lsm_out_test))
    print('number of reservoirs : ', num_res)
    print('num partitions : ', num_partitions)

    print("training linear model:")
    clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
    clf.fit(lsm_out_train, lsm_label_train)

    score = clf.score(lsm_out_test, lsm_label_test)
    print("test score = " + str(score))
    return score
