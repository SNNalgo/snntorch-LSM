import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn import linear_model
import time

from lsm_weight_definitions import initWeights1
from lsm_models import LSM

if __name__ == "__main__":

    #Load dataset (Using NMNIST here)
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=3000),
                                          transforms.ToFrame(sensor_size=sensor_size,time_window=1000)])

    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

    cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
    cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

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

    Win, Wlsm = initWeights1(27, 2, 0.15, in_sz)
    N = Wlsm.shape[0]
    lsm_net = LSM(N, in_sz, np.float32(curr_prefac*Win), np.float32(curr_prefac*Wlsm), alpha=alpha, beta=beta, th=th).to(device)
    lsm_net.eval()
    #Run with no_grad for LSM
    with torch.no_grad():
        start_time = time.time()
        for i, (data, targets) in enumerate(iter(trainloader)):
            if i%25 == 24:
                print("train batches completed: ", i)
            flat_data = torch.reshape(data, (data.shape[0], data.shape[1], -1)).to(device)
            spk_rec = lsm_net(flat_data)
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
            lsm_net.eval()
            spk_rec = lsm_net(flat_data)
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

    print("training linear model:")
    clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-6)
    clf.fit(lsm_out_train, lsm_label_train)

    score = clf.score(lsm_out_test, lsm_label_test)
    print("test score = " + str(score))
