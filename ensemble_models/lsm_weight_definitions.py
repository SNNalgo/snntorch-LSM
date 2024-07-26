import numpy as np
import cv2
import torch

def gabor_filter(theta, lambda_val, sigma=10.0, gamma=0.5):
    """Generate a Gabor filter."""
    phi = np.pi / 2  # Phase offset
    kernel_size = 5  # Reduced kernel size
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_val, gamma, phi, ktype=cv2.CV_64F)
    kernel = kernel / np.linalg.norm(kernel)  # Normalize t
    return kernel

def build_gabor_filter_bank(thetas, lambdas):
    """Build a Gabor filter bank."""
    filters = []
    for theta in thetas:
        for lambda_val in lambdas:
            g_filter = np.float32(gabor_filter(np.radians(theta), lambda_val))
            filters.append(torch.stack([torch.from_numpy(g_filter), torch.from_numpy(g_filter)], dim=0))
    return torch.stack(filters, dim=0)

def initWeights1(LqWin, LqWlsm, in_conn_density, in_size, lam=9, inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None):
    N = Nx*Ny*Nz
    W_in = np.zeros((in_size,N))
    in_conn_range = np.int32(N*in_conn_density)
    
    for i in range(in_size):
        input_perm_i = np.arange(N)
        np.random.shuffle(input_perm_i)
        pos_conn = input_perm_i[:in_conn_range]
        neg_conn = input_perm_i[-in_conn_range:]
        W_in[i,pos_conn] = LqWin
        W_in[i,neg_conn] = -LqWin 
  
    input_perm = np.arange(N)
    np.random.shuffle(input_perm) # first 0.2*N indices are inhibitory
    inh_range = np.int32(inh_fr*N) # indices 0 to inh_range-1 are inhibitory

    if init_Wlsm:
        W_lsm = np.zeros((N,N))
        for i in range(N):
            posti = input_perm[i] # input_perm[i] is the post-neuron index
            zi = posti//(Nx*Ny)
            yi = (posti-zi*Nx*Ny)//Nx
            xi = (posti-zi*Nx*Ny)%Nx
            for j in range(N):
                prej = input_perm[j] # input_perm[j] is the pre-neuron index
                zj = prej//(Nx*Ny)
                yj = (prej-zj*Nx*Ny)//Nx
                xj = (prej-zj*Nx*Ny)%Nx
                D = ((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                if i<inh_range and j<inh_range: # II connection, C = 0.3
                    P = 0.3*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = -LqWlsm
                if i<inh_range and j>=inh_range: # EI connection, C = 0.1
                    P = 0.1*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = LqWlsm
                if i>=inh_range and j<inh_range: # IE connection, C = 0.05
                    P = 0.05*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = -LqWlsm
                if i>=inh_range and j>=inh_range: # EE connection, C = 0.2
                    P = 0.2*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = LqWlsm

        for i in range(N):
            W_lsm[i,i] = 0

    return W_in.T,W_lsm.T #need to transpose matrices for compatibility with torch nn linear

def initWeights_embed_Gabor_in(LqWin, LqWlsm, num_partitions, Res_H, Res_W, Res_ch, lam=9, inh_fr=0.2, init_Wlsm=True, W_lsm=None):
    Res_H = np.int32(Res_H)
    Res_W = np.int32(Res_W)
    Res_ch = np.int32(Res_ch)
    partition_ch = Res_ch//num_partitions
    N = np.int32(Res_H*Res_W*Res_ch)
    partition_N = np.int32(Res_H*Res_W*partition_ch)
    
    W_in_identity = LqWin*np.identity(N)
    W_ins = []
    
    for part in range(num_partitions):
        W_in = np.zeros((N,N))
        W_in[:, part*partition_N:(part+1)*partition_N] = W_in_identity[:, part*partition_N:(part+1)*partition_N]
        W_ins.append(W_in.T)

    input_perm = np.arange(partition_N)
    np.random.shuffle(input_perm) # first 0.2*N indices are inhibitory
    inh_range = np.int32(inh_fr*partition_N) # indices 0 to inh_range-1 are inhibitory

    if init_Wlsm:
        W_lsm_part = np.zeros((partition_N,partition_N))
        W_lsm = np.zeros((N,N))
        for i in range(partition_N):
            posti = input_perm[i] # input_perm[i] is the post-neuron index
            chi = posti//(Res_H*Res_W) #ch refers to channel dimension
            yi = (posti-chi*Res_H*Res_W)//Res_W #y refers to ROWS (height)
            xi = (posti-chi*Res_H*Res_W)%Res_W #x refers to COLS (width)
            for j in range(partition_N):
                prej = input_perm[j] # input_perm[j] is the pre-neuron index
                chj = prej//(Res_H*Res_W)
                yj = (prej-chj*Res_H*Res_W)//Res_W
                xj = (prej-chj*Res_H*Res_W)%Res_W
                D = ((xi-xj)**2 + (yi-yj)**2 + (chi-chj)**2)
                if i<inh_range and j<inh_range: # II connection, C = 0.3
                    P = 0.3*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i<inh_range and j>=inh_range: # EI connection, C = 0.1
                    P = 0.1*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = LqWlsm
                if i>=inh_range and j<inh_range: # IE connection, C = 0.05
                    P = 0.05*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i>=inh_range and j>=inh_range: # EE connection, C = 0.2
                    P = 0.2*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = LqWlsm

        for i in range(partition_N):
            W_lsm_part[i,i] = 0
        
        for part in range(num_partitions):
            W_lsm[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = W_lsm_part

    return W_ins,W_lsm.T #need to transpose matrices for compatibility with torch nn linear

def initWeights_partition(LqWin, LqWlsm, in_conn_density, in_size, num_partitions, lam=9, inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None):
    partition_Nz = Nz//num_partitions
    N = Nx*Ny*Nz
    partition_N = Nx*Ny*partition_Nz
    
    in_conn_range = np.int32(partition_N*in_conn_density)
    W_ins = []
    for part in range(num_partitions):
        W_in = np.zeros((in_size,N))
        W_in_part = np.zeros((in_size,partition_N))
        for i in range(in_size):
            input_perm_i = np.arange(partition_N)
            np.random.shuffle(input_perm_i)
            pos_conn = input_perm_i[:in_conn_range]
            neg_conn = input_perm_i[-in_conn_range:]
            W_in_part[i,pos_conn] = LqWin
            W_in_part[i,neg_conn] = -LqWin
        W_in[:, part*partition_N:(part+1)*partition_N] = W_in_part
        W_ins.append(W_in.T)
    
    W_in_part_cp = np.zeros((in_size,partition_N))
    W_in_part_cp[W_in_part>0] = W_in_part[W_in_part>0]
    in_to_res_fanout = np.sum(W_in_part_cp, axis=1)/LqWin
    res_fanin_from_in = np.sum(W_in_part_cp, axis=0)/LqWin
    print('in-res fanout shape: ', in_to_res_fanout.shape, ' average fanout: ', np.mean(in_to_res_fanout))
    print('res fanin from in shape: ', res_fanin_from_in.shape, ' average fanout: ', np.mean(res_fanin_from_in))
  
    input_perm = np.arange(partition_N)
    np.random.shuffle(input_perm) # first 0.2*N indices are inhibitory
    inh_range = np.int32(inh_fr*partition_N) # indices 0 to inh_range-1 are inhibitory

    if init_Wlsm:
        W_lsm_part = np.zeros((partition_N,partition_N))
        W_lsm = np.zeros((N,N))
        for i in range(partition_N):
            posti = input_perm[i] # input_perm[i] is the post-neuron index
            zi = posti//(Nx*Ny)
            yi = (posti-zi*Nx*Ny)//Nx
            xi = (posti-zi*Nx*Ny)%Nx
            for j in range(partition_N):
                prej = input_perm[j] # input_perm[j] is the pre-neuron index
                zj = prej//(Nx*Ny)
                yj = (prej-zj*Nx*Ny)//Nx
                xj = (prej-zj*Nx*Ny)%Nx
                D = ((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                if i<inh_range and j<inh_range: # II connection, C = 0.3
                    P = 0.3*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i<inh_range and j>=inh_range: # EI connection, C = 0.1
                    P = 0.1*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = LqWlsm
                if i>=inh_range and j<inh_range: # IE connection, C = 0.05
                    P = 0.05*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i>=inh_range and j>=inh_range: # EE connection, C = 0.2
                    P = 0.2*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = LqWlsm

        for i in range(partition_N):
            W_lsm_part[i,i] = 0
        
        for part in range(num_partitions):
            W_lsm[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = W_lsm_part

    return W_ins,W_lsm.T #need to transpose matrices for compatibility with torch nn linear

def initWeights_partitionV2(LqWin, LqWlsm, in_conn_density, in_size, num_partitions, lam=9, inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None):
    #V2 - Input goes to particular regions inside a larger Reservoir rather than disconnected smaller reservoir fragments
    partition_Nz = Nz//num_partitions
    N = Nx*Ny*Nz
    partition_N = Nx*Ny*partition_Nz
    
    in_conn_range = np.int32(partition_N*in_conn_density)
    W_ins = []
    for part in range(num_partitions):
        W_in = np.zeros((in_size,N))
        W_in_part = np.zeros((in_size,partition_N))
        for i in range(in_size):
            input_perm_i = np.arange(partition_N)
            np.random.shuffle(input_perm_i)
            pos_conn = input_perm_i[:in_conn_range]
            neg_conn = input_perm_i[-in_conn_range:]
            W_in_part[i,pos_conn] = LqWin
            W_in_part[i,neg_conn] = -LqWin
        W_in[:, part*partition_N:(part+1)*partition_N] = W_in_part
        W_ins.append(W_in.T)
    
    W_in_part_cp = np.zeros((in_size,partition_N))
    W_in_part_cp[W_in_part>0] = W_in_part[W_in_part>0]
    in_to_res_fanout = np.sum(W_in_part_cp, axis=1)/LqWin
    res_fanin_from_in = np.sum(W_in_part_cp, axis=0)/LqWin
    print('in-res fanout shape: ', in_to_res_fanout.shape, ' average fanout: ', np.mean(in_to_res_fanout))
    print('res fanin from in shape: ', res_fanin_from_in.shape, ' average fanout: ', np.mean(res_fanin_from_in))
  
    input_perm = np.arange(N)
    np.random.shuffle(input_perm) # first 0.2*N indices are inhibitory
    inh_range = np.int32(inh_fr*N) # indices 0 to inh_range-1 are inhibitory

    if init_Wlsm:
        W_lsm = np.zeros((N,N))
        for i in range(N):
            posti = input_perm[i] # input_perm[i] is the post-neuron index
            zi = posti//(Nx*Ny)
            yi = (posti-zi*Nx*Ny)//Nx
            xi = (posti-zi*Nx*Ny)%Nx
            for j in range(N):
                prej = input_perm[j] # input_perm[j] is the pre-neuron index
                zj = prej//(Nx*Ny)
                yj = (prej-zj*Nx*Ny)//Nx
                xj = (prej-zj*Nx*Ny)%Nx
                D = ((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                if i<inh_range and j<inh_range: # II connection, C = 0.3
                    P = 0.3*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = -LqWlsm
                if i<inh_range and j>=inh_range: # EI connection, C = 0.1
                    P = 0.1*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = LqWlsm
                if i>=inh_range and j<inh_range: # IE connection, C = 0.05
                    P = 0.05*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = -LqWlsm
                if i>=inh_range and j>=inh_range: # EE connection, C = 0.2
                    P = 0.2*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = LqWlsm

        for i in range(N):
            W_lsm[i,i] = 0
    
    return W_ins,W_lsm.T #need to transpose matrices for compatibility with torch nn linear

def initWeights_receptive_field_partition(LqWin, LqWlsm, in_conn_density, in_size, num_partitions, inH, inW, inCh, window, lam=9, inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None):
    #ch -> Nz, H (rows) -> Ny, W (cols) -> Nx
    #inH, inW, inCh -> dimensions of input image
    inH = np.int32(inH)
    inW = np.int32(inW)
    inCh = np.int32(inCh)
    partition_Nz = Nz//num_partitions
    N = Nx*Ny*Nz
    partition_N = Nx*Ny*partition_Nz
    
    in_conn_range = np.int32(partition_N*in_conn_density)
    W_ins = []
    for part in range(num_partitions):
        W_in = np.zeros((in_size,N))
        W_in_part = np.zeros((in_size,partition_N))
        for i in range(in_size):
            ch = i//(inH*inW)
            x = (i - ch*inH*inW)%inW
            y = (i - ch*inH*inW)//inW
            res_x = np.int32((x*Nx)/inW)
            res_y = np.int32((y*Ny)/inH)
            res_x_min = res_x - window//2
            if res_x_min < 0:
                res_x_min = 0
            res_x_max = res_x_min + window
            if res_x_max > Nx:
                res_x_max = Nx
                res_x_min = Nx-window
            res_y_min = res_y - window//2
            if res_y_min < 0:
                res_y_min = 0
            res_y_max = res_y_min + window
            if res_y_max > Nx:
                res_y_max = Nx
                res_y_min = Nx-window
            window_locs = []
            for j in range(window):
                window_locs.append((res_y_min + j)*Nx + np.arange(res_x_min, res_x_max))
            window_idxs = np.concatenate(window_locs)
            channel_locs = []
            for k in range(partition_Nz):
                channel_locs.append(k*(Nx*Ny) + window_idxs)
            #input_perm_i = np.arange(partition_N)
            input_perm_i = np.int32(np.concatenate(channel_locs))
            np.random.shuffle(input_perm_i)
            pos_conn = input_perm_i[:in_conn_range]
            neg_conn = input_perm_i[-in_conn_range:]
            W_in_part[i,pos_conn] = LqWin
            W_in_part[i,neg_conn] = -LqWin
        W_in[:, part*partition_N:(part+1)*partition_N] = W_in_part
        W_ins.append(W_in.T)
    
    W_in_part_cp = np.zeros((in_size,partition_N))
    W_in_part_cp[W_in_part>0] = W_in_part[W_in_part>0]
    in_to_res_fanout = np.sum(W_in_part_cp, axis=1)/LqWin
    res_fanin_from_in = np.sum(W_in_part_cp, axis=0)/LqWin
    print('in-res fanout shape: ', in_to_res_fanout.shape, ' average fanout: ', np.mean(in_to_res_fanout))
    print('res fanin from in shape: ', res_fanin_from_in.shape, ' average fanout: ', np.mean(res_fanin_from_in))
    
    input_perm = np.arange(partition_N)
    np.random.shuffle(input_perm) # first 0.2*N indices are inhibitory
    inh_range = np.int32(inh_fr*partition_N) # indices 0 to inh_range-1 are inhibitory

    if init_Wlsm:
        W_lsm_part = np.zeros((partition_N,partition_N))
        W_lsm = np.zeros((N,N))
        for i in range(partition_N):
            posti = input_perm[i] # input_perm[i] is the post-neuron index
            zi = posti//(Nx*Ny)
            yi = (posti-zi*Nx*Ny)//Nx
            xi = (posti-zi*Nx*Ny)%Nx
            for j in range(partition_N):
                prej = input_perm[j] # input_perm[j] is the pre-neuron index
                zj = prej//(Nx*Ny)
                yj = (prej-zj*Nx*Ny)//Nx
                xj = (prej-zj*Nx*Ny)%Nx
                D = ((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                if i<inh_range and j<inh_range: # II connection, C = 0.3
                    P = 0.3*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i<inh_range and j>=inh_range: # EI connection, C = 0.1
                    P = 0.1*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = LqWlsm
                if i>=inh_range and j<inh_range: # IE connection, C = 0.05
                    P = 0.05*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i>=inh_range and j>=inh_range: # EE connection, C = 0.2
                    P = 0.2*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = LqWlsm

        for i in range(partition_N):
            W_lsm_part[i,i] = 0
        
        for part in range(num_partitions):
            W_lsm[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = W_lsm_part

    return W_ins,W_lsm.T #need to transpose matrices for compatibility with torch nn linear

def initWeights_receptive_field_partitionV2(LqWin, LqWlsm, in_conn_density, in_size, num_partitions, inH, inW, inCh, window, lam=9, inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None):
    #V2 - Input goes to particular regions inside a larger Reservoir rather than disconnected smaller reservoir fragments
    #ch -> Nz, H (rows) -> Ny, W (cols) -> Nx
    #inH, inW, inCh -> dimensions of input image
    inH = np.int32(inH)
    inW = np.int32(inW)
    inCh = np.int32(inCh)
    partition_Nz = Nz//num_partitions
    N = Nx*Ny*Nz
    partition_N = Nx*Ny*partition_Nz
    
    in_conn_range = np.int32(partition_N*in_conn_density)
    W_ins = []
    for part in range(num_partitions):
        W_in = np.zeros((in_size,N))
        W_in_part = np.zeros((in_size,partition_N))
        for i in range(in_size):
            ch = i//(inH*inW)
            x = (i - ch*inH*inW)%inW
            y = (i - ch*inH*inW)//inW
            res_x = np.int32((x*Nx)/inW)
            res_y = np.int32((y*Ny)/inH)
            res_x_min = res_x - window//2
            if res_x_min < 0:
                res_x_min = 0
            res_x_max = res_x_min + window
            if res_x_max > Nx:
                res_x_max = Nx
                res_x_min = Nx-window
            res_y_min = res_y - window//2
            if res_y_min < 0:
                res_y_min = 0
            res_y_max = res_y_min + window
            if res_y_max > Nx:
                res_y_max = Nx
                res_y_min = Nx-window
            window_locs = []
            for j in range(window):
                window_locs.append((res_y_min + j)*Nx + np.arange(res_x_min, res_x_max))
            window_idxs = np.concatenate(window_locs)
            channel_locs = []
            for k in range(partition_Nz):
                channel_locs.append(k*(Nx*Ny) + window_idxs)
            #input_perm_i = np.arange(partition_N)
            input_perm_i = np.int32(np.concatenate(channel_locs))
            np.random.shuffle(input_perm_i)
            pos_conn = input_perm_i[:in_conn_range]
            neg_conn = input_perm_i[-in_conn_range:]
            W_in_part[i,pos_conn] = LqWin
            W_in_part[i,neg_conn] = -LqWin
        W_in[:, part*partition_N:(part+1)*partition_N] = W_in_part
        W_ins.append(W_in.T)
    
    W_in_part_cp = np.zeros((in_size,partition_N))
    W_in_part_cp[W_in_part>0] = W_in_part[W_in_part>0]
    in_to_res_fanout = np.sum(W_in_part_cp, axis=1)/LqWin
    res_fanin_from_in = np.sum(W_in_part_cp, axis=0)/LqWin
    print('in-res fanout shape: ', in_to_res_fanout.shape, ' average fanout: ', np.mean(in_to_res_fanout))
    print('res fanin from in shape: ', res_fanin_from_in.shape, ' average fanout: ', np.mean(res_fanin_from_in))
    
    input_perm = np.arange(N)
    np.random.shuffle(input_perm) # first 0.2*N indices are inhibitory
    inh_range = np.int32(inh_fr*N) # indices 0 to inh_range-1 are inhibitory

    if init_Wlsm:
        W_lsm = np.zeros((N,N))
        for i in range(N):
            posti = input_perm[i] # input_perm[i] is the post-neuron index
            zi = posti//(Nx*Ny)
            yi = (posti-zi*Nx*Ny)//Nx
            xi = (posti-zi*Nx*Ny)%Nx
            for j in range(N):
                prej = input_perm[j] # input_perm[j] is the pre-neuron index
                zj = prej//(Nx*Ny)
                yj = (prej-zj*Nx*Ny)//Nx
                xj = (prej-zj*Nx*Ny)%Nx
                D = ((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                if i<inh_range and j<inh_range: # II connection, C = 0.3
                    P = 0.3*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = -LqWlsm
                if i<inh_range and j>=inh_range: # EI connection, C = 0.1
                    P = 0.1*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = LqWlsm
                if i>=inh_range and j<inh_range: # IE connection, C = 0.05
                    P = 0.05*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = -LqWlsm
                if i>=inh_range and j>=inh_range: # EE connection, C = 0.2
                    P = 0.2*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm[prej,posti] = LqWlsm

        for i in range(N):
            W_lsm[i,i] = 0
    
    return W_ins,W_lsm.T #need to transpose matrices for compatibility with torch nn linear

def initWeights_partition_cross_partition_inh(LqWin, LqWlsm, LqWlsm_long, in_conn_density, in_size, num_partitions, lam=9, inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None):
    partition_Nz = Nz//num_partitions
    N = Nx*Ny*Nz
    partition_N = Nx*Ny*partition_Nz
    
    in_conn_range = np.int32(partition_N*in_conn_density)
    W_in_part = np.zeros((in_size,partition_N))
    W_ins = []
    for i in range(in_size):
        input_perm_i = np.arange(partition_N)
        np.random.shuffle(input_perm_i)
        pos_conn = input_perm_i[:in_conn_range]
        neg_conn = input_perm_i[-in_conn_range:]
        W_in_part[i,pos_conn] = LqWin
        W_in_part[i,neg_conn] = -LqWin
    
    W_in_part_cp = np.zeros((in_size,partition_N))
    W_in_part_cp[W_in_part>0] = W_in_part[W_in_part>0]
    in_to_res_fanout = np.sum(W_in_part_cp, axis=1)/LqWin
    res_fanin_from_in = np.sum(W_in_part_cp, axis=0)/LqWin
    print('in-res fanout shape: ', in_to_res_fanout.shape, ' average fanout: ', np.mean(in_to_res_fanout))
    print('res fanin from in shape: ', res_fanin_from_in.shape, ' average fanout: ', np.mean(res_fanin_from_in))
    
    for part in range(num_partitions):
        W_in = np.zeros((in_size,N))
        W_in[:, part*partition_N:(part+1)*partition_N] = W_in_part
        W_ins.append(W_in.T)
    
    input_perm = np.arange(partition_N)
    np.random.shuffle(input_perm) # first 0.2*N indices are inhibitory
    inh_range = np.int32(inh_fr*partition_N) # indices 0 to inh_range-1 are inhibitory

    if init_Wlsm:
        W_lsm_part = np.zeros((partition_N,partition_N))
        W_lsm = np.zeros((N,N))
        W_lsm_long = np.zeros((N,N))
        for i in range(partition_N):
            posti = input_perm[i] # input_perm[i] is the post-neuron index
            zi = posti//(Nx*Ny)
            yi = (posti-zi*Nx*Ny)//Nx
            xi = (posti-zi*Nx*Ny)%Nx
            for j in range(partition_N):
                prej = input_perm[j] # input_perm[j] is the pre-neuron index
                zj = prej//(Nx*Ny)
                yj = (prej-zj*Nx*Ny)//Nx
                xj = (prej-zj*Nx*Ny)%Nx
                D = ((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                if i<inh_range and j<inh_range: # II connection, C = 0.3
                    P = 0.3*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i<inh_range and j>=inh_range: # EI connection, C = 0.1
                    P = 0.1*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = LqWlsm
                if i>=inh_range and j<inh_range: # IE connection, C = 0.05
                    P = 0.05*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i>=inh_range and j>=inh_range: # EE connection, C = 0.2
                    P = 0.2*np.exp(-D/lam)
                    Pu1 = np.random.uniform()
                    if Pu1<P:
                        W_lsm_part[prej,posti] = LqWlsm

        for i in range(partition_N):
            W_lsm_part[i,i] = 0
        
        for i in range (N):
            W_lsm_long[i, (i+partition_N)%N] = -LqWlsm_long
        
        for part in range(num_partitions):
            W_lsm[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = W_lsm_part

    return W_ins,W_lsm.T,W_lsm_long.T #need to transpose matrices for compatibility with torch nn linear


def initWeights_short_long_dist_partition(LqWin, LqWlsm, in_conn_density, in_size, long_dist, num_partitions, lam=9, inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None):
    partition_Nz = Nz//num_partitions
    N = Nx*Ny*Nz
    partition_N = Nx*Ny*partition_Nz
    
    in_conn_range = np.int32(partition_N*in_conn_density)
    W_ins = []
    for part in range(num_partitions):
        W_in = np.zeros((in_size,N))
        W_in_part = np.zeros((in_size,partition_N))
        for i in range(in_size):
            input_perm_i = np.arange(partition_N)
            np.random.shuffle(input_perm_i)
            pos_conn = input_perm_i[:in_conn_range]
            neg_conn = input_perm_i[-in_conn_range:]
            W_in_part[i,pos_conn] = LqWin
            W_in_part[i,neg_conn] = -LqWin
        W_in[:, part*partition_N:(part+1)*partition_N] = W_in_part
        W_ins.append(W_in.T)
    
    W_in_part_cp = np.zeros((in_size,partition_N))
    W_in_part_cp[W_in_part>0] = W_in_part[W_in_part>0]
    in_to_res_fanout = np.sum(W_in_part_cp, axis=1)/LqWin
    res_fanin_from_in = np.sum(W_in_part_cp, axis=0)/LqWin
    print('in-res fanout shape: ', in_to_res_fanout.shape, ' average fanout: ', np.mean(in_to_res_fanout))
    print('res fanin from in shape: ', res_fanin_from_in.shape, ' average fanout: ', np.mean(res_fanin_from_in))
    
    input_perm = np.arange(partition_N)
    np.random.shuffle(input_perm) # first 0.2*N indices are inhibitory
    inh_range = np.int32(inh_fr*partition_N) # indices 0 to inh_range-1 are inhibitory

    if init_Wlsm:
        W_lsm_part = np.zeros((partition_N,partition_N))
        conn_P1_part = np.zeros((partition_N,partition_N))
        conn_P2_part = np.zeros((partition_N,partition_N))
        W_lsm = np.zeros((N,N))
        conn_P1 = np.zeros((N,N))
        conn_P2 = np.zeros((N,N))
        for i in range(partition_N):
            posti = input_perm[i] # input_perm[i] is the post-neuron index
            zi = posti//(Nx*Ny)
            yi = (posti-zi*Nx*Ny)//Nx
            xi = (posti-zi*Nx*Ny)%Nx
            for j in range(partition_N):
                prej = input_perm[j] # input_perm[j] is the pre-neuron index
                zj = prej//(Nx*Ny)
                yj = (prej-zj*Nx*Ny)//Nx
                xj = (prej-zj*Nx*Ny)%Nx
                D = ((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                if i<inh_range and j<inh_range: # II connection, C = 0.3
                    P = 0.3*np.exp(-D/lam)
                    P2 = 0.3*np.exp(-((np.sqrt(D)-long_dist)**2)/lam)
                    conn_P1_part[prej,posti] = P
                    conn_P2_part[prej,posti] = P2
                    Pu1 = np.random.uniform()
                    if Pu1<P or Pu1<P2:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i<inh_range and j>=inh_range: # EI connection, C = 0.1
                    P = 0.1*np.exp(-D/lam)
                    P2 = 0.1*np.exp(-((np.sqrt(D)-long_dist)**2)/lam)
                    conn_P1_part[prej,posti] = P
                    conn_P2_part[prej,posti] = P2
                    Pu1 = np.random.uniform()
                    if Pu1<P or Pu1<P2:
                        W_lsm_part[prej,posti] = LqWlsm
                if i>=inh_range and j<inh_range: # IE connection, C = 0.05
                    P = 0.05*np.exp(-D/lam)
                    P2 = 0.05*np.exp(-((np.sqrt(D)-long_dist)**2)/lam)
                    conn_P1_part[prej,posti] = P
                    conn_P2_part[prej,posti] = P2
                    Pu1 = np.random.uniform()
                    if Pu1<P or Pu1<P2:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i>=inh_range and j>=inh_range: # EE connection, C = 0.2
                    P = 0.2*np.exp(-D/lam)
                    P2 = 0.2*np.exp(-((np.sqrt(D)-long_dist)**2)/lam)
                    conn_P1_part[prej,posti] = P
                    conn_P2_part[prej,posti] = P2
                    Pu1 = np.random.uniform()
                    if Pu1<P or Pu1<P2:
                        W_lsm_part[prej,posti] = LqWlsm

        for i in range(partition_N):
            W_lsm_part[i,i] = 0
        
        for part in range(num_partitions):
            W_lsm[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = W_lsm_part
            conn_P1[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = conn_P1_part
            conn_P2[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = conn_P2_part

    return W_ins,W_lsm.T,conn_P1.T,conn_P2.T #need to transpose matrices for compatibility with torch nn linear

def initWeights_receptive_field_short_long_dist_partition(LqWin, LqWlsm, in_conn_density, in_size, inH, inW, inCh, window, long_dist, num_partitions, lam=9, inh_fr=0.2, Nx=10, Ny=10, Nz=10, init_Wlsm=True, W_lsm=None):
    #ch -> Nz, H (rows) -> Ny, W (cols) -> Nx
    #inH, inW, inCh -> dimensions of input image
    inH = np.int32(inH)
    inW = np.int32(inW)
    inCh = np.int32(inCh)
    partition_Nz = Nz//num_partitions
    N = Nx*Ny*Nz
    partition_N = Nx*Ny*partition_Nz
    
    in_conn_range = np.int32(partition_N*in_conn_density)
    W_ins = []
    for part in range(num_partitions):
        W_in = np.zeros((in_size,N))
        W_in_part = np.zeros((in_size,partition_N))
        for i in range(in_size):
            ch = i//(inH*inW)
            x = (i - ch*inH*inW)%inW
            y = (i - ch*inH*inW)//inW
            res_x = np.int32((x*Nx)/inW)
            res_y = np.int32((y*Ny)/inH)
            res_x_min = res_x - window//2
            if res_x_min < 0:
                res_x_min = 0
            res_x_max = res_x_min + window
            if res_x_max > Nx:
                res_x_max = Nx
                res_x_min = Nx-window
            res_y_min = res_y - window//2
            if res_y_min < 0:
                res_y_min = 0
            res_y_max = res_y_min + window
            if res_y_max > Nx:
                res_y_max = Nx
                res_y_min = Nx-window
            window_locs = []
            for j in range(window):
                window_locs.append((res_y_min + j)*Nx + np.arange(res_x_min, res_x_max))
            window_idxs = np.concatenate(window_locs)
            channel_locs = []
            for k in range(partition_Nz):
                channel_locs.append(k*(Nx*Ny) + window_idxs)
            #input_perm_i = np.arange(partition_N)
            input_perm_i = np.int32(np.concatenate(channel_locs))
            np.random.shuffle(input_perm_i)
            pos_conn = input_perm_i[:in_conn_range]
            neg_conn = input_perm_i[-in_conn_range:]
            W_in_part[i,pos_conn] = LqWin
            W_in_part[i,neg_conn] = -LqWin
        W_in[:, part*partition_N:(part+1)*partition_N] = W_in_part
        W_ins.append(W_in.T)
    
    W_in_part_cp = np.zeros((in_size,partition_N))
    W_in_part_cp[W_in_part>0] = W_in_part[W_in_part>0]
    in_to_res_fanout = np.sum(W_in_part_cp, axis=1)/LqWin
    res_fanin_from_in = np.sum(W_in_part_cp, axis=0)/LqWin
    print('in-res fanout shape: ', in_to_res_fanout.shape, ' average fanout: ', np.mean(in_to_res_fanout))
    print('res fanin from in shape: ', res_fanin_from_in.shape, ' average fanout: ', np.mean(res_fanin_from_in))
    
    input_perm = np.arange(partition_N)
    np.random.shuffle(input_perm) # first 0.2*N indices are inhibitory
    inh_range = np.int32(inh_fr*partition_N) # indices 0 to inh_range-1 are inhibitory

    if init_Wlsm:
        W_lsm_part = np.zeros((partition_N,partition_N))
        conn_P1_part = np.zeros((partition_N,partition_N))
        conn_P2_part = np.zeros((partition_N,partition_N))
        W_lsm = np.zeros((N,N))
        conn_P1 = np.zeros((N,N))
        conn_P2 = np.zeros((N,N))
        for i in range(partition_N):
            posti = input_perm[i] # input_perm[i] is the post-neuron index
            zi = posti//(Nx*Ny)
            yi = (posti-zi*Nx*Ny)//Nx
            xi = (posti-zi*Nx*Ny)%Nx
            for j in range(partition_N):
                prej = input_perm[j] # input_perm[j] is the pre-neuron index
                zj = prej//(Nx*Ny)
                yj = (prej-zj*Nx*Ny)//Nx
                xj = (prej-zj*Nx*Ny)%Nx
                D = ((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)
                if i<inh_range and j<inh_range: # II connection, C = 0.3
                    P = 0.3*np.exp(-D/lam)
                    P2 = 0.3*np.exp(-((np.sqrt(D)-long_dist)**2)/lam)
                    conn_P1_part[prej,posti] = P
                    conn_P2_part[prej,posti] = P2
                    Pu1 = np.random.uniform()
                    if Pu1<P or Pu1<P2:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i<inh_range and j>=inh_range: # EI connection, C = 0.1
                    P = 0.1*np.exp(-D/lam)
                    P2 = 0.1*np.exp(-((np.sqrt(D)-long_dist)**2)/lam)
                    conn_P1_part[prej,posti] = P
                    conn_P2_part[prej,posti] = P2
                    Pu1 = np.random.uniform()
                    if Pu1<P or Pu1<P2:
                        W_lsm_part[prej,posti] = LqWlsm
                if i>=inh_range and j<inh_range: # IE connection, C = 0.05
                    P = 0.05*np.exp(-D/lam)
                    P2 = 0.05*np.exp(-((np.sqrt(D)-long_dist)**2)/lam)
                    conn_P1_part[prej,posti] = P
                    conn_P2_part[prej,posti] = P2
                    Pu1 = np.random.uniform()
                    if Pu1<P or Pu1<P2:
                        W_lsm_part[prej,posti] = -LqWlsm
                if i>=inh_range and j>=inh_range: # EE connection, C = 0.2
                    P = 0.2*np.exp(-D/lam)
                    P2 = 0.2*np.exp(-((np.sqrt(D)-long_dist)**2)/lam)
                    conn_P1_part[prej,posti] = P
                    conn_P2_part[prej,posti] = P2
                    Pu1 = np.random.uniform()
                    if Pu1<P or Pu1<P2:
                        W_lsm_part[prej,posti] = LqWlsm

        for i in range(partition_N):
            W_lsm_part[i,i] = 0
        
        for part in range(num_partitions):
            W_lsm[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = W_lsm_part
            conn_P1[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = conn_P1_part
            conn_P2[part*partition_N:(part+1)*partition_N, part*partition_N:(part+1)*partition_N] = conn_P2_part

    return W_ins,W_lsm.T,conn_P1.T,conn_P2.T #need to transpose matrices for compatibility with torch nn linear
