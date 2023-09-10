import numpy as np

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
