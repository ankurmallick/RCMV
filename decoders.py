#Decoders for different coded computing schemes

import numpy as np
from scipy.sparse import csr_matrix

def lt_dec(b_e,decmat):
    
    #b_e is the vector of encoded row-vector products
    #decmat is the subset of encmat corresponding to completed products
    
    num_par, num_dec = decmat.shape
    degrees = decmat.sum(1)
    b_d = np.zeros(num_dec)
    dec_ctr = 0
    #Starting ripple
    while dec_ctr < num_dec:
        ripple = np.where(degrees==1)[0] 
        if len(ripple) == 0:
            return None
        else:
            #Decode
            par_pos = ripple[0]
            dec_pos = np.where(decmat[par_pos]==1)[0][0]
            b_d[dec_pos] = b_e[par_pos] 
            #Peel
            dec_neighbors = np.where(decmat[:,dec_pos]==1)[0] #Neighbors of decoded symbol
            b_e[dec_neighbors] -= b_d[dec_pos]
            #Update graph
            decmat[dec_neighbors,dec_pos] = 0
            degrees[dec_neighbors] -= 1
            #Increment counter
            dec_ctr += 1
    
    return b_d

def rep_dec(b_e,decmat):
    
    #b_e is the vector of encoded row-vector products
    #decmat is the subset of encmat corresponding to completed products
    
    repinv= decmat.T #decmat is permutation of identity
    m = b_e.shape[0]
    k = decmat.shape[0]
    s = int(m/k)
    b_d = np.empty(b_e.shape)
    
    for i in range(k):
        j = np.where(repinv[i]==1)[0][0]
        b_d[i*s:(i+1)*s] = b_e[j*s:(j+1)*s]
    
    return b_d

def mds_dec(b_e,decmat):
    
    #b_e is the vector of encoded row-vector products
    #decmat is the subset of encmat corresponding to completed products
    
    mdsinv=np.linalg.inv(decmat)
    m = b_e.shape[0]
    k = decmat.shape[0]
    s = int(m/k)
    b_d = np.empty(b_e.shape)
    
    for i in range(k):
        b_d[i*s:(i+1)*s] = sum([mdsinv[i][j]*b_e[j*s:(j+1)*s] for j in range(k)])
    
    return b_d

