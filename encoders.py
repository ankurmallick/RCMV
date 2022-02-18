#Encoders for different coded computing schemes

import numpy as np

def RS(m,c=0.03,delta=0.5):
    
    #Returns Robust Soliton CDF for given parameters
    
    S=c*np.log(m/delta)*np.sqrt(m)
    rho=0.0
    tau=0.0
    r=round(m/S)
    u=np.zeros((1,m))
    
    for d_iter in range(m):
        d=float(d_iter+1)
        if d==1:
            rho=1/float(m)
        else:
            rho=1/(d*(d-1))
        if d>=1 and d<r:
            tau=1/(d*r)
        elif d==r:
            tau=np.log(S/delta)/r
        else:
            tau=0
        u[0][d_iter]=tau+rho
        
    u=u/np.sum(u)
    return np.ravel(u)

def lt_enc(A,p=20,alpha=2.0):
    
    #Generates LT Encoded matrix A_e with alpha times the rows of A
    #A_e will be distributed equally over p workers
    
    m, n = A.shape
    m_e = int(alpha*m)
    A_e = np.zeros((m_e,n))
    u = RS(m)
    degs = 1+np.random.choice(m, size=m_e, replace=True, p=u) #Degree chosen according to RS distributions
    encmat = np.zeros((m_e,m))
    
    for enc_row in range(m_e):
        rows = np.random.choice(m, size=degs[enc_row], replace=False, p=None)
        encmat[enc_row][rows] = 1
        A_e[enc_row] = np.sum(A[rows],0)

    return A_e, encmat

def mds_enc(A,p=20,k=10):
    
    #Generates MDS Encoded matrix A_e using a (p,k) MDS code on A
    #A_e will be distributed equally over p workers
    
    m, n = A.shape
    s = int(m/k) #Size of block at each worker
    m_e = p*s
    A_e = np.empty((m_e,n))
    encmat = np.random.normal(size=(p,k))
    
    for i in range(p):
        A_e[i*s:(i+1)*s] = sum([encmat[i][j]*A[j*s:(j+1)*s] for j in range(k)])
    
    return A_e, encmat
    

def rep_enc(A,p=20,r=2.0):
    
    #Generates a r-Rep Encoded matrix A_e by copying each row of A r times
    #A_e will be distributed equally over p workers
    
    A_e = np.concatenate([A for _ in range(r)],0)
    encmat = np.concatenate([np.eye(int(p/r)) for _ in range(r)],0)
    
    return A_e, encmat
    