import numpy as np
import matplotlib.pyplot as plt
from encoders import mds_encoder, lt_encoder, syslt_pardist

def get_ER_mat(p,rows,cols):
    mat = np.random.binomial(1,p,size=(rows,cols))
    return mat

def get_PL_mat(gamma,rows,cols):
    PL_dist = np.asarray([x**(-gamma) for x in range(1,cols+1)])
    PL_dist = PL_dist/np.sum(PL_dist)
    row_nnz = 1+np.random.choice(cols, size=rows, replace=True, p=PL_dist) #Degree
    mat = np.zeros((rows,cols))
    for mat_row in range(rows):
        nnz_cols=np.random.choice(cols, size=row_nnz[mat_row], replace=False, p=None) #Col indices of nnz
        mat[mat_row,nnz_cols] = 1
    return mat
        
def setupdelay(delay_info,num_workers):
    if delay_info['type']=='Exp':
        return np.random.exponential(scale=delay_info['param'],size=num_workers)
    elif delay_info['type']=='Par':
        mu=1 #Shape parameter
        return mu*(1.0+np.random.pareto(a=delay_info['param'],size=num_workers))

def nnz_task_allocator(degree_dist,num_workers):
    nnz = np.sum(degree_dist)
    tasks_per_worker = nnz/num_workers
    current_worker = 0
    worker_load = 0
    assignment = np.zeros(degree_dist.shape)
    num_tasks = degree_dist.shape[0]
    for task in range(num_tasks):
        degree = degree_dist[task]
        worker_load+= degree
        assignment[task] = current_worker
        if worker_load>=tasks_per_worker and current_worker<num_workers-1:
            #Overflow and not last worker 
            #reset
            worker_load = 0
            current_worker+= 1
    return assignment

def get_degree_dist(mat_info, rows, cols, num_workers, mdsnum, c, delta, alpha):
    if mat_info['type'] == 'ER':
        #Erdos-Renyi
        mat = get_ER_mat(mat_info['param'],rows,cols)
    elif mat_info['type'] == 'PL':
        #Power-Law
        mat = get_PL_mat(mat_info['param'],rows,cols)
    degree_dist = np.count_nonzero(mat,axis=1)
    print ('Matrix generated')
    mds_encmat = mds_encoder(mat,mdsnum,num_workers)
    degree_dist_mds = np.count_nonzero(mds_encmat,axis=1)
    print ('MDS Encoding done')
    lt_encmat = lt_encoder(mat, c, delta, alpha, num_workers)
    degree_dist_lt = np.count_nonzero(lt_encmat,axis=1)
    print ('LT Encoding done')
    degree_dist_syslt_parity = syslt_pardist(mat, c, delta, alpha, num_workers)
    degree_dist_syslt = np.concatenate([degree_dist,degree_dist_syslt_parity],axis=0)
    print ('SysLT Encoding done')
    return degree_dist, degree_dist_mds, degree_dist_lt, degree_dist_syslt