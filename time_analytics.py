import numpy as np
import heapq

def get_worker_info(worker_dict,stop_time):
    
    times_dict = {}
    comps_dict = {}
    
    for w in worker_dict:
        times_dict[w] = max([t[0] for t in worker_dict[w]]) 
        comps_dict[w] = sum([1 for t in worker_dict[w] if t[0] <= stop_time])
    
    return times_dict, comps_dict

def lt_ana(worker_dict,encmat,decthresh):
    
    #worker_dict[w] : List of (time,index,value) tuples from worker w
    #encmat: Generator/Encoding matrix for the LT Code
    #decthresh: Number of encoded row vector products required for successful decoding w.h.p
    #Typically for 10000 rows, ~11000 encoded rows are required for successful decoding w.p. 0.99 
    
    tasks_list = []
    
    for w in worker_dict:
        tasks_list.extend(worker_dict[w])
    
    tasks_list_sub = heapq.nsmallest(decthresh, tasks_list, key = lambda x: x[0])
    
    decmat = encmat[[t[1] for t in tasks_list_sub]]
    b_e = np.array([t[2] for t in tasks_list_sub])
    stop_time = max([t[0] for t in tasks_list_sub])
    
    return b_e, decmat, stop_time

def mds_ana(worker_dict,encmat,mdsnum):
    
    #worker_dict[w] : List of (time,index,value) tuples from worker w
    #encmat: Generator/Encoding matrix for the MDS Code
    #mdsnum: Number of workers that need to be done with all tasks for successful decoding
    #mdsnum = k if a (p,k) MDS Code is used for encoding
    
    times_dict = {}
    
    for w in worker_dict:
        times_dict[w] = max([t[0] for t in worker_dict[w]]) 
    
    top_workers = heapq.nsmallest(mdsnum, times_dict, key= times_dict.get)
    decmat = encmat[top_workers]
    b_e_list = []
    for w in top_workers:
        b_e_list += [t[2] for t in worker_dict[w]]
    b_e = np.array(b_e_list)
    stop_time = max([times_dict[w] for w in top_workers])
    
    return b_e, decmat, stop_time

def rep_ana(worker_dict,encmat,numrep):
    
    #worker_dict[w] : List of (time,index,value) tuples from worker w
    #encmat: Generator/Encoding matrix for Replication
    #numrep: Number of replicas
    #m rows and r-replication (numrep = r) implies each submatrix with m/r rows is replicated at r workers
    #Atleast one replica of each submatrix needs to be completed for successful decoding
    
    times_dict = {}
    num_groups = int(encmat.shape[0]/numrep)
    top_workers = []
    
    for g in range(num_groups):
        min_time = float('Inf')
        min_ind = None
        for r in range(numrep):
            w = num_groups*r + g
            t_w = max([t[0] for t in worker_dict[w]]) 
            if t_w < min_time:
                min_ind = w
                min_time = t_w
        top_workers.append(min_ind)
        times_dict[min_ind] = min_time
    
    decmat = encmat[top_workers]
    b_e_list = []
    for w in top_workers:
        b_e_list += [t[2] for t in worker_dict[w]]
    b_e = np.array(b_e_list)
    stop_time = max([times_dict[w] for w in top_workers])
    
    return b_e, decmat, stop_time
        
        
        
    
    