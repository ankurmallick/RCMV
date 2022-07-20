import numpy as np
import matplotlib.pyplot as plt
from encoders import mds_encoder, lt_encoder, syslt_pardist
from scipy.stats import powerlaw
from plotting_helpers import get_tail_plots
from utils import setupdelay, get_degree_dist, nnz_task_allocator
plt.style.use('publication')

class SpMV:
    def __init__(self, tau, delay_info, mat_info, rows, cols, num_workers, mdsnum, c, delta, alpha):
        self.comp_time=tau
        self.delay_info=delay_info
        self.mat_info = mat_info
        self.num_workers=num_workers
        self.numtasks = rows #Total number of tasks(dot-products)
        self.degree_dist, self.degree_dist_mds, self.degree_dist_lt, self.degree_dist_syslt = get_degree_dist(mat_info, rows, cols, num_workers, mdsnum, c, delta, alpha)
        self.degree_dist_syslt_parity = self.degree_dist_syslt[rows:]
        
    def rep(self,numreps):
        tasks_per_worker = int((self.numtasks/self.num_workers)*numreps)
        worker_loads = np.zeros(self.num_workers)
        for worker in range(self.num_workers):
            worker_startpos = worker*tasks_per_worker
            worker_endpos = worker_startpos + tasks_per_worker
            worker_degrees = self.degree_dist[worker_startpos:worker_endpos]
            worker_loads[worker] = np.sum(worker_degrees)
        latency = 0
        numgroups = int(self.num_workers/numreps)
        grouptimes = np.zeros((numgroups,numreps))
        # groupcomps = np.zeros((numgroups,numreps))
        mingrouptimes=np.zeros(numgroups)
        for group in range(numgroups):
            for group_worker in range(numreps):
                worker = group*numreps + group_worker
                grouptimes[group][group_worker]=setupdelay(self.delay_info,1)+self.comp_time*worker_loads[worker]
        mingrouptimes=np.amin(grouptimes,axis=1) #Fastest worker in each group
        latency=np.amax(mingrouptimes) #Slowest among fastest workers of each group
        return latency
    
    def rep_ord(self,numreps):
        task_assignment = nnz_task_allocator(self.degree_dist,self.num_workers)
        worker_loads = np.zeros(self.num_workers)
        for worker in range(self.num_workers):
            worker_degrees = self.degree_dist[task_assignment==worker]
            worker_loads[worker] = np.sum(worker_degrees)
        latency = 0
        numgroups = int(self.num_workers/numreps)
        grouptimes = np.zeros((numgroups,numreps))
        mingrouptimes=np.zeros(numgroups)
        for group in range(numgroups):
            for group_worker in range(numreps):
                worker = group*numreps + group_worker
                grouptimes[group][group_worker]=setupdelay(self.delay_info,1)+self.comp_time*worker_loads[worker]
        mingrouptimes=np.amin(grouptimes,axis=1) #Fastest worker in each group
        latency=np.amax(mingrouptimes) #Slowest among fastest workers of each group
        return latency

    def mds(self,mdsnum):
        tasks_per_worker=int(np.ceil((self.numtasks/mdsnum)))
        worker_loads = np.zeros(self.num_workers)
        for worker in range(self.num_workers):
            worker_startpos = worker*tasks_per_worker
            worker_endpos = worker_startpos + tasks_per_worker
            worker_degrees = self.degree_dist_mds[worker_startpos:worker_endpos]
            worker_loads[worker] = np.sum(worker_degrees)
        print (worker_loads)
        latency=0
        # comp=0
        delays = setupdelay(self.delay_info,self.num_workers)
        print (delays)
        workertimes=delays + self.comp_time*worker_loads
        print (workertimes)
        workertimes_order=np.argsort(workertimes)
        latency=workertimes[workertimes_order[mdsnum-1]]
        return latency

    def lt(self,alpha,decthresh):
        comp = decthresh
        tasks_per_worker=int(alpha*self.numtasks/self.num_workers)
        worker_loads = np.zeros(self.num_workers)
        worker_degrees = {}
        for worker in range(self.num_workers):
            worker_startpos = worker*tasks_per_worker
            worker_endpos = worker_startpos + tasks_per_worker
            worker_degrees[worker] = self.degree_dist_lt[worker_startpos:worker_endpos].tolist()
            worker_loads[worker] = np.sum(worker_degrees[worker])
        latency = 0
        setuptime=setupdelay(self.delay_info,self.num_workers)
        currenttime=setuptime
        workercount=np.zeros(self.num_workers)
        for counter in range(comp):
            # print (currenttime)
            minpos=np.argmin(currenttime)
            # print (minpos)
            # print ([latency,currenttime[minpos]])
            latency+=currenttime[minpos]
            currenttime=currenttime-currenttime[minpos]
            if workercount[minpos]==tasks_per_worker:
                #Worker is done with all tasks
                currenttime[minpos]=float('Inf')
                # workercount=np.delete(workercount,minpos)
            else:
                currenttime[minpos]=self.comp_time*worker_degrees[minpos].pop()
                workercount[minpos]+=1
        return latency
    
    def lt_ord(self,alpha,decthresh):
        comp = decthresh
        task_assignment = nnz_task_allocator(self.degree_dist_lt,self.num_workers)
        worker_loads = np.zeros(self.num_workers)
        worker_degrees = {}
        for worker in range(self.num_workers):
            worker_degrees[worker] = self.degree_dist_lt[task_assignment==worker].tolist()
            worker_loads[worker] = np.sum(worker_degrees[worker])
        latency = 0
        setuptime=setupdelay(self.delay_info,self.num_workers)
        currenttime=setuptime
        workercount=np.zeros(self.num_workers)
        for counter in range(comp):
            # print ([counter,comp])
            # print (currenttime)
            minpos=np.argmin(currenttime)
            # print (minpos)
            # print ([latency,currenttime[minpos]])
            latency+=currenttime[minpos]
            currenttime=currenttime-currenttime[minpos]
            if len(worker_degrees[minpos])==0:
                #Worker is done with all tasks
                currenttime[minpos]=float('Inf')
                # workercount=np.delete(workercount,minpos)
            else:
                currenttime[minpos]=self.comp_time*worker_degrees[minpos].pop()
                workercount[minpos]+=1
        return latency
    
    def sys_lt(self,alpha,decthresh):
        comp = decthresh
        systasks_per_worker = int(self.numtasks/self.num_workers)
        partasks_per_worker=int((alpha-1)*self.numtasks/self.num_workers)
        tasks_per_worker = systasks_per_worker + partasks_per_worker
        worker_loads = np.zeros(self.num_workers)
        worker_degrees = {}
        # worker_pardegrees = np.ones(partasks_per_worker)*self.parity_length
        for worker in range(self.num_workers):
            worker_sys_startpos = worker*systasks_per_worker
            worker_sys_endpos = worker_sys_startpos + systasks_per_worker
            worker_par_startpos = worker*partasks_per_worker
            worker_par_endpos = worker_par_startpos + partasks_per_worker
            worker_sysdegrees = self.degree_dist[worker_sys_startpos:worker_sys_endpos]
            worker_pardegrees = self.degree_dist_syslt_parity[worker_par_startpos:worker_par_endpos]
            worker_degrees[worker] = np.concatenate([worker_sysdegrees, worker_pardegrees]).tolist()
            worker_loads[worker] = np.sum(worker_degrees[worker])
        latency = 0
        setuptime=setupdelay(self.delay_info,self.num_workers)
        currenttime=setuptime
        workercount=np.zeros(self.num_workers)
        for counter in range(comp):
            # print (currenttime)
            minpos=np.argmin(currenttime)
            # print (minpos)
            # print ([latency,currenttime[minpos]])
            latency+=currenttime[minpos]
            currenttime=currenttime-currenttime[minpos]
            if workercount[minpos]==tasks_per_worker:
                #Worker is done with all tasks
                currenttime[minpos]=float('Inf')
                # workercount=np.delete(workercount,minpos)
            else:
                currenttime[minpos]=self.comp_time*worker_degrees[minpos].pop()
                workercount[minpos]+=1
        return latency
    
    def sys_lt_ord(self,alpha,decthresh):
        comp = decthresh
        sys_task_assignment = nnz_task_allocator(self.degree_dist,self.num_workers)
        par_task_assignment = nnz_task_allocator(self.degree_dist_syslt_parity,self.num_workers)
        worker_loads = np.zeros(self.num_workers)
        partasks_per_worker=int((alpha-1)*self.numtasks/self.num_workers)
        # worker_pardegrees = np.ones(partasks_per_worker)*self.parity_length
        worker_degrees = {}
        for worker in range(self.num_workers):
            worker_sysdegrees = self.degree_dist[sys_task_assignment==worker]
            worker_pardegrees = self.degree_dist_syslt_parity[par_task_assignment==worker]
            worker_degrees[worker] = np.concatenate([worker_sysdegrees, worker_pardegrees]).tolist()
            worker_loads[worker] = np.sum(worker_degrees[worker])
        latency = 0
        setuptime=setupdelay(self.delay_info,self.num_workers)
        currenttime=setuptime
        workercount=np.zeros(self.num_workers)
        for counter in range(comp):
            # print ([counter,comp])
            # print (currenttime)
            minpos=np.argmin(currenttime)
            # print (minpos)
            # print ([latency,currenttime[minpos]])
            latency+=currenttime[minpos]
            currenttime=currenttime-currenttime[minpos]
            if len(worker_degrees[minpos])==0:
                #Worker is done with all tasks
                currenttime[minpos]=float('Inf')
                # workercount=np.delete(workercount,minpos)
            else:
                currenttime[minpos]=self.comp_time*worker_degrees[minpos].pop()
                workercount[minpos]+=1
        return latency