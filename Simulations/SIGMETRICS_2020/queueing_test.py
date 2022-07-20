import numpy as np
import matplotlib.pyplot as plt
from delaysim import delaysim
plt.style.use('publication')

def get_latency(worker):
    # return float(worker+1)/10.0
    return np.random.exponential(scale=0.5)+1

def get_arrival_times(rate,time_stop):
    time_start = 0
    time_arr = []
    while time_start < time_stop:
        time_start += np.random.exponential(scale=1/rate)
        if time_start <= time_stop:
            time_arr.append(time_start)
    return time_arr

def queueing(rates_list, num_iters, num_jobs, num_workers):
    latency_list = []
    for rate in rates_list:
        print (rate)
        latency = 0
        for iter in range(num_iters):
            time_arr = np.cumsum(np.random.exponential(scale=1/rate,size=num_jobs)) #Poisson arrival time instants
            # print (time_arr)
            time_dep = np.zeros((num_jobs,num_workers)) #Departure time instants of each job from each worker
            jobs_queue = [[] for _ in range(num_workers)] #Queue of pending jobs
            for job in range(num_jobs):
                for worker in range(num_workers):
                    jobs_queue_current = jobs_queue[worker][:] #Pending jobs from past data at current worker
                    for past_job in jobs_queue[worker]:
                        if time_dep[past_job][worker] < time_arr[job]:
                            #Worker has processed past job before current job has arrived
                            jobs_queue_current.remove(past_job)
                    jobs_queue[worker] = jobs_queue_current[:]
                for worker in range(num_workers):
                    job_latency = get_latency(worker)
                    # print (job_latency)
                    time_dep[job][worker] = job_latency+time_arr[job]
                    if len(jobs_queue[worker])>0:
                        #If worker queue is not empty add waiting time to departure time
                        time_dep[job][worker] += time_dep[jobs_queue[worker][-1]][worker] - time_arr[job] #Waiting time = Time until job just before current job has departed
                    jobs_queue[worker].append(job)
            latency_vect = np.max(time_dep,axis=1) - time_arr
            latency += np.mean(latency_vect)
        latency/= num_iters
        latency_list.append(latency)
    return np.asarray(latency_list)

num_jobs = 100
num_iters = 10
num_workers = 10
rates_list = np.arange(start=0.1,stop=1.1, step=0.1).tolist()
latency_list = queueing(rates_list, num_iters, num_jobs, num_workers)
plt.plot(rates_list,latency_list)
plt.show()