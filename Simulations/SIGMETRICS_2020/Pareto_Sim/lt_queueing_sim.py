import numpy as np
import matplotlib.pyplot as plt
from delaysim import delaysim
plt.style.use('publication')

# num_mc=500 #Number of Monte-Carlo Simulations
mu=3.0
tau=mu/1000
numtasks=10000
delay_type = 'par'

def set_time(time_array,t_set):
    #Set all times in time_array that are larger than t_set to t_set
    for ind in range(time_array.shape[0]):
        time_diff = time_array[ind] - t_set
        if time_diff > 0:
            comp_diff = np.floor(time_diff/tau)
            time_array[ind] -= tau*comp_diff
    return time_array

def get_start_times(alpha,num_workers):
    if delay_type=='exp':
        t_start = np.random.exponential(scale=mu,size=num_workers)
        return t_start
    elif delay_type == 'par':
        # mu=1 #Shape parameter
        t_start = 1*(1.0+np.random.pareto(a=mu,size=num_workers))
        return t_start

def checkthresh(current_comps,workerthresh):
    worker_list = list(range(current_comps.shape[0]))
    busy_list = [worker for worker in worker_list if current_comps[worker]<workerthresh]
    done_list = [worker for worker in worker_list if current_comps[worker]>=workerthresh]
    return busy_list, done_list

def get_lt_times(start_times,start_comps,end_time,alpha,decthresh):
    #Given a vector of start_times and an end_time find how many computations have been completed
    #DLB is LT with alpha=inf, decthresh=numtasks
    #Start at start_times +tau*start_comps
    done_flag = False #Set to true if computation is completed
    num_workers = start_times.shape[0]
    workerthresh = alpha*numtasks/num_workers
    lt_times = start_times + tau*start_comps #Time at which each worker currently is located
    lt_comps = start_comps
    busy_list, done_list = checkthresh(lt_comps,workerthresh)
    current_times = lt_times[busy_list]
    current_comps = lt_comps[busy_list]
    stop_time = None
    current_times+= tau
    while np.amin(current_times) <= end_time:
        #At least another computation is completed
        minpos=np.argmin(current_times)
        current_comps[minpos] += 1
        total_comp = np.sum(current_comps) + np.sum(lt_comps[done_list]) #Current comps + Done comps
        if total_comp>=decthresh:
            #Task is done
            done_flag = True
            stop_time = current_times[minpos]
            break
        if current_comps[minpos] == workerthresh:
            lt_times[busy_list] = current_times
            lt_comps[busy_list] = current_comps
            done_list.append(busy_list.pop(minpos))
            current_times = lt_times[busy_list]
            current_comps = lt_comps[busy_list]
        else: 
            current_times[minpos]+= tau
    lt_times[busy_list] = current_times
    lt_comps[busy_list] = current_comps
    if not done_flag:
        lt_times[busy_list]-= tau
    return lt_times,lt_comps, done_flag,stop_time

def update_one(past_job,t0,done_tasks,pending_jobs_current,start_times,stop_times,alpha,decthresh):
    num_jobs, num_workers = start_times.shape
    start_times_old= start_times[past_job]
    lt_times, lt_comps, done_flag, stop_time = get_lt_times(start_times[past_job],done_tasks[past_job],t0,alpha,decthresh)
    if done_flag:
        #Past Job is done
        stop_times[past_job] = stop_time
        if alpha == float('inf') and past_job<num_jobs-1:
            start_times[past_job+1] = lt_times
        else:
            start_times_diff = start_times_old + tau*(alpha*numtasks/num_workers) - lt_times
            for pending_job in range(past_job+1,num_jobs):
                if np.sum(start_times[pending_job,:])>0:
                    #Job is indeed pending
                    start_times[pending_job,:] -= start_times_diff
        pending_jobs_current.remove(past_job)
    done_tasks[past_job]= lt_comps
    return done_tasks, pending_jobs_current, start_times, stop_times

def update_all(pending_jobs,start_times,stop_times,done_tasks,alpha,decthresh_sample,t0):
    #Updating queues at t=t0
    pending_jobs_current = pending_jobs[:] #Currently pending jobs (initial estimate)
    for past_job in pending_jobs:
        #Check if any worker has made progress on the past job at the current time
        if np.any(start_times[past_job]+tau*done_tasks[past_job]<t0):
            #If some worker has made progress then update done_tasks, pending_jobs, start_times and stop_times for all jobs and workers
            done_tasks, pending_jobs_current, start_times, stop_times = update_one(past_job,t0,done_tasks,pending_jobs_current,start_times,stop_times,alpha,decthresh_sample[past_job])
        else:
            #If no worker has made progress on past_job then it definitely has not made progress on subsequent jobs
            break
    pending_jobs = pending_jobs_current[:]
    return pending_jobs, start_times, stop_times, done_tasks

def lt_queue(time_arr,num_workers,alpha,decthresh_sample):
    num_jobs = time_arr.shape[0]
    pending_jobs = [] #List of currently pending jobs
    done_tasks = {} #For each job, maintain an array of number of tasks done by each worker
    start_times = np.zeros((num_jobs,num_workers)) #Current forecast of start time (after setup time) of each pending job from each worker
    stop_times = np.zeros(num_jobs) #Time instant at which each job is done
    #Simulating fork join queue
    for job in range(num_jobs):
        # print (pending_jobs)
        pending_jobs, start_times, stop_times, done_tasks = update_all(pending_jobs,start_times, stop_times,done_tasks,alpha,decthresh_sample,time_arr[job]) #Update data at current time
        done_tasks[job] = np.zeros(num_workers)
        start_times[job] = get_start_times(alpha,num_workers)+time_arr[job]
        if len(pending_jobs)>0:
            for worker in range(num_workers):
            #If worker queue is not empty add waiting time to departure time
            #Waiting time = Time until job just before current job has departed
                worker_free_time = start_times[pending_jobs[-1]][worker] + tau*(alpha*numtasks/num_workers)
                wait_time = max([0,worker_free_time - time_arr[job]])
                start_times[job][worker] += wait_time
        pending_jobs.append(job)
    pending_jobs, start_times, stop_times, done_tasks = update_all(pending_jobs,start_times,stop_times,done_tasks,alpha,decthresh,float('Inf')) #Updating at t=Infinity
    latency_avg = np.mean(stop_times - time_arr) #Avg latency for all jobs
    return latency_avg

def queueing_main(rates_list, num_iters, num_jobs, num_workers, alpha, decthresh):
    latency_vect = np.zeros(len(rates_list))
    for (ind,rate) in enumerate(rates_list):
        print (rate)
        latency_avg = 0
        for iter in range(num_iters):
            print (iter)
            time_arr = np.cumsum(np.random.exponential(scale=1/rate,size=num_jobs))
            decthresh_sample = np.random.choice(decthresh,size=num_jobs)
            latency_avg+= lt_queue(time_arr,num_workers,alpha,decthresh_sample)
        latency_vect[ind]= latency_avg/num_iters
    return latency_vect
                
num_jobs = 100
num_iters = 10
num_workers = 10
if delay_type == 'par':
    rates_list = np.arange(start=0.1,stop=0.3, step=0.03).tolist()
elif delay_type == 'exp':
    rates_list = np.arange(start=0.1,stop=0.55, step=0.05).tolist()
decthresh = np.load('encnum_10000.npy')
# decthresh = decthresh.reshape((num_iters,num_jobs))
# #LT (2.0)
alpha = 2.0
latency_lt = queueing_main(rates_list, num_iters, num_jobs, num_workers, alpha, decthresh)
np.save(delay_type+'/latency_lt_2.npy',latency_lt)
# plt.plot(rates_list,latency_lt)
# plt.show()
# #LT (inf)
# alpha = float('inf')
# latency_lt = queueing_main(rates_list, num_iters, num_jobs, num_workers, alpha, decthresh)
# np.save('latency_lt_inf.npy',latency_lt)
# # plt.plot(rates_list,latency_lt)
# plt.show()
#DLB
alpha = float('inf')
latency_lt = queueing_main(rates_list, num_iters, num_jobs, num_workers, alpha, 10000*np.ones(decthresh.shape))
np.save(delay_type+'/latency_dlb.npy',latency_lt)