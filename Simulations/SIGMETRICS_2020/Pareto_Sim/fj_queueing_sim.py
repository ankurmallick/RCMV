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

def get_proc_time(num_fork):
    shift = tau*(numtasks/num_fork)
    if delay_type=='exp':
        return np.random.exponential(scale=mu)+shift
    elif delay_type == 'par':
        # mu=1 #Shape parameter
        return 1*(1.0+np.random.pareto(a=mu))+shift #Check this from slack messages

def update_one(past_job,t0,done_workers,pending_jobs_current,time_dep,stop_times,group_size,num_fork):
    num_jobs, num_workers = time_dep.shape
    num_groups = int(num_workers/group_size)
    done_groups = {}
    time_dep_old = time_dep[past_job,:]
    for worker in range(num_workers):
        group_num = int(worker/group_size)
        if done_workers[past_job][worker] == 1 and not(group_num in done_groups):
            group_start = group_num*group_size
            done_workers[past_job][group_start:group_start+group_size] = 1
            #Update the departure time of all workers in that group to the minimum departure time
            t_set = np.amin(time_dep[past_job][group_start:group_start+group_size])
            time_dep[past_job,group_start:group_start+group_size] = set_time(time_dep[past_job][group_start:group_start+group_size],t_set)
            done_groups[group_num] = 1
    if len(done_groups)>=num_fork:
        #job is done (remove from pending_jobs_current)
        pending_jobs_current.remove(past_job)
        #Get done times for each group
        done_times = []
        for group_num in done_groups:
            group_start = group_num*group_size
            done_times.append(np.amin(time_dep[past_job][group_start:group_start+group_size]))
        done_times_sorted = sorted(done_times)
        t_set = done_times_sorted[num_fork - 1]
        stop_times[past_job] = t_set
        #Set all remaining times to done time for group k
        for group_num in range(num_groups):
            group_start = group_num*group_size
            time_dep[past_job,group_start:group_start+group_size] = set_time(time_dep[past_job][group_start:group_start+group_size],t_set)
    time_dep_diff = time_dep_old - time_dep[past_job,:]
    for pending_job in range(past_job+1,num_jobs):
        if np.sum(time_dep[pending_job,:])>0:
            time_dep[pending_job,:] -= time_dep_diff
    return done_workers, pending_jobs_current, time_dep, stop_times

def update_all(pending_jobs,time_dep,stop_times,done_workers,group_size,num_fork,t0):
    #Updating queues at t=t0
    pending_jobs_current = pending_jobs[:] #Currently pending jobs (initial estimate)
    for past_job in pending_jobs:
        for worker in range(num_workers):
            if time_dep[past_job][worker] < t0:
                #Job has definitely been processed by worker
                done_workers[past_job][worker] = 1 
        done_workers, pending_jobs_current, time_dep, stop_times = update_one(past_job,t0,done_workers,pending_jobs_current,time_dep,stop_times,group_size,num_fork)
    pending_jobs = pending_jobs_current[:]
    return pending_jobs, time_dep, stop_times, done_workers

def fork_join_queue(time_arr,num_workers,group_size,num_fork):
    num_jobs = time_arr.shape[0]
    pending_jobs = [] #List of currently pending jobs
    done_workers = {} #For each job, maintain an array of workers that are done
    time_dep = np.zeros((num_jobs,num_workers)) #Current forecast of departure time instants of each pending job from each worker
    stop_times = np.zeros(num_jobs)
    #Simulating fork join queue
    for job in range(num_jobs):
        pending_jobs, time_dep, stop_times, done_workers = update_all(pending_jobs,time_dep,stop_times,done_workers,group_size,num_fork,time_arr[job]) #Update data at current time
        done_workers[job] = np.zeros(num_workers)
        for worker in range(num_workers):
            job_proc_time = get_proc_time(num_fork) #Job processing time at a worker
            time_dep[job][worker] = job_proc_time+time_arr[job]
            if len(pending_jobs)>0:
                #If worker queue is not empty add waiting time to departure time
                 #Waiting time = Time until job just before current job has departed
                time_dep[job][worker] += max([0,time_dep[pending_jobs[-1]][worker] - time_arr[job]])
        pending_jobs.append(job)
    pending_jobs, time_dep, stop_times, done_workers = update_all(pending_jobs,time_dep,stop_times,done_workers,group_size,num_fork,float('Inf')) #Updating at t=Infinity
    latency_avg = np.mean(stop_times - time_arr) #Avg latency for all jobs
    return latency_avg

def queueing_main(rates_list, num_iters, num_jobs, num_workers, group_size, num_fork):
    latency_vect = np.zeros(len(rates_list))
    for (ind,rate) in enumerate(rates_list):
        print (rate)
        latency_avg = 0
        for iter in range(num_iters):
            time_arr = np.cumsum(np.random.exponential(scale=1/rate,size=num_jobs))
            latency_avg+= fork_join_queue(time_arr,num_workers,group_size,num_fork)
        latency_vect[ind]= latency_avg/num_iters
    return latency_vect
                
num_jobs = 100
num_iters = 10
num_workers = 10
if delay_type == 'par':
    rates_list = np.arange(start=0.1,stop=0.3, step=0.03).tolist()
elif delay_type == 'exp':
    rates_list = np.arange(start=0.1,stop=0.55, step=0.05).tolist()
latencymat = np.zeros((3,len(rates_list)))
#Call queueing repeatedly to get expected latency for different schemes and generate plots of E[T] v/s lambda
#Uncoded
numrep = 1
mdsnum = int(num_workers/numrep)
latency = queueing_main(rates_list, num_iters, num_jobs, num_workers, numrep, mdsnum)
np.save(delay_type+'/latency_rep_1.npy',latency)
#2-Rep
numrep = 2
mdsnum = int(num_workers/numrep)
latency = queueing_main(rates_list, num_iters, num_jobs, num_workers, numrep, mdsnum)
np.save(delay_type+'/latency_rep_2.npy',latency)
#MDS
numrep = 1
mdsnum = 8
latency = queueing_main(rates_list, num_iters, num_jobs, num_workers, numrep, mdsnum)
np.save(delay_type+'/latency_mds.npy',latency)

# color=['blue','blue','green','black']
# marker=['o','o','^','s']
# exptnames=['Uncoded','Rep','MDS','LT']
# scheme_num=['1','2','5','2.0']
# keys=['r','r','k',r'$\alpha$']
# linestyle=[':','--','-.','-']
# plt.figure()
# for expt_type in range(3):
#     x= rates_list
#     y = latencymat[expt_type,:]
#     labelstr=exptnames[expt_type]+' ('+keys[expt_type]+' = '+scheme_num[expt_type]+')'
#     plt.plot(x,y,color=color[expt_type],marker=marker[expt_type],linestyle=linestyle[expt_type],label=exptnames[expt_type])
# plt.ylabel(r'$E[T]$')
# plt.xlabel(r'$\lambda$')
# plt.legend(loc='upper right')
# #plt.show()
# if delay_type == 'par':
#     plt.savefig('Plot5_QueueingPar.pdf')
# elif delay_type == 'exp':
#     plt.savefig('Plot5_QueueingExp.pdf')
