import numpy as np
import matplotlib.pyplot as plt
plt.style.use('publication')
#Event-Driven Simulations for delay model

def setupdelay(mu,num_workers,delaytype):
    if delaytype == 'exp':
        return np.random.exponential(scale=mu,size=num_workers) #Small mu means smaller E[T]
    elif delaytype == 'par':
        # mu=1 #Shape parameter
        return 1*(1.0+np.random.pareto(a=mu,size=num_workers)) #Check this from slack messages

class delaysim:
    def __init__(self,mu,tau,num_workers,numtasks,delaytype):
        self.mu = mu
        self.tau = tau
        self.num_workers = num_workers
        self.numtasks = numtasks #Total number of tasks(dot-products)
        self.delaytype = delaytype

    def rep(self,numreps):
        #Latency and number of computations for a single instance of numreps-Replication
        maxtasks_worker=(self.numtasks/self.num_workers)*numreps
        latency=0
        comp=0
        numgroups=int(self.num_workers/numreps)
        grouptimes=np.zeros((numgroups,numreps))
        groupcomps=np.zeros((numgroups,numreps))
        mingrouptimes=np.zeros(numgroups)
        for group in range(numgroups):
            grouptimes[group,:]=setupdelay(self.mu,numreps,self.delaytype)
            mingrouptimes[group]=np.amin(grouptimes[group,:])
        latency=np.amax(mingrouptimes)+self.tau*maxtasks_worker #Slowest among top workers of each group
        for group in range(numgroups):
            for worker in range(numreps):
                C=np.floor((latency-grouptimes[group][worker])/self.tau)
                if C<0:
                    C=0
                elif C>maxtasks_worker or numreps==1:
                    C=maxtasks_worker
                groupcomps[group][worker]=C
       # print groupcomps
        comp=np.sum(groupcomps)
        return latency,comp

    def mds(self,mdsnum):
        #Latency and number of computations for a single instance of (numworkers,mdsnum)-MDS
        maxtasks_worker=np.ceil((self.numtasks/mdsnum))
        latency=0
        comp=0
        workertimes=setupdelay(self.mu,self.num_workers,self.delaytype)
        workertimes_order=np.argsort(workertimes)
        latency=workertimes[workertimes_order[mdsnum-1]]+self.tau*maxtasks_worker
        for worker in range(self.num_workers):
            C=np.floor((latency-workertimes[worker])/self.tau)
            if C<0:
                C=0
            elif C>maxtasks_worker or mdsnum==10:
                C=maxtasks_worker
            comp+=C
        return latency,comp

    def lt(self,alpha,decthresh):
        #Latency and number of computations for a single instance of (alpha*numtasks,decthresh) LT
        #DLB is LT with alpha=inf, decthresh=numtasks
        maxtasks_worker=alpha*self.numtasks/self.num_workers
        comp=int(decthresh) #Read decoding threshold from file
        latency=0
        setuptime=setupdelay(self.mu,self.num_workers,self.delaytype)
        currenttime=setuptime + self.tau #Time taken by each worker to complete 1st computation
        workercount=np.zeros(self.num_workers)
        for counter in range(comp):
            #Each iteration corresponds to 1 computation being completed
            minpos=np.argmin(currenttime)
            latency+=currenttime[minpos]
            currenttime=currenttime-currenttime[minpos]
            currenttime[minpos]=self.tau
            workercount[minpos]+=1
            if workercount[minpos]==maxtasks_worker:
                currenttime=np.delete(currenttime,minpos)
                workercount=np.delete(workercount,minpos)
        return latency,comp

if __name__=='__main__':
    num_mc=500 #Number of Monte-Carlo Simulations
    mu = 3.0
    tau=mu/1000
    num_workers=10
    numtasks=10000
    delaytype = 'par'
    DS=delaysim(mu,tau,num_workers,numtasks,delaytype) #Delaysim object
    #Replication
    print ("Replication Strategy: ")
    numreps=np.asarray([1,2])
    latency=np.zeros((numreps.size,num_mc))
    comp=np.zeros((numreps.size,num_mc))
    for paramnum in range(numreps.size):
        print ('r = '+str(numreps[paramnum]))
        for exptnum in range(num_mc):
            latency[paramnum,exptnum],comp[paramnum,exptnum]=DS.rep(numreps[paramnum])
        print ('Average Latency: '+str(np.mean(latency[paramnum,:])))
        print ('Average Computations: '+str(np.mean(comp[paramnum,:])))
    np.save('latency_rep.npy',latency)
    np.save('comp_rep.npy',comp)
    np.save('numreps_rep.npy',numreps)
    #MDS
    print ("MDS-Coded Strategy: ")
    mdsnum=np.asarray([10,9,8,7,6,5])
    latency=np.zeros((mdsnum.size,num_mc))
    comp=np.zeros((mdsnum.size,num_mc))
    for paramnum in range(mdsnum.size):
        print ('k = '+str(mdsnum[paramnum]))
        for exptnum in range(num_mc):
            latency[paramnum,exptnum],comp[paramnum,exptnum]=DS.mds(int(mdsnum[paramnum]))
        print ('Average Latency: '+str(np.mean(latency[paramnum,:])))
        print ('Average Computations: '+str(np.mean(comp[paramnum,:])))
    np.save('latency_mds.npy',latency)
    np.save('comp_mds.npy',comp)
    np.save('mdsnum_mds.npy',mdsnum)
    #LT
    print ("LT-Coded Strategy: ")
    alpha=np.asarray([1.25,1.5,2.0])
    latency=np.zeros((alpha.size,num_mc))
    comp=np.zeros((alpha.size,num_mc))
    decthresh=np.load('encnum_10000.npy')
    for paramnum in range(alpha.size):
        print ('alpha = '+str(alpha[paramnum]))
        for exptnum in range(num_mc):
            latency[paramnum,exptnum],comp[paramnum,exptnum]=DS.lt(alpha[paramnum],int(decthresh[exptnum]))
        print ('Average Latency: '+str(np.mean(latency[paramnum,:])))
        print ('Average Computations: '+str(np.mean(comp[paramnum,:])))
    np.save('latency_lt.npy',latency)
    np.save('comp_lt.npy',comp)
    np.save('alpha_lt.npy',alpha)
    print ("Dynamic Load Balancing: ")
    latency=np.zeros(num_mc)
    comp=np.zeros(num_mc)
    for exptnum in range(num_mc):
        latency[exptnum],comp[exptnum]=DS.lt(float('Inf'),DS.numtasks)
    print ('Average Latency: '+str(np.mean(latency)))
    print ('Average Computations: '+str(np.mean(comp)))
    np.save('latency_dlb.npy',latency)
    np.save('comp_dlb.npy',comp)
