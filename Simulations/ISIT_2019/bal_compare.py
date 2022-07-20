import numpy as np
import matplotlib.pyplot as plt
from encoders import mds_encoder, lt_encoder, syslt_pardist
from scipy.stats import powerlaw
from utils import setupdelay, get_degree_dist, nnz_task_allocator
from SpMV import SpMV
plt.style.use('publication')
#Event-Driven Simulations for delay model

def gettail(data):
    #Returns x and y coordinates for tail plot of data
    hist,bin_edges=np.histogram(data,bins=100)
    hist_fl=[float(h) for h in hist]
    tail=1-np.cumsum(hist)/np.sum(hist_fl)
    return bin_edges[:-1],tail
    
def get_tail_plots(data,legends,figname):
    plt.figure()
    plt_info = [('orange',':'),('orange','-'),('purple',':'),('purple','-')]
    num_elems = len(legends)
    for elem in range(num_elems):
        x,y = gettail(data[elem])
        color = plt_info[elem][0]
        linestyle = plt_info[elem][1]
        plt.semilogx(x,y,color=color,linestyle=linestyle,label=legends[elem])
    #Latency Plot
    plt.ylabel(r'$\Pr (T>t)$')
    plt.xlabel('$t$')
    plt.legend(loc='upper right')
    plt.savefig(figname)

if __name__=='__main__':
    tau=5e-7
    num_workers=10
    rows = 10000
    cols = 10000
    c = 0.03
    delta = 0.5
    mdsnum = 8
    alpha = 2.0
    num_mc=int(500) #Number of Monte-Carlo Simulations
    Exp_delay_info = {'type':'Exp','param':2.0}
    Par_delay_info = {'type':'Par','param':1.0}
    ER_mat_info = {'type':'ER','param':0.01}
    PL_mat_info = {'type':'PL','param':2.5}
    DS=SpMV(tau, Par_delay_info, PL_mat_info, rows, cols, num_workers, mdsnum, c, delta, alpha) #Delaysim object
    latencytail_inputs = []
    
    #Uncoded
    numreps = 1
    print ("Uncoded without balancing: ")
    latency=np.zeros(num_mc)
    for exptnum in range(num_mc):
            latency[exptnum]=DS.rep(numreps)
    latencytail_inputs.append(latency) 
    print ('Average Latency: '+str(np.mean(latency)))
    
    print ("Uncoded with balancing: ")
    latency=np.zeros(num_mc)
    for exptnum in range(num_mc):
            latency[exptnum]=DS.rep_ord(numreps)
    latencytail_inputs.append(latency) 
    print ('Average Latency: '+str(np.mean(latency)))
    
    #LT
    print ("LT-Coded Strategy without balancing: ")
    alpha = 2.0
    latency=np.zeros(num_mc)
    decthresh=np.load('encnum.npy')
    # for paramnum in range(alpha.size):
    print ('alpha = '+str(alpha))
    for exptnum in range(num_mc):
        latency[exptnum]=DS.lt(alpha,int(decthresh[exptnum]))
    latencytail_inputs.append(latency) #LT (unbalanced)
    print ('Average Latency: '+str(np.mean(latency)))

    print ("LT-Coded Strategy with balancing: ")
    alpha = 2.0
    latency=np.zeros(num_mc)
    decthresh=np.load('encnum.npy')
    # for paramnum in range(alpha.size):
    print ('alpha = '+str(alpha))
    for exptnum in range(num_mc):
        latency[exptnum]=DS.lt_ord(alpha,int(decthresh[exptnum]))
    latencytail_inputs.append(latency) #LT (balanced)
    print ('Average Latency: '+str(np.mean(latency)))
    
    #Latency Tail Plots
    legends = ['Uncoded', 'Uncoded (Bal)', 'LT', 'LT (Bal)']
    figname = 'Plot4_Balcompare.pdf'
    get_tail_plots(latencytail_inputs,legends,figname)
    