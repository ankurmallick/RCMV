import numpy as np
import matplotlib.pyplot as plt
from encoders import mds_encoder, lt_encoder, syslt_pardist
from scipy.stats import powerlaw
from plotting_helpers import get_tail_plots
from utils import setupdelay, get_degree_dist, nnz_task_allocator
from SpMV import SpMV
plt.style.use('publication')
#Event-Driven Simulations for delay model

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
    Exp_delay_info = {'type':'Exp','param':1.0}
    Par_delay_info = {'type':'Par','param':1.0}
    ER_mat_info = {'type':'ER','param':0.01}
    PL_mat_info = {'type':'PL','param':2.5}
    DS=SpMV(tau, Exp_delay_info, PL_mat_info, rows, cols, num_workers, mdsnum, c, delta, alpha) #Delaysim object
    latencytail_inputs = []
    #Uncoded
    print ("Uncoded: ")
    numreps = 1
    latency=np.zeros(num_mc)
    for exptnum in range(num_mc):
        latency[exptnum]=DS.rep(numreps)
    latencytail_inputs.append(latency) 
    print ('Average Latency: '+str(np.mean(latency)))
    legends = ['Uncoded', 'Balanced', 'MDS', 'LT', 'Sys LT']
    figname = 'ISIT_Uncoded.pdf'
    get_tail_plots(latencytail_inputs,legends,'lat',figname)
    #Balanced
    print ("Balanced: ")
    numreps = 1
    latency=np.zeros(num_mc)
    for exptnum in range(num_mc):
        latency[exptnum]=DS.rep_ord(numreps)
    latencytail_inputs.append(latency) 
    print ('Average Latency: '+str(np.mean(latency)))
    legends = ['Uncoded', 'Balanced', 'MDS', 'LT', 'Sys LT']
    figname = 'ISIT_Balanced.pdf'
    get_tail_plots(latencytail_inputs,legends,'lat',figname)
    # MDS
    print ("MDS Strategy: ")
    mdsnum = 8
    latency=np.zeros(num_mc)
    print ('k = '+str(mdsnum))
    for exptnum in range(num_mc):
        latency[exptnum]=DS.mds(int(mdsnum))
    latencytail_inputs.append(latency)
    print ('Average Latency: '+str(np.mean(latency)))
    legends = ['Uncoded', 'Balanced', 'MDS', 'LT', 'Sys LT']
    figname = 'ISIT_MDS.pdf'
    get_tail_plots(latencytail_inputs,legends,'lat',figname)
    #LT
    print ("LT-Coded Strategy Ord: ")
    # alpha=np.asarray([1.25,1.5,2.0])
    alpha = 2.0
    latency=np.zeros(num_mc)
    decthresh=np.load('encnum.npy')
    # for paramnum in range(alpha.size):
    print ('alpha = '+str(alpha))
    for exptnum in range(num_mc):
        latency[exptnum]=DS.lt_ord(alpha,int(decthresh[exptnum]))
    latencytail_inputs.append(latency) #LT (balanced)
    print ('Average Latency: '+str(np.mean(latency)))
    legends = ['Uncoded', 'Balanced', 'MDS', 'LT', 'Sys LT']
    figname = 'ISIT_LT.pdf'
    get_tail_plots(latencytail_inputs,legends,'lat',figname)
    #Sys LT
    print ("Sys LT-Coded Strategy Ord: ")
    # alpha=np.asarray([1.25,1.5,2.0])
    alpha = 2.0
    latency=np.zeros(num_mc)
    decthresh=np.load('encnum.npy')
    # for paramnum in range(alpha.size):
    print ('alpha = '+str(alpha))
    for exptnum in range(num_mc):
        latency[exptnum]=DS.sys_lt_ord(alpha,int(decthresh[exptnum]))
    latencytail_inputs.append(latency) #LT (balanced)
    print ('Average Latency: '+str(np.mean(latency)))
    legends = ['Uncoded', 'Balanced', 'MDS', 'LT', 'Sys LT']
    figname = 'ISIT_SysLT.pdf'
    get_tail_plots(latencytail_inputs,legends,'lat',figname)
    
    #Latency Tail Plots
    
    