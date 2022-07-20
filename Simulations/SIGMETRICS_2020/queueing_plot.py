import numpy as np
import matplotlib.pyplot as plt
plt.style.use('publication')

#Loading data
delay_type = 'exp'
if delay_type == 'par':
    rates_list = np.arange(start=0.1,stop=0.3, step=0.03).tolist()
elif delay_type == 'exp':
    rates_list = np.arange(start=0.1,stop=0.55, step=0.05).tolist()
latencymat = np.zeros((5,len(rates_list)))
latencymat[0,:] = np.load(delay_type+'/latency_rep_1.npy')
latencymat[1,:] = np.load(delay_type+'/latency_rep_2.npy')
latencymat[2,:] = np.load(delay_type+'/latency_mds.npy')
latencymat[3,:] = np.load(delay_type+'/latency_lt_2.npy')
latencymat[4,:] = np.load(delay_type+'/latency_dlb.npy')

#Plotting
color=['lightskyblue','blue','green','orange','black']
marker=['o','o','^','s','None']
exptnames=['Uncoded','Rep','MDS','LT','Ideal']
scheme_num=['1','2','8','2.0',r'$\infty$']
keys=['r','r','k',r'$\alpha$',r'$\alpha$']
linestyle=[':','--','-.','-','-']
plt.figure()
for expt_type in range(5):
    x= rates_list
    y = latencymat[expt_type,:]
    if expt_type == 4:
        labelstr=exptnames[expt_type]#+' ('+keys[expt_type]+r'$\rightarrow$'+scheme_num[expt_type]+')'
    else:
        labelstr=exptnames[expt_type]+' ('+keys[expt_type]+' = '+scheme_num[expt_type]+')'
    plt.plot(x,y,color=color[expt_type],marker=marker[expt_type],linestyle=linestyle[expt_type],label=labelstr)
plt.ylabel(r'$E[Z]$')
plt.xlabel(r'$\lambda$')
plt.legend(loc='upper left')
#plt.show()
if delay_type == 'par':
    plt.savefig('Plot9_QueueingPar.pdf')
elif delay_type == 'exp':
    plt.savefig('Plot9_QueueingExp.pdf')

