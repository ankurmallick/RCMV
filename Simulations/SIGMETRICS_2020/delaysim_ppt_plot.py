import numpy as np
import matplotlib.pyplot as plt
plt.style.use('publication')

def gettail(data):
    #Returns x and y coordinates for tail plot of data
    hist,bin_edges=np.histogram(data,bins=100)
    hist_fl=[float(h) for h in hist]
    tail=1-np.cumsum(hist)/np.sum(hist_fl)
    return bin_edges[:-1],tail

avgplot={} #Dictionary for avg latency v/s comp plot
keys=['r','k',r'$\alpha$']
for key in keys:
    avgplot[key]={'latency':[],'comp':[],'param':[]}

plt.figure()
linestyle=[':','--','-.','-','-']
linestyle_ctr=0
#plt.subplot(131)

# Latency_Tail(DLB)
latency_dlb=np.load('latency_dlb.npy')
labelstr='Ideal'
x,y=gettail(latency_dlb)
plt.plot(x,y,color='green',marker='+',linestyle=linestyle[4],label=labelstr)

#Latency Tail(Rep)
latency_rep=np.load('latency_rep.npy')
numreps_rep=np.load('numreps_rep.npy')
for i in range(len(numreps_rep)):
    if i==0:
        labelstr='Uncoded'
        currentcolor = 'orange'
    else:
        labelstr='Rep'
        currentcolor = 'gold'
    x,y=gettail(latency_rep[i,:])
    plt.plot(x,y,color=currentcolor,linestyle=linestyle[linestyle_ctr],label=labelstr)
    linestyle_ctr+=1
    # break

#Latency_Tail(MDS)
latency_mds=np.load('latency_mds.npy')
mdsnum_mds=np.load('mdsnum_mds.npy')
j=0
for i in range(len(mdsnum_mds)):
    # if mdsnum_mds[i]==8:
    #     labelstr='MDS ('+keys[1]+' = '+str(mdsnum_mds[i])+')'
    #     x,y=gettail(latency_mds[i,:])
    #     #plt.plot(x,y,color='magenta',marker='s',linestyle=linestyle[j],label=labelstr)
    #     j+=1
    if mdsnum_mds[i]==8:
        labelstr='MDS'
        x,y=gettail(latency_mds[i,:])
        plt.plot(x,y,color='magenta',linestyle=linestyle[linestyle_ctr],label=labelstr)
        linestyle_ctr+=1

#Latency_Tail(LT)
latency_lt=np.load('latency_lt.npy')
alpha_lt=np.load('alpha_lt.npy')
avgplot[keys[2]]['params']=alpha_lt
j=0
for i in range(len(alpha_lt)):
    if alpha_lt[i]==1.25:
        labelstr='LT ('+keys[2]+' = '+str(alpha_lt[i])+')'
        x,y=gettail(latency_lt[i,:])
        #plt.plot(x,y,color='blue',marker='^',linestyle=linestyle[j],label=labelstr)
        j+=1
    elif alpha_lt[i]==2.0:
        labelstr='LT'
        x,y=gettail(latency_lt[i,:])
        plt.plot(x,y,color='blue',linestyle=linestyle[linestyle_ctr],label=labelstr)
        linestyle_ctr+=1



#plt.title('Latency tail')
plt.ylabel(r'$\Pr (T>t)$')
plt.xlabel('$t$')
plt.legend(loc='upper right')
plt.xlim([1.0,11.5])
plt.tight_layout()
plt.savefig('PPT_plot5.pdf')



