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
linestyle=[':','--','-.','-']
linestyle_ctr=0
#plt.subplot(131)

#Latency Tail(Rep)
latency_rep=np.load('latency_rep.npy')
numreps_rep=np.load('numreps_rep.npy')
avgplot[keys[0]]['params']=numreps_rep
for i in range(latency_rep.shape[0]):
    avgplot[keys[0]]['latency'].append(np.mean(latency_rep[i,:]))
for i in range(len(numreps_rep)):
    if i==0:
        labelstr='Uncoded ('+keys[0]+' = '+str(numreps_rep[i])+')'
        currentcolor = 'lightskyblue'
    else:
        labelstr='Rep ('+keys[0]+' = '+str(numreps_rep[i])+')'
        currentcolor = 'blue'
    x,y=gettail(latency_rep[i,:])
    plt.plot(x,y,color=currentcolor,linestyle=linestyle[linestyle_ctr],label=labelstr)
linestyle_ctr+=1

#Latency_Tail(MDS)
latency_mds=np.load('latency_mds.npy')
mdsnum_mds=np.load('mdsnum_mds.npy')
# print (mdsnum_mds)
avgplot[keys[1]]['params']=mdsnum_mds
for i in range(latency_mds.shape[0]):
    avgplot[keys[1]]['latency'].append(np.mean(latency_mds[i,:]))
j=0
for i in range(len(mdsnum_mds)):
    # if mdsnum_mds[i]==8:
    #     labelstr='MDS ('+keys[1]+' = '+str(mdsnum_mds[i])+')'
    #     x,y=gettail(latency_mds[i,:])
    #     #plt.plot(x,y,color='green',marker='s',linestyle=linestyle[j],label=labelstr)
    #     j+=1
    if mdsnum_mds[i]==8:
        labelstr='MDS ('+keys[1]+' = '+str(mdsnum_mds[i])+')'
        x,y=gettail(latency_mds[i,:])
        plt.plot(x,y,color='green',linestyle=linestyle[linestyle_ctr],label=labelstr)
        linestyle_ctr+=1

#Latency_Tail(LT)
latency_lt=np.load('latency_lt.npy')
alpha_lt=np.load('alpha_lt.npy')
avgplot[keys[2]]['params']=alpha_lt
for i in range(latency_lt.shape[0]):
    avgplot[keys[2]]['latency'].append(np.mean(latency_lt[i,:]))
j=0
for i in range(len(alpha_lt)):
    if alpha_lt[i]==1.25:
        labelstr='LT ('+keys[2]+' = '+str(alpha_lt[i])+')'
        x,y=gettail(latency_lt[i,:])
        #plt.plot(x,y,color='black',marker='^',linestyle=linestyle[j],label=labelstr)
        j+=1
    elif alpha_lt[i]==2.0:
        labelstr='LT ('+keys[2]+' = '+str(alpha_lt[i])+')'
        x,y=gettail(latency_lt[i,:])
        plt.plot(x,y,color='orange',linestyle=linestyle[linestyle_ctr],label=labelstr)
        linestyle_ctr+=1

# Latency_Tail(DLB)
latency_dlb=np.load('latency_dlb.npy')
avglatency_dlb=np.mean(latency_dlb)
labelstr='Ideal'
x,y=gettail(latency_dlb)
plt.plot(x,y,color='black',marker='+',mew=2,ms=10,linestyle='None',label=labelstr)

#plt.title('Latency tail')
plt.ylabel(r'$\Pr (T>t)$')
plt.xlabel('$t$')
plt.legend(loc='upper right')
plt.savefig('Plot6_runtimetail.pdf')

plt.figure()
#plt.subplot(132)

#Comp_Tail (DLB)
comp_dlb=np.load('comp_dlb.npy')
avgcomp_dlb=np.mean(comp_dlb)
labelstr='Ideal'
x,y=gettail(comp_dlb)
plt.plot(x,y,color='black',marker='+',mew=2,ms=10,linestyle='--',label=labelstr)

linestyle_ctr=0
#Comp_Tail(Rep)
comp_rep=np.load('comp_rep.npy')
numreps_rep=np.load('numreps_rep.npy')
avgplot[keys[0]]['params']=numreps_rep
for i in range(comp_rep.shape[0]):
    avgplot[keys[0]]['comp'].append(np.mean(comp_rep[i,:]))
for i in range(len(numreps_rep)):
    if i==0:
        labelstr='Uncoded ('+keys[0]+' = '+str(numreps_rep[i])+')'
        currentcolor = 'lightskyblue'
    else:
        labelstr='Rep ('+keys[0]+' = '+str(numreps_rep[i])+')'
        currentcolor = 'blue'
    x,y=gettail(comp_rep[i,:])
    plt.plot(x,y,color=currentcolor,linestyle=linestyle[linestyle_ctr],label=labelstr)
linestyle_ctr+=1

#Comp_Tail(MDS)
comp_mds=np.load('comp_mds.npy')
mdsnum_mds=np.load('mdsnum_mds.npy')
avgplot[keys[1]]['params']=mdsnum_mds
for i in range(comp_mds.shape[0]):
    avgplot[keys[1]]['comp'].append(np.mean(comp_mds[i,:]))
j=0
for i in range(len(mdsnum_mds)):
    # if mdsnum_mds[i]==8:
    #     labelstr='MDS ('+keys[1]+' = '+str(mdsnum_mds[i])+')'
    #     x,y=gettail(comp_mds[i,:])
    #     #plt.plot(x,y,color='green',marker='s',linestyle=linestyle[j],label=labelstr)
    #     j+=1
    if mdsnum_mds[i]==8:
        labelstr='MDS ('+keys[1]+' = '+str(mdsnum_mds[i])+')'
        x,y=gettail(comp_mds[i,:])
        plt.plot(x,y,color='green',linestyle=linestyle[linestyle_ctr],label=labelstr)
        linestyle_ctr+=1

#Comp_Tail(LT)
comp_lt=np.load('comp_lt.npy')
alpha_lt=np.load('alpha_lt.npy')
avgplot[keys[2]]['params']=alpha_lt
for i in range(comp_lt.shape[0]):
    avgplot[keys[2]]['comp'].append(np.mean(comp_lt[i,:]))
j=0
for i in range(len(alpha_lt)):
    if alpha_lt[i]==1.25:
        labelstr='LT ('+keys[2]+' = '+str(alpha_lt[i])+')'
        x,y=gettail(comp_lt[i,:])
        #plt.plot(x,y,color='black',marker='^',linestyle=linestyle[j],label=labelstr)
        j+=1
    elif alpha_lt[i]==2.0:
        labelstr='LT ('+keys[2]+' = '+str(alpha_lt[i])+')'
        x,y=gettail(comp_lt[i,:])
        plt.plot(x,y,color='orange',linestyle=linestyle[linestyle_ctr],label=labelstr)
        linestyle_ctr+=1

#plt.title('Computations tail')
plt.ylabel(r'$\Pr (C>c)$')
plt.xlabel('$c$')
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,3,4,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right')

# plt.legend(loc='upper right')
plt.savefig('Plot7_comptail.pdf')

plt.figure()
#plt.subplot(133)
#Avg Latency vs Comp
#linestyle_ctr=0
#linestyle_new='-.'
color=['blue','green','orange']
marker=['o','^','s']
exptnames=['Rep','MDS','LT']
fy_lt=[0.95,1.02,0.95]
fx_lt=[0.98,1,0.98]
for expt_type in range(3):
    num_pts=len(avgplot[keys[expt_type]]['params'])
    x=np.asarray(avgplot[keys[expt_type]]['latency'])
    y=np.asarray(avgplot[keys[expt_type]]['comp'])/10000
    plt.plot(x,y,color=color[expt_type],marker=marker[expt_type],linestyle=linestyle[expt_type+1],label=exptnames[expt_type])
    for pt in range(num_pts):
        if expt_type==0:
            label=keys[expt_type]+'='+str(avgplot[keys[expt_type]]['params'][pt])
            if avgplot[keys[expt_type]]['params'][pt]==1:
                fx,fy=1,1.01
                label='Uncoded'
            else:
                fx,fy=1.01,1
        elif expt_type==1:
            label=keys[expt_type]+'='+str(avgplot[keys[expt_type]]['params'][pt])
            if avgplot[keys[expt_type]]['params'][pt]==10:
                fx,fy=0.97,0.96
                label ='(r=1/k=10)'
            else:
                fx,fy=1.01,1
        else:
        #For LT Codes
            label=keys[expt_type]+'='+str(avgplot[keys[expt_type]]['params'][pt])
            fx,fy=fx_lt[pt],fy_lt[pt]
        plt.annotate(label,xy=(x[pt],y[pt]),xytext=(fx*x[pt],fy*y[pt]))

x=avglatency_dlb
y=avgcomp_dlb/10000
f=1.05
plt.plot(x,y,color='black',marker='+',label='Ideal',mew=5, ms=10)
plt.annotate('Ideal',xy=(x,y),xytext=(0.9*x,1.02*y))
plt.title('Avg. Computations v/s Latency')
plt.ylabel(r'$E[C]/m$')
plt.xlabel(r'$E[T]$')
plt.legend(loc='upper left')
#plt.tight_layout()
plt.savefig('Plot8_runtimevscomp.pdf')
#plt.savefig('Plot2_Simulations.pdf')

