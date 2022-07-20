import matplotlib.pyplot as plt
import numpy as np
plt.style.use('publication')

def gettail(data):
    #Returns x and y coordinates for tail plot of data
    hist,bin_edges=np.histogram(data,bins=100)
    hist_fl=[float(h) for h in hist]
    tail=1-np.cumsum(hist)/np.sum(hist_fl)
    return bin_edges[:-1],tail

encnum = np.load('encnum_1000.npy')
x,y = gettail(encnum)
plt.figure()
plt.plot(x,y,color='blue')
plt.xlabel(r"$m$")
plt.ylabel(r"$\Pr(M'>m')$")
xthresh = np.where(y<0.01)
xthresh = xthresh[0][0]
labelstr = r"$m' = $"+str(x[xthresh])
plt.plot(x[xthresh]*np.ones(y.shape[0]),y,color='red',linestyle='--',label= labelstr)
plt.legend()
# plt.savefig('Plot16_decthresh_10000.pdf')
plt.savefig('Plot_decthresh_1000.pdf')