import numpy as np
import matplotlib.pyplot as plt
plt.style.use('publication')

def gettail(data):
    #Returns x and y coordinates for tail plot of data
    hist,bin_edges=np.histogram(data,bins=100)
    hist_fl=[float(h) for h in hist]
    tail=1-np.cumsum(hist)/np.sum(hist_fl)
    return bin_edges[:-1],tail
    
def get_tail_plots(data,legends,plot_type,figname):
    plt.figure()
    if plot_type == 'nnz':
        plt_info = [('orange',':'),('purple','--'),('black','-.')]
    elif plot_type == 'lat':
        plt_info = [('orange',':'),('purple','--'),('green','-.'),('blue','-'),('black','^'),('black','s')]
    # linestyle=[':','--','-.','-']
    # marker=['o','^','s']
    # color=['orange','cyan','green','purple','black']
    num_elems = len(data)
    for elem in range(num_elems):
        x,y = gettail(data[elem])
        color = plt_info[elem][0]
        if color == 'black' and plot_type == 'lat':
            marker = plt_info[elem][1]
            plt.semilogx(x,y,color=color,marker=marker,label=legends[elem])
        else:
            linestyle = plt_info[elem][1]
            plt.semilogx(x,y,color=color,linestyle=linestyle,label=legends[elem])
    if plot_type == 'nnz':
        #NNZ Plot
        plt.ylabel(r'$\Pr (S>s)$')
        plt.xlabel('$s$')
        plt.legend(loc='upper right')
        plt.savefig(figname)
    elif plot_type == 'lat':
        #Latency Plot
        plt.ylabel(r'$\Pr (T>t)$')
        plt.xlabel('$t$')
        plt.legend(loc='upper right')
        plt.savefig(figname)