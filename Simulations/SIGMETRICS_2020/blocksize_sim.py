import numpy as np
import matplotlib.pyplot as plt
import time

#Attempting to find the optimal block size for 1000 x 1000 by 1000 x 1 multiplication
# RAM = 8 GB
# Space for 1 number = 8 bytes (Double)
# We can store 10^9 numbers (1000 x 1000 x 1000)

def get_factors(num):
    #Returns factors of a number
    fact_list = []
    for fact in range(1,num+1):
        if num%fact==0:
            fact_list.append(fact)
    return fact_list

mat = np.random.normal(size = (1000,1000))
vect = np.random.normal(size = 1000)

num_iters = 100 #number of MonteCarlo Simulations

blocksizes = get_factors(1000) #Available block sizes
print (blocksizes)

times = []
prod = np.zeros(1000)
for blocksize in blocksizes:
    print ('Blocksize = '+str(blocksize))
    num_blocks = int(1000/blocksize)
    t_current = 0
    for iter in range(num_iters):
        t0 = time.clock()
        block_start = 0
        for block in range(num_blocks):
            prod[block_start:block_start+blocksize]= mat[block_start:block_start+blocksize,:].dot(vect)
            block_start+=blocksize
        tf = time.clock()
        t_current += tf-t0
    times.append(t_current/num_iters)

blocksizes_str = [str(blocksize) for blocksize in blocksizes]
plt.bar(blocksizes_str,times)
plt.xlabel('Blocksizes')
plt.ylabel('Avg. Computation Time')
plt.title('Comparison of Different Blocksizes')
plt.savefig('Blocksize_comp.pdf')