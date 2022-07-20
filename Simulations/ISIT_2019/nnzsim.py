import numpy as np
import matplotlib.pyplot as plt
from plotting_helpers import get_tail_plots
from utils import get_degree_dist
plt.style.use('publication')

#Simulating the number of non-zero elements in the original and encoded matrices

num_mc=int(500) #Number of Monte-Carlo Simulations
num_workers=10
rows = 10000
cols = 10000
c = 0.03
delta = 0.5
mdsnum = 8
alpha = 2.0
ER_mat_info = {'type':'ER','param':0.01}
PL_mat_info = {'type':'PL','param':2.5}
degree_dist, degree_dist_mds, degree_dist_lt, degree_dist_syslt=get_degree_dist(PL_mat_info, rows, cols, num_workers, mdsnum, c, delta, alpha)
legends = ['Original', 'LT', 'Sys LT']
nnztail_inputs = [degree_dist,degree_dist_lt,degree_dist_syslt]
get_tail_plots(nnztail_inputs,legends,'nnz','Plot1_NNZtail.pdf')