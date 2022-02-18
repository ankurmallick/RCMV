# RCMV

Code for the ACM SIGMETRICS 2020 paper 'Rateless Codes for Near-Perfect Load Balancing in Distributed Matrix-Vector Multiplication' (https://arxiv.org/abs/1804.10331)  

We consider the following coded distributed computing schemes for computing b = Ax (A has 'm' rows):  

1. Uncoded ('Unc'): Each worker computes a distinct submatrix-vector product. Results are aggregated at the central node.

2. r-Replication ('Rep'): Each submatrix is replicated at 'r' distinct workers. Results of the fastest replicas for each submatrix are aggregated at the central node.

3. MDS ('MDS'): A is encoded using a (p,k) MDS code. Workers compute encoded submatrix-vector products. Rsults of the fastest k workers are collected and decoded by the central node to give b = Ax

4. LT ('LT'): Our Solution. A is encoded using an LT code. Workers compute encoded row-vector products. The fastest (approximately) 'm' row vector products are collected and decoded by the central node to give b = Ax.

In this repository, workers correspond to separate processes and the distributed computation is performed using Python's Multiprocessing library. Run Multiprocessing_expts.ipynb to see the latency and number of computations performed by each of the aforementioned schemes.  

The repository also includes the following additional files:

1. encoders.py: Contains the encoding functions for the different coded computing schemes (converts input matrix to its encoded form)

2. time_analytics.py: Contains functions to process the results from each worker and calculate the latency and number of computations for each scheme (converts worker results to decoder inputs)

3. decoders.py: Contains the decoding functions for the different coded computing schemes (converts encoded results to the final/decoded result)

