The excel sheet contains the runtimes given in the paper: Speeding up K-Means Algorithm by GPU.pdf  by Li et al.

The table numbers 1 to 9 contain the data from table 1-9 given in the paper.

Columns:
n   : number of points
k   : number of clusters
d   : number of dimensions
iter: total iterations for which runtime is given.

Rest columns are in threee goups of five columns each:
1st group   : Analysis of runtime for finding the closest centroid
calc ops    : Minimum no. floating point operations required to find the closest centroid for all the points.
mem ops     : Minimum no. of memory operations required to find the closest centroid for all the points. Each memory operation is of 4 bytes as all accesses are ints and floats.
Find (ms)   : Time taken in millisec to finish the kernel
GFLOP/sec   : calc ops / (Find (ms) * 1000000) . Minimum GFLOP/s for the kernel.
GB/s        : (mem ops * 4) / (Find (ms) * 1000000). Minimum global memory bandwidth for the kernel.

2nd group   : Analysis of runtime for computing new centroids
calc ops    : Minimum no. floating point operations required to compute new centroids irrespective of the technique used in the paper.
mem ops     : Minimum no. of memory operations required to compute new centroids irrespective of the technique used in the paper. Each memory operation is of 4 bytes as all accesses are ints and floats.
Compute(ms) : Time taken in millisec to finish the kernel
GFLOP/sec   : calc ops / (Compute (ms) * 1000000) . Minimum GFLOP/s for the kernel.
GB/s        : (mem ops * 4) / (Compute (ms) * 1000000). Minimum global memory bandwidth for the kernel.

3rd group   : Analysis of runtime for the complete kmeans
calc ops    : Sum of calc ops of above two groups.
mem ops     : Sum of mem ops of above two groups.
Total(ms)   : Total time taken in millisec
GFLOP/sec   : calc ops / (Total (ms) * 1000000) . Minimum GFLOP/s for kmeans.
GB/s        : (mem ops * 4) / (Total (ms) * 1000000). Minimum global memory bandwidth for the kmeans.


Note: The time fields have been left empty wherever there values were not specified in the respective tables in the paper. As a result crres. GFlop/s and GB/s fileds are also empty.


In cloumn I one value is 1024 GFlop/s which does not seem theoritically possible.
There are numerous other values with more than 730 GFlop/s, which as per our benchmarking does not seem achievable even for pure register-based operations.
