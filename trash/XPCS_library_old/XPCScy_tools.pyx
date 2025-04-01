import numpy as np
#from libc.stdint cimport int64_t as i64
from cython cimport boundscheck, wraparound
from cython.parallel import prange
from os import cpu_count

@boundscheck(False)
@wraparound(False)

def mean_trace_float64(double[:,::1] A):
    cdef:
        int i,j
        double[:] t = np.zeros(A.shape[0]-1, dtype=np.float64)
    for i in range(1, A.shape[0]):
        for j in range(i+1, A.shape[0]):
            t[j-i-1] += A[i,j]

    for i in range(A.shape[0]-1):
        t[i] /= A.shape[0]-i-1

    return np.array(t)


@boundscheck(False)
@wraparound(False)

def mean_trace_float64_parallel(double[:,::1] A):
    cdef:
        int i,j, idx
        double[:] t = np.zeros(A.shape[0]-1, dtype=np.float64)
        int c_count = cpu_count()

    for idx in prange(A.shape[0]*A.shape[0], nogil=True, num_threads=c_count):
        i = idx // A.shape[0]
        j = idx % A.shape[0]
        if j>=i+1:
            t[j-i-1] += A[i,j]

    for i in range(A.shape[0]-1):
        t[i] /= A.shape[0]-i-1

    return np.array(t)