"""
@author: phdenzel

Cython versions of some methods in GLEAM
"""
# cython: infer_types=True
import numpy as np
cimport cython


cdef inline int int_max(int a, int b):
    return a if a >= b else b

cdef inline int int_min(int a, int b):
    return a if a <= b else b

# 2D index
cdef struct idx2D_t:
    int x
    int y

# 1D to 2D index
@cython.cdivision(True)
cdef idx2D_t idx2yx(int idx, int cols):
    cdef idx2D_t* p
    p.y = idx // cols
    p.x = idx % cols
    return p

# 2D to 1D index
cdef int yx2idx(idx2D p, int cols):
    cdef int idx
    idx = p.y * cols + p.x
    return idx

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calc_psf(np.ndarray[np.float64_t, ndim=2] P_kl,
             np.ndarray[np.float64_t, ndim=2] psf_data,
             int x0, int y0,
             int Nx, int Ny,
             int window_size):
    cdef int k, l
    cdef int dx, dy
    cdef int xl, yl
    cdef int x_psf, y_psf
    cdef np.float64_t psf_val
    cdef idx2D_t* pk
    cdef idx2D_t* pl
    # write matrix elements
    for k in range(Nx*Nx):
        pk = idx2yx(k, Ny)
        for xl in range(int_max(pk.x-window_size, 0), int_min(pk.x+window_size+1, Nx)):
            pl.x = xl
            for yl in range(int_max(pk.y-window_size, 0), int_min(pk.y+window_size+1, Ny)):
                pl.y = yl
                l = yx2idx(pl, Ny)
                dx = pk.x-xl
                dy = pk.y-yl
                # calculate psf value for symmetric matrix element (l, k) = (k, l)
                x_psf = x0 + dx
                y_psf = y0 + dy
                if (0 <= psf_y < psf_data.shape[0]) and (0 <= psf_x < psf_data.shape[1]):
                    psf_val = psf_data[y_psf, x_psf]
                else:
                    psf_val = 0
                P_kl[k, l] = psf_val
                P_kl[l, k] = psf_val
    return P_kl
                
    
