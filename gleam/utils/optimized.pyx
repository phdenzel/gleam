# cython: infer_types=True, language_level=2
# cython: profile=True
"""
@author: phdenzel

Cython versions of some methods in GLEAM
"""
import numpy as np
import scipy.sparse.linalg as splin
cimport numpy as np
cimport cython
from libc.math cimport abs, floor, pi, atan, log
from cymem.cymem cimport Pool


cdef inline int int_max(int a, int b):
    return a if a >= b else b

cdef inline int int_min(int a, int b):
    return a if a <= b else b

cdef inline double double_max(double a, double b):
    return a if a >= b else b

cdef inline double double_min(double a, double b):
    return a if a <= b else b


# Optimizing skyf.py methods ###################################################

# 2D index
cdef struct idx2D_t:
    int x
    int y

# theta coordinates
cdef struct theta_t:
    double x
    double y


# 1D to 2D index
@cython.cdivision(True)
cdef idx2D_t idx2yx(int idx, int cols):
    # Convert 1D index into a 2D index
    # Args (2): (idx, cols)
    cdef idx2D_t p
    # p.y = idx // cols
    # p.x = idx % cols
    p.x = idx // cols
    p.y = idx % cols
    return p


# 2D to 1D index
cdef int yx2idx(idx2D_t p, int cols):
    # Convert 2D index into a 1D index
    # Args (2): (p, cols)
    cdef int idx
    idx = p.y * cols + p.x
    return idx


# theta vector coordinates
cdef theta_t theta(int idx, int cols, idx2D_t origin, double px2arcsec):
    # Return a flattened angular position map array
    # Args (4): (idx, cols, origin, px2arcsec)
    cdef double OFFSET = 1e-12  # default hardcoded
    cdef idx2D_t p
    cdef theta_t t
    p = idx2yx(idx, cols)
    t.x = (p.x - origin.x)*px2arcsec + OFFSET
    t.y = (p.y - origin.y)*px2arcsec + OFFSET
    return t
    

# Optimizing reconsrc.py methods ###############################################
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def calc_psf(np.ndarray[np.float64_t, ndim=2] P_kl,
             np.ndarray[np.float64_t, ndim=2] psf_data,
             int x0, int y0,
             int Lx, int Ly,
             int window_size):
    # Fill the input matrix array using the PSF image data
    # Args (7): (P_kl, psf_data, x0, y0, Lx, Ly, window_size)
    cdef int k, l
    cdef int dx, dy
    cdef int xl, yl
    cdef int x_psf, y_psf
    cdef double psf_val
    cdef idx2D_t pk
    cdef idx2D_t pl
    # write matrix elements
    for k in range(Lx*Ly):
        pk = idx2yx(k, Ly)
        for xl in range(int_max(pk.x-window_size, 0), int_min(pk.x+window_size+1, Lx)):
            pl.x = xl
            for yl in range(int_max(pk.y-window_size, 0), int_min(pk.y+window_size+1, Ly)):
                pl.y = yl
                l = yx2idx(pl, Ly)
                dx = pk.x-xl
                dy = pk.y-yl
                # calculate psf value for symmetric matrix element (l, k) = (k, l)
                x_psf = x0 + dx
                y_psf = y0 + dy
                if (0 <= y_psf < psf_data.shape[0]) and (0 <= x_psf < psf_data.shape[1]):
                    psf_val = psf_data[y_psf, x_psf]
                else:
                    psf_val = 0
                P_kl[k, l] = psf_val
                P_kl[l, k] = psf_val
    return P_kl


@cython.cdivision(True)
cdef theta_t grad(theta_t t,
                  np.ndarray[np.float64_t, ndim=1] kappa,
                  np.ndarray[np.complex_t, ndim=1] ploc,
                  np.ndarray[np.float64_t, ndim=1] cell_size):
    # Calculate the lens potential gradient to calculate the deflection
    # Args (4): (t, kappa, ploc, cell_size)
    cdef int i
    cdef double vx, vy
    cdef double a
    cdef theta_t v
    cdef theta_t dr

    cdef double xi, yi
    cdef double xm, xp, ym, yp
    cdef double xm2, xp2, ym2, yp2
    cdef double log_xm2_ym2, log_xp2_yp2, log_xp2_ym2, log_xm2_yp2

    v.x = 0
    v.y = 0

    for i in range(len(ploc)):
        dr.x = t.x - ploc[i].real
        dr.y = t.y - ploc[i].imag
        a = cell_size[i]/2
        xm = dr.x - a
        xp = dr.x + a
        ym = dr.y - a
        yp = dr.y + a
        xm2 = xm*xm
        xp2 = xp*xp
        ym2 = ym*ym
        yp2 = yp*yp
        log_xm2_ym2 = log(xm2 + ym2)
        log_xp2_yp2 = log(xp2 + yp2)
        log_xp2_ym2 = log(xp2 + ym2)
        log_xm2_yp2 = log(xm2 + yp2)

        vx = (xm*atan(ym/xm) + xp*atan(yp/xp)) + (ym*log_xm2_ym2 + yp*log_xp2_yp2) / 2 \
             - (xm*atan(yp/xm) + xp*atan(ym/xp)) - (ym*log_xp2_ym2 + yp*log_xm2_yp2) / 2
        vx /= pi
        vx *= kappa[i]

        vy = (ym*atan(xm/ym) + yp*atan(xp/yp)) + (xm*log_xm2_ym2 + xp*log_xp2_yp2) / 2 \
             - (ym*atan(xp/ym) + yp*atan(xm/yp)) - (xm*log_xm2_yp2 + xp*log_xp2_ym2) / 2
        vy /= pi
        vy *= kappa[i]

        v.x += vx
        v.y += vy
    return v


cdef theta_t deflect(theta_t t,
                     np.ndarray[np.float64_t, ndim=1] kappa,
                     np.ndarray[np.complex_t, ndim=1] ploc,
                     np.ndarray[np.float64_t, ndim=1] cell_size,
                     complex extra_deflection):
    # Get the full deflection angle for a GLASS model kappa map
    # Args (5): (t, kappa, ploc, cell_size, extra_deflection)
    cdef theta_t alpha
    alpha = grad(t, kappa, ploc, cell_size)
    alpha.x += extra_deflection.real
    alpha.y += extra_deflection.imag
    return alpha


cdef theta_t delta_beta(theta_t t, theta_t beta, double zcap,
                        np.ndarray[np.float64_t, ndim=1] kappa,
                        np.ndarray[np.complex_t, ndim=1] ploc,
                        np.ndarray[np.float64_t, ndim=1] cell_size,
                        complex extra_deflection):
    # Calculate deviations from modelled beta positions for an angular position map
    # Args (7): (t, beta, zcap, kappa, ploc, cell_size, extra_deflection)
    cdef theta_t dbeta
    cdef theta_t alpha
    alpha = deflect(t, kappa, ploc, cell_size, extra_deflection)
    dbeta.x = beta.x - t.x + alpha.x / zcap
    dbeta.y = beta.y - t.y + alpha.y / zcap
    return dbeta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef theta_t* srcgrid_deflections(theta_t* dbetas, int Lx, int Ly, ij,
                                  int origin_x, int origin_y, double px2arcsec,
                                  double source_x, double source_y, double zcap,
                                  np.ndarray[np.float64_t, ndim=1] kappa,
                                  np.ndarray[np.complex_t, ndim=1] ploc,
                                  np.ndarray[np.float64_t, ndim=1] cell_size,
                                  extra_potentials):
    # Calculate deviated source positions for an angular map
    # Args (14): (dbetas, Lx, Ly, ij, origin_x, origin_y, px2arcsec, source_x, source_y, zcap, \
    #             kappa, ploc, cell_size, extra_potentials)
    cdef:
        int LxL = Lx * Ly
        int i
        idx2D_t origin
        theta_t src, t
        double complex extra_d
    origin.x = origin_x
    origin.y = origin_y
    src.x = source_x
    src.y = source_y
    for i in range(len(ij)):
        t = theta(ij[i], Ly, origin, px2arcsec)
        # extra_deflection = complex(0, 0)
        for scale, e in extra_potentials:
            extra_deflection = complex(sum(scale * e.poten_dx(complex(t.x, t.y))),
                                        sum(scale * e.poten_dy(complex(t.x, t.y))))
        dbetas[i] = delta_beta(t, src, zcap, kappa, ploc, cell_size, extra_deflection)
    return dbetas


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double max_deflection(theta_t* dbetas, int l):
    # Get the maximal x or y value from the deviated source positions
    # Args (2): (dbetas, l)
    cdef int i
    cdef double r_max = 0
    for i in range(l):
        r_max = double_max(abs(dbetas[i].x), r_max)
        r_max = double_max(abs(dbetas[i].y), r_max)
    return r_max


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef idx2D_t* srcgrid_mapping(idx2D_t* xy, theta_t* dbetas, int l, int pixrad, double maprad):
    # Convert deviated source positions into pixel coordinates
    # Args (5): (xy, dbetas, l, pixrad, maprad)
    cdef int i
    cdef theta_t b
    for i in range(l):
        b = dbetas[i]
        xy[i].x = <int>(floor(pixrad*(1+b.x/maprad))+.5)
        xy[i].y = <int>(floor(pixrad*(1+b.y/maprad))+.5)
    return xy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inv_proj_matrix(Mij_p,
                      int Lx, int Ly, int N, np.ndarray[np.int32_t, ndim=1] image_ij,
                      int origin_x, int origin_y, double px2arcsec,
                      double source_x, double source_y, double zcap,
                      np.ndarray[np.float64_t, ndim=1] kappa,
                      np.ndarray[np.complex_t, ndim=1] ploc,
                      np.ndarray[np.float64_t, ndim=1] cell_size,
                      extra_potentials):
    # Fill the input matrix array using GLASS's deflection algorithm
    # Args (15): (Mij_p, Lx, Ly, N, image_ij, origin_x, origin_y, px2arcsec, source_x, source_y, zcap \
    #             kappa, ploc, cell_size, extra_potentials)
    cdef:
        int i, pidx, ijidx
        int LxL = Lx*Ly
        int NxN = N*N
        int M = N / 2
        int M_fullres, N_fullres, N_l, N_r, N_nil
        double map2px, r_max, r_fullres
        idx2D_t xy_i
        Pool mem = Pool()
        theta_t* dbetas = <theta_t*>mem.alloc(len(image_ij), sizeof(theta_t))
        theta_t* dbetas_fullres = <theta_t*>mem.alloc(LxL, sizeof(theta_t))
        idx2D_t* xy = <idx2D_t*>mem.alloc(LxL, sizeof(idx2D_t))

    ij = range(LxL)
    dbetas = srcgrid_deflections(dbetas, Lx, Ly, image_ij,
                                 origin_x, origin_y, px2arcsec,
                                 source_x, source_y, zcap, kappa, ploc, cell_size, extra_potentials)
    dbetas_fullres = srcgrid_deflections(dbetas_fullres, Lx, Ly, ij,
                                         origin_x, origin_y, px2arcsec,
                                         source_x, source_y, zcap, kappa, ploc, cell_size, extra_potentials)
    r_max = max_deflection(dbetas, len(image_ij))
    r_fullres = max_deflection(dbetas_fullres, LxL)
    
    map2px = M / r_max
    M_fullres = <int>(r_fullres*map2px+0.5)
    N_fullres = 2*M_fullres + 1
    N_l = N_fullres/2 - M
    N_r = N_fullres/2 + M
    N_nil = 0

    xy = srcgrid_mapping(xy, dbetas_fullres, LxL, M_fullres, r_fullres)

    for i in range(LxL):
        xy_i.x = xy[i].x
        xy_i.y = xy[i].y
        if (N_l < xy_i.x < N_r) and (N_l < xy_i.y < N_r):
            xy_i.x -= N_l
            xy_i.y -= N_l
        else:
            N_nil += 1
            continue
        pidx = yx2idx(xy_i, N)
        ijidx = ij[i]
        if pidx >= NxN:
            message = "Warning! Projection discovered pixel out of range in matrix construction!"
            message += " Something might be wrong!"
            print(message)
            continue
        Mij_p[ijidx, pidx] = 1
    return Mij_p, N_nil, M_fullres, r_fullres, r_max

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef d_p(np.float64_t[:] dij, Mij_p, sigmaM2,
          method='lsmr'):
    Qij_p = sigmaM2 * Mij_p
    A = Mij_p.T.tocsr() * Qij_p
    b = dij * Qij_p
    if method == 'lsmr':
        dp = splin.lsmr(A, b)[0]
    elif method == 'lsqr':
        dp = splin.lsqr(A, b)[0]
    elif method == 'cgs':
        dp = splin.cgs(A, b)[0]
    elif method == 'lgmres':
        dp = splin.lgmres(A, b, atol=1e-05)[0]
    elif method == 'minres':
        dp = splin.minres(A, b)[0]
    elif method == 'qmr':
        dp = splin.qmr(A, b)[0]
    elif method == 'row_norm':
        print("Row-norm is not yet compatible with cython!!!")
        exit(1)
    return dp
        


