#!/usr/bin/env python
"""
@authot: phdenzel

Color transformation utilities for gleam maps
"""
import numpy as np
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb


def lupton_like(i, r, g, method='standard'):
    """
    Stack (u)gri(z) .fits data to produce a false-color picture;
    similar to Lupton et al. (2004): http://iopscience.iop.org/article/10.1086/382245

    Args:
        i, r, g <np.ndarray> - data maps of i, r, and g bands from .fits files

    Kwargs:
        method <str> - method description; chooses the set of weights for the stacking

    Return:
        stack <np.ndarray> - the stacked image data ready to use with matplotlib.pyplot.imshow
    """
    stack = np.zeros(i.shape+(4,))
    # determine type of stacking
    if method == 'standard':
        s_i = 0.4
        s_r = 0.6
        s_g = 1.7
        alpha = 0.09
        Q = 1.0
    elif method == 'brighter':
        s_i = 0.4
        s_r = 0.6
        s_g = 1.7
        alpha = 0.17
        Q = 1.0
    elif method == 'bluer':
        s_i = 0.4
        s_r = 0.6
        s_g = 2.5
        alpha = 0.11
        Q = 2.0
    else:
        s_i = 0.4
        s_r = 0.6
        s_g = 1.7
        alpha = 0.09
        Q = 1.0
    I = i*s_i + r*s_r + g*s_g
    # reds
    stack[:, :, 0] = i*s_i*np.arcsinh(alpha*Q*I)/(Q*I)
    # greens
    stack[:, :, 1] = r*s_r*np.arcsinh(alpha*Q*I)/(Q*I)
    # blues
    stack[:, :, 2] = g*s_g*np.arcsinh(alpha*Q*I)/(Q*I)
    # alphas
    stack[:, :, 3] = 1
    # limit numbers outside of range [0, 1]
    stack[stack < 0] = 0
    stack[stack > 1] = 1
    return stack


# def rgba_map(stellar, total, error=None, Aexp=0.5, fexp=2.5, dexp=0.25,
#              delta3=1./3, log=False, alpha_mask=False):
#     """
#     Return rgba map for lens pixels using I. Ferreras' scheme
#     see Ferreras et al.; https://arxiv.org/abs/0710.3159

#     Scaling exponents:
#        A:     fading
#        f:     central concentration of red
#        delta: greyness
#     """
#     # handle default
#     if error is None:
#         error = np.zeros(total.shape)
#         if log:
#             error = 1 + error
#     if log:
#         stellar = np.log10(stellar)
#         total = np.log10(total)
#         error = np.log10(error)
#     # mask for mapextent
#     msk = total > 0
#     # init A, f, delta
#     A = total*0
#     f = A*1
#     delta = A*1
#     # apply scaling
#     A[msk] = np.power(total[msk], Aexp)
#     f[msk] = np.power(stellar[msk]/total[msk], fexp)
#     delta[msk] = np.power(error[msk]/total[msk], dexp)
#     # normalize
#     A /= np.max(A)
#     f /= np.max(f)
#     delta /= np.max(delta)
#     # color transform to rgb
#     transf = np.zeros((f.shape[0], f.shape[1], 4))
#     transf[:, :, 0] = A*(delta3*delta+(1-delta)*f)
#     transf[:, :, 1] = A*(delta3*delta)
#     transf[:, :, 2] = A*(delta3*delta+(1-delta)*(1-f))
#     # apply alpha = 0 to regions outside of mapextent
#     if alpha_mask:
#         transf[msk, 3] = 1
#         transf[~msk, 3] = 0
#     else:
#         transf[:, :, 3] = 1
#     # normalize for brightest channel
#     transf[:, :, :-1] /= np.max(transf[:, :, :-1])
#     return transf


# def hsv2rgb_map(stellar, total, error=None, Aexp=0.5, fexp=2.5, dexp=0.25,
#                 log=False, reverse_A=False,
#                 alpha_mask=False, center_fix=False, center_print=False):
#     """
#     Return rgba map for lens pixels using an hsv scheme

#        f:     hue (stellar-to-lens mass fraction)
#        delta: saturation (lens mass error)
#        A:     value (lens mass normalization)
#     """
#     # handle default
#     if error is None:
#         error = np.zeros(total.shape)
#         if log:
#             error = 1 + error
#     if log:
#         stellar = np.log10(stellar)
#         total = np.log10(total)
#         error = np.log10(error)
#     # mask for mapextent
#     msk = total > 0
#     # init A, f, delta
#     A = total*0
#     f = A*1
#     delta = A*1
#     # apply scaling
#     A[msk] = np.power(total[msk], Aexp)
#     f[msk] = np.power(stellar[msk]/total[msk], fexp)
#     delta[msk] = np.power(error[msk]/total[msk], dexp)
#     # normalize
#     A /= np.max(A)
#     f /= np.max(f)
#     f = 2./3 + f/3  # limit colors from blue to red
#     delta /= np.max(delta)
#     # color transform
#     N = f.shape[0]
#     M = f.shape[1]
#     transf = np.zeros((N, M, 4))
#     transf[:, :, 0] = f
#     transf[:, :, 1] = 1-delta
#     if reverse_A:
#         transf[:, :, 2] = 1-A
#     else:
#         transf[:, :, 2] = A
#     # map hsv to rgb
#     hsv = transf[:, :, :-1].reshape((N*M, 3))
#     pixel = np.array(
#         [hsv_to_rgb(hsv[i, 0], hsv[i, 1], hsv[i, 2]) for i in xrange(N*M)])
#     transf[:, :, :-1] = pixel.reshape((N, M, 3))
#     # apply alpha = 0 to regions outside of mapextent
#     if alpha_mask:
#         transf[msk, 3] = 1
#         transf[~msk, 3] = 0
#     else:
#         transf[:, :, 3] = 1
#     if center_print:  # print the innermost 9 pixels
#         print("f: {:.2f}".format(f[N//2, M//2]))
#         print("A: {:.2f}".format(A[N//2, M//2]))
#         print("d: {:.2f}".format(1-delta[N//2, M//2]))
#         np.set_printoptions(formatter={'float': '{:.2f}'.format})
#         print("\t r,   g,   b,   a")
#         for i in [-2, -1, 0, 1, 2]:
#             for j in [-2, -1, 0, 1, 2]:
#                 print("transf: {} at ({: d},{: d})".format(
#                     transf[N//2+i, M//2+j, :],
#                     i,
#                     j))
#             print
#     if center_fix:  # fix the innermost 9 pixels
#         for i in [-1, 0, 1]:
#             for j in [-1, 0, 1]:
#                 if abs(transf[N//2+i, M//2+j, 1]
#                        - transf[N//2+i, M//2+j, 2]) < 0.025:
#                     transf[N//2+i, M//2+j, 0] = transf[N//2, M//2, 0]
#                     transf[N//2+i, M//2+j, 0] += transf[N//2+i*2, M//2+j*2, 0]
#                     transf[N//2+i, M//2+j, 0] /= 2
#                     transf[N//2+i, M//2+j, 1] = transf[N//2, M//2, 1]
#                     transf[N//2+i, M//2+j, 1] += transf[N//2+i*2, M//2+j*2, 1]
#                     transf[N//2+i, M//2+j, 1] /= 2
#                     transf[N//2+i, M//2+j, 2] = transf[N//2, M//2, 2]
#                     transf[N//2+i, M//2+j, 2] += transf[N//2+i*2, M//2+j*2, 2]
#                     transf[N//2+i, M//2+j, 2] /= 2
#     # normalize for brightest channel
#     transf[:, :, :-1] /= np.max(transf[:, :, :-1])
#     return transf


# def rgb_cbar(error=None, bins=200):
#     """
#     Return an rgb array for the custom colorbar
#     """
#     # handle default
#     if error is None:
#         error = 0
#     elif error > 1:
#         error = 1
#     elif error < 0:
#         error = 0
#     color_range = np.zeros((bins, 3))
#     f = np.linspace(0.0, 1.0, bins)
#     color_range[:, 0] = 1./3*error+(1-error)*f
#     color_range[:, 1] = 1./3*error
#     color_range[:, 2] = 1./3*error+(1-error)*(1-f)
#     return color_range


# def hsv2rgb_cbar(error=None, A=1, bins=200, reverse_A=False):
#     """
#     Return an rgb array for the custom colorbar derived from hsv colors
#     """
#     if error is None:
#         error = 0
#     elif error > 1:
#         error = 1
#     elif error < 0:
#         error = 0
#     hsv_range = np.zeros((bins, 3))
#     f = np.linspace(2./3, 1.0, bins)
#     hsv_range[:, 0] = f
#     hsv_range[:, 1] = 1-error
#     if reverse_A:
#         hsv_range[:, 2] = 1-A
#     else:
#         hsv_range[:, 2] = A
#     return np.array([hsv_to_rgb(c[0], c[1], c[2]) for c in hsv_range])


# def scale_gray(composite):
#     """
#     Return a grayscale copy of composite rgb picture; see Digital ITU BT.601
#     """
#     grayscale = composite*0
#     luminance = np.dot(composite[..., :-1], [0.299, 0.587, 0.114])
#     grayscale[..., 0] = luminance
#     grayscale[..., 1] = luminance
#     grayscale[..., 2] = luminance
#     grayscale[..., 3] = 1.
#     return grayscale


# def filter_from_color(rgba, color, delta=0.1):
#     """
#     Filter out the colors of images
#     """
#     mask = []
#     return mask


# def primary_transf(rgba, roundup=0.5):
#     """
#     Pronounce pictures with primary colors
#     """
#     width, height, channels = rgba.shape
#     primary = rgba*1
#     primary[np.greater_equal(primary, roundup)] = 1
#     primary[np.less(primary, roundup)] = 0
#     return primary


# def primary_mask(primary, r=True, g=True, b=True, verbose=False):
#     """
#     Get a mask where the primary colors are
#     """
#     msk = False
#     if verbose:
#         print("Selected primary color(s)...\t"),
#     if r:
#         msk = np.equal(primary[:, :, 0], 1) | (msk)
#         if verbose:
#             print("red\t"),
#     if g:
#         msk = np.equal(primary[:, :, 1], 1) | (msk)
#         if verbose:
#             print("green\t"),
#     if b:
#         msk = np.equal(primary[:, :, 2], 1) | (msk)
#         if verbose:
#             print("blue\t")
#     return msk


# def plot_rgbamap(rgba, scalebar=None, **kwargs):
#     """
#     Plot rgba-arraylike data (N, M, 4) in a grid (N, M)
#     """
#     fig = plt.figure(figsize=(8, 8))
#     try:
#         nx, ny, _ = rgba.shape
#     except:
#         nx, ny, = rgba.shape
#     ax = fig.add_subplot(111)
#     ax.imshow(rgba, **kwargs)
#     ax.set_xlim(left=0, right=nx)
#     ax.set_ylim(bottom=0, top=ny)
#     ax.set_aspect('equal')
#     plt.axis('off')
#     plt.tight_layout()
#     if scalebar is not None:
#         from matplotlib import patches
#         barpos = (0.05*nx, 0.025*ny)
#         w = nx*0.15
#         h = ny*0.01
#         rect = patches.Rectangle(barpos, w, h,
#                                  facecolor='white', edgecolor=None,
#                                  alpha=0.85)
#         ax.add_patch(rect)
#         ax.text(barpos[0]+w/4, barpos[1]+ny*0.02,
#                 "$\mathrm{{{:.1f}''}}$".format(scalebar*w),
#                 color='white', fontsize=16)
#     plt.subplots_adjust(wspace=0.003, hspace=0.003)
#     return fig
