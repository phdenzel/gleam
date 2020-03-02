#!/usr/bin/env python
"""
@authot: phdenzel

Color transformation utilities for gleam maps
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration
from skimage.color import hsv2rgb
# from colorsys import hsv_to_rgb


def lupton_like(i, r, g, method='standard'):
    """
    Stack (u)gri(z) .fits data to produce a false-color picture;
    similar to Lupton et al. (2004): http://iopscience.iop.org/article/10.1086/382245

    Args:
        i, r, g <np.ndarray> - data maps of i, r, and g bands from .fits files

    Kwargs:
        method <str> - method description; chooses the set of weights for the stacking

    Return:
        stack <np.ndarray> - stacked image data of shape(N,M,4); ready for matplotlib.pyplot.imshow
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
    intensity = i*s_i + r*s_r + g*s_g
    # reds
    stack[:, :, 0] = i*s_i*np.arcsinh(alpha*Q*intensity)/(Q*intensity)
    # greens
    stack[:, :, 1] = r*s_r*np.arcsinh(alpha*Q*intensity)/(Q*intensity)
    # blues
    stack[:, :, 2] = g*s_g*np.arcsinh(alpha*Q*intensity)/(Q*intensity)
    # alphas
    stack[:, :, 3] = 1
    # limit numbers outside of range [0, 1]
    stack[stack < 0] = 0
    stack[stack > 1] = 1
    return stack


def asin_stack(r, g, b, s_r=0.4, s_g=0.6, s_b=1.7, alpha=0.09, Q=1.0):
    """
    Stack data from 3 .fits files to produce a false-color picture;

    Args:
        r, g, b <np.ndarray> - data maps using arbitrary filters

    Return:
        stack <np.ndarray> - stacked image data of shape(N,M,4); ready for matplotlib.pyplot.imshow
    """
    stack = np.zeros(r.shape+(4,))
    # determine type of stacking
    # r = (r - r.min()) / (r.max() - r.min())
    # g = (g - g.min()) / (g.max() - g.min())
    # b = (b - b.min()) / (b.max() - b.min())
    intensity = r*s_r + g*s_g + b*s_b
    # reds
    stack[:, :, 0] = r*s_r*np.arcsinh(alpha*Q*intensity)/(Q*intensity)
    # greens
    stack[:, :, 1] = g*s_g*np.arcsinh(alpha*Q*intensity)/(Q*intensity)
    # blues
    stack[:, :, 2] = b*s_b*np.arcsinh(alpha*Q*intensity)/(Q*intensity)
    # alphas
    stack[:, :, 3] = 1
    # limit numbers outside of range [0, 1]
    stack[stack < 0] = 0
    stack[stack > 1] = 1
    return stack


def hsv_stack(c1, c2, c3, hues=[360, 180, 240], rf=1., gf=1., bf=1.,
              l1=1., l2=1., l3=1., alpha=0.09, Q=1.0,
              saturation_curve=np.arcsinh):
    """
    Sets the hue of each individual channel and converts into rgba data image
    """
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    c3 = np.asarray(c3)
    c1 = (c1 - c1.min()) / (c1.max() - c1.min())
    c2 = (c2 - c2.min()) / (c2.max() - c2.min())
    c3 = (c3 - c3.min()) / (c3.max() - c3.min())
    # hue = (hues[0]*c1*rf + hues[1]*c2*gf + hues[2]*c3*bf) / (c1*rf+c2*gf+c3*bf)
    # value = c1*l1 + c2*l2 + c3*l3
    # saturation = saturation_curve(Q*value) / (Q*value)
    stacks = []
    for i, (l, c) in enumerate(zip([l1, l2, l3], [c1, c2, c3])):
        stack = np.zeros(c1.shape+(4,))
        value = c#*l*3
        hue = hues[i] / 360.
        stack[:, :, 0] = hue
        stack[:, :, 1] = 1. - (1.-2*value)**2  # np.ones_like(value)
        stack[:, :, 2] = value
        stack[:, :, :3] = hsv2rgb(stack[:, :, :3])
        stack[:, :, 3] = 1
        stacks.append(stack[:])
    # stack = (stacks[0]*rf + stacks[1]*gf + stacks[2]*bf)
    return stacks


def richardson_lucy(img, psf=np.ones((5, 5))/25, iterations=30):
    """
    Richardson-Lucy deconvolution
    """
    deconv = restoration.richardson_lucy(img.copy(), psf, iterations=iterations)
    return deconv


def rgba(data, cmap=None):
    """
    Create a 4-channel rgba image array from data

    Args:
        data <np.ndarray> - data to be converted into an rgba array

    Kwargs:
        cmap <str> - color map to be used to create colors; default: matplotlib default
    """
    if len(data.shape) == 3 and data.shape[-1] == 4:  # already rgba
        return data
    if len(data.shape) != 2:
        raise IndexError(
            "Wrong input shape {}! Input 2D data with shape of (N, M)!".format(data.shape))
    if cmap is None:
        cmap = plt.get_cmap()
    img = cmap(plt.Normalize(data.min(), data.max())(data))
    return img


def grayscale(rgba):
    """
    Transform rgba image data into a grayscale picture copy; see Digital ITU BT.601

    Args:
        rgba <np.ndarray> - rgba image data of shape(N,M,i=1,3,4)

    Kwargs:
        None

    Return:
        gray <np.ndarray> - grayscale image of the rgba image data
    """
    if len(rgba.shape) == 2:
        rgba = rgba.reshape(rgba.shape+(1,))
    if len(rgba.shape) != 3 and rgba.shape[-1] > 4:
        raise IndexError("Wrong input shape! rgba requires shape of (N, M, i={1,3,4})!")
    if rgba.shape[-1] < 4:
        img = np.ones(rgba.shape[:-1]+(4,))
        img[..., :-1] = rgba
    else:
        img = rgba
    gray = img*0
    luminance = np.dot(img[..., :-1], [0.299, 0.587, 0.114])
    gray[..., 0] = luminance
    gray[..., 1] = luminance
    gray[..., 2] = luminance
    gray[..., 3] = 1.
    gray = plt.Normalize(gray[..., :-1].min(), gray[..., :-1].max())(gray)
    return gray


def radial_mask(data, center=None, radius=None):
    """
    Create a circular mask for the input data

    Args:
        data <np.ndarray> - image data for which the mask is created

    Kwargs:
        center <tuple/list(int)> - center indices of the mask on the data
        radius <int> - radius of the circular mask

    Return:
        mask <np.ndarray(bool)> - circular boolean mask
    """
    if 2 > len(data.shape) > 3:
        raise IndexError(
            "Wrong input shape {}! " \
            + "Input 2D data with shape of (N, M), (N, M, 3) or (N, M, 4)!".format(data.shape))
    h, w = data.shape[:2]
    if center is None:
        center = [int(w//2), int(h//2)]
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


def primary_transf(rgba, roundup=0.5):
    """
    Pronounce pictures with primary colors

    Args:
        rgba <np.ndarray> - rbga image data of shape(N,M,4)

    Kwargs:
        roundup <float> - roundup threshold

    Return:
        primary <np.ndarray> - primary color data of the rgba image
    """
    if len(rgba.shape) != 3 and rgba.shape[-1] != 4:
        raise IndexError("Wrong input shape! rgba requires shape of (N, M, 4)!")
    primary = rgba*1
    primary[np.greater_equal(primary, roundup)] = 1
    primary[np.less(primary, roundup)] = 0
    return primary


def primary_mask(primary, r=True, g=True, b=True, verbose=False):
    """
    Get a mask where the primary colors are red, green, and/or blue

    Args:
        primary <np.ndarray> - primary color data of an rgba image

    Kwargs:
        r <bool> - include red in the mask
        g <bool> - include green in the mask
        b <bool> - include blue in the mask

    Return:
        msk <np.ndarray(bool)> - boolean mask of the selected primary colors
    """
    msk = False
    if verbose:
        print("Selected primary color(s)...\t"),
    if r:
        msk = np.equal(primary[:, :, 0], 1) | (msk)
        if verbose:
            print("red\t"),
    if g:
        msk = np.equal(primary[:, :, 1], 1) | (msk)
        if verbose:
            print("green\t"),
    if b:
        msk = np.equal(primary[:, :, 2], 1) | (msk)
        if verbose:
            print("blue\t")
    return msk


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


# def filter_from_color(rgba, color, delta=0.1):
#     """
#     Filter out the colors of images
#     """
#     mask = []
#     return mask


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
