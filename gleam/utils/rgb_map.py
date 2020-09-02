#!/usr/bin/env python
"""
@authot: phdenzel

Color transformation utilities for gleam maps
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration
from skimage.color import hsv2rgb
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


def stellar_fraction_map(stellar, total, error=None,
                         eA=0.5, ef=0.3, eDelta=0.75,
                         color_offset=0.666, center_fix=False,
                         alpha_mask=False):
    """
    Transform spatially resolved stellar fraction maps into an rgb channel image
    from an hsv combination of [f, 1-Delta, A]
    Inspired by Ferreras et al.; https://arxiv.org/abs/0710.3159

    Args:
        stellar <np.ndarray> - stellar mass map
        total <np.ndarray> - total mass map

    Kwargs:
        error <np.ndarray> - uncertainty map
        eA <float> - exponent fo the A component
        ef <float> - exponent fo the f component
        eDelta <float> - exponent fo the Delta component
        color_offset <float> - color wheel offset (to exclude some range)
        center_fix <bool> - fix center by averaging neighbors
        alpha_mask <bool> - use alpha mask derived from non-zero total map

    Return:
        rgba_channels <np.ndarray> - rgba channel arrays

    Notes:
        A:     fading
        f:     central concentration of red
        Delta: greyness
    """
    if error is None:
        error = np.zeros(total.shape)
    # calculate each component
    msk = total > 0
    A = 1*total
    f = 1*total
    Delta = 1*total
    A[msk] = np.power(total[msk], eA)
    f[msk] = np.power(stellar[msk]/total[msk], ef)
    Delta[msk] = np.power(Delta[msk], eDelta)
    # normalize
    A /= np.nanmax(A)
    f /= np.nanmax(f)
    Delta /= np.nanmax(Delta)
    # limit colour wheel (default: limits colors from blue to red)
    if isinstance(color_offset, float):
        coffset = color_offset
        crange = (1-color_offset)
    elif isinstance(color_offset, (tuple, list, np.ndarray)):
        coffset = color_offset[0]
        crange = color_offset[1]
    f = coffset + f*crange
    # calculate channels
    hsv = np.stack([f, 1.-Delta, A], axis=-1)
    rgb = hsv2rgb(hsv)
    alpha = total*0
    if alpha_mask:
        alpha[msk] = 1
    else:
        alpha[:, :] = 1
    rgba = np.dstack((rgb, alpha))
    N = rgba.shape[0]//2
    if center_fix:
        center = rgba[N-center_fix:N+center_fix+1,
                      N-center_fix:N+center_fix+1, :]
        rgba[N, N, :] = np.average(center, axis=(0, 1))
    return rgba

def stellar_fraction_cbar(A=1, error=None, color_offset=0.666, bins=200):
    """
    Return an rgb array for the custom colorbar derived from hsv colors
    """
    if error is None:
        error = 0
    elif error > 1:
        error = 1
    elif error < 0:
        error = 0
    if isinstance(color_offset, float):
        coffset = color_offset
        crange = (1-color_offset)
    elif isinstance(color_offset, (tuple, list, np.ndarray)):
        coffset = color_offset[0]
        crange = color_offset[1]
    f = np.linspace(coffset, coffset+crange, bins)
    hsv = np.zeros((bins, 3))
    hsv[:, 0] = f
    hsv[:, 1] = 1-error
    hsv[:, 2] = A
    rgb = np.array([hsv_to_rgb(c[0], c[1], c[2]) for c in hsv])
    return rgb