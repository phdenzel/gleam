#!/usr/bin/env python
"""
@author: phdenzel

Linear algebra utilities for 2D quantities
"""
###############################################################################
# Imports
###############################################################################
import numpy as np
from gleam.utils.rgb_map import radial_mask


###############################################################################
def eigvals(q):
    """
    Calculate the eigenvalues for a 2 by 2 matrix

    Args:
        q <np.matrix> - the matrix for which to calculate the eigenvalues

    Kwargs:
        None

    Return:
        lambda1, lambda2 <float,float> - the two eigenvalues
    """
    a, b, c, d = q.flatten().tolist()[0]
    T = a + d
    D = a*d-b*c
    lambda1 = 0.5*T + np.sqrt(0.25*T*T - D)
    lambda2 = 0.5*T - np.sqrt(0.25*T*T - D)
    return lambda1, lambda2


def diag(q):
    """
    Calculate the diagonal matrix of a 2 by 2 matrix

    Args:
        q <np.matrix> - the matrix for which to calculate the eigenvalues

    Kwargs:
        None

    Return:
        d <np.matrix> - the diagonal matrix of q
    """
    l1, l2 = eigvals(q)
    if l1 != l2:
        return np.matrix([[l1, 0], [0, l2]])
    else:
        None


def eigvecs(q):
    """
    Calculate the eigenvectors for a 2 by 2 matrix

    Args:
        q <np.matrix> - the matrix for which to calculate the eigenvalues

    Kwargs:
        None

    Return:
        v <np.ndarray, np.ndarray> - the eigenvectors of q
    """
    a, b, c, d = q.flatten().tolist()[0]
    ratio = np.array([(a+c-li)/(li-b-d) for li in eigvals(q)])
    v1 = np.array([1, 1])  # choose something
    v = np.array([v1, ratio*v1]).T
    # assert angle(v[0], v[1]) - 0.5*np.pi < 1e-12
    return v[0], v[1]


def angle(v1, v2):
    """
    Find the angle between two 2D vectors

    Args:
        v1, v2 <np.ndarray> - the vectors

    Kwargs:
        None

    Return:
        angle <float> - the angle between the vectors
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    vdotv = v1.dot(v2)
    vvnorm = np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2))
    return vdotv/(vvnorm)


def is_symm2D(data, center=None):
    """
    Test if a 2D array is symmetric in its values

    Args:
        data <np.ndarray> - 2D data array

    Kwargs:
        center <tuple/list> - indices of center of symmetry

    Return:
        is_symm <bool> - True if data is symmetric, False otherwise
    """
    if center is None:
        center = [X//2 for X in data.shape]
    is_odd = [X % 2 for X in data.shape]
    left = data[:center[0], :]
    right = data[center[0]+is_odd[0]:, :][::-1]
    upper = data[:, :center[1]]
    lower = data[:, center[1]+is_odd[1]:][:, ::-1]
    is_symm = np.all(right == left) and np.all(upper == lower)
    return is_symm


def inner_product(grid1, grid2, rmask=False, radius=1):
    """
    The inner product of two grids in order to compare likeness

    Args:
        grid1, grid2 <np.ndarray> - the grid data, e.g. of two different models

    Kwargs:
        rmask <bool> - apply a radial mask before calculating the inner product

    Return:
        None
    """
    g1 = grid1.copy()
    g2 = grid2.copy()
    if rmask:
        radius = int(radius*0.5*g1.shape[0])
        msk = radial_mask(g1, radius=radius)
        g1[~msk] = 0
        g2[~msk] = 0
    return np.sum(g1*g2)/(np.linalg.norm(g1)*np.linalg.norm(g2))


def sigma_product(grid1, grid2, rmask=True, radius=0.7):
    """
    The inner product of two grids in order to compare likeness as deviations from the mean

    Args:
        grid1, grid2 <np.ndarray> - the grid data, e.g. of two different models


    Kwargs:
        rmask <bool> - apply a radial mask before calculating the inner product
        radius <float> - fractional radius of the radial mask

    Return:
        None
    """
    g1 = grid1.copy()
    g2 = grid2.copy()
    if rmask:
        radius = int(radius*0.5*g1.shape[0])
        msk = radial_mask(g1, radius=radius)
        g1[~msk] = np.mean(g1[msk])
        g2[~msk] = np.mean(g2[msk])
    g1 = g1 - np.mean(g1)
    g2 = g2 - np.mean(g2)
    return inner_product(g1, g2, rmask=False)


def sigma_product_map(grid1, grid2, rmask=True, radius=0.7):
    """
    The inner product map of two grids in order to locally compare likeness
    as deviations from the mean

    Args:
        grid1, grid2 <np.ndarray> - the grid data, e.g. of two different models

    Kwargs:
        rmask <bool> - apply a radial mask before calculating the inner product
        radius <float> - fractional radius of the radial mask

    Return:
        None
    """
    g1 = grid1.copy()
    g2 = grid2.copy()
    if rmask:
        radius = radius*0.5*g1.shape[0]
        msk = radial_mask(g1, radius=int(radius))
        g1[~msk] = np.mean(g1[msk])
        g2[~msk] = np.mean(g2[msk])
    g1 = g1 - np.mean(g1)
    g2 = g2 - np.mean(g2)
    normalization = (np.linalg.norm(g1)*np.linalg.norm(g2))
    return g1*g2/normalization
