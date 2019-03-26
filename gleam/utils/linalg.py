#!/usr/bin/env python
"""
@author: phdenzel

Linear algebra utilities for 2D quantities
"""
###############################################################################
# Imports
###############################################################################
import numpy as np


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
    return np.arccos(vdotv/(vvnorm))


def inner_product(grid1, grid2):
    """
    The inner product of two grids in order to compare likeness

    Args:
        grid1, grid2 <np.ndarray> - the grid data, e.g. of two different models

    Kwargs:
        None

    Return:
        None
    """
    return np.sum(grid1*grid2)/(np.linalg.norm(grid1)*np.linalg.norm(grid2))


def sigma_product(grid1, grid2):
    """
    The inner product of two grids in order to compare likeness as deviations from the mean

    Args:
        grid1, grid2 <np.ndarray> - the grid data, e.g. of two different models

    Kwargs:
        None

    Return:
        None
    """
    grid1 = grid1 - np.mean(grid1)
    grid2 = grid2 - np.mean(grid2)
    return inner_product(grid1, grid2)
