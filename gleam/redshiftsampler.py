#!/usr/bin/env python
"""
@author: phdenzel

Mica, Mica, parva stella
Miror quaenam sis tam bella.
"""
###############################################################################
# Imports
###############################################################################
import sys
import os
from random import randint
import numpy as np

from gleam.skyf import SkyF
from gleam.skypatch import SkyPatch
from gleam.lensobject import LensObject
from gleam.multilens import MultiLens
from gleam.model.sersic import Sersic
from gleam.megacam import ADU2Mag


###############################################################################
class RedshiftSampler(object):
    """
    MCMC sampler class for stellar mass profiles of galaxy lenses
    """
    def __init__(self, data, lens_mask=None, image_masks=None, mag_transf=None,
                 sampler='bpz', verbose=False):
        """
        Read in data and masks

        Args:
            data <list(np.ndarray)> - data to be used to extract magnitudes

        Kwargs:
            lens_mask <np.ndarray(bool)> - boolean mask to extract lens magnitude
            image_masks <list(np.ndarray(bool))> - boolean mask to extract image magnitudes
            mag_transf <func> - data (ADU) to magnitude conversion function
            verbose <bool> - verbose mode; print command line statements

        Return:
            <RedshiftSampler object> - standard initializer
        """
        self.data = data
        self.lens_mask = lens_mask
        self.image_masks = image_masks
        self.mag_transf = mag_transf
        if sampler == 'bpz':
            self.root = sys.path[0] or sys.path[1]
            self.bpz_lib = os.path.join(self.root, 'src', 'bpz-1.99.3')
            self.bpz_catalogue = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'bpz')
            if len(self.data) == 5:
                self.bpz_catalogue = os.path.join(self.bpz_catalogue, 'bpz_ugriz.cat')
            elif len(self.data) == 6:
                self.bpz_catalogue = os.path.join(self.bpz_catalogue, 'bpz_ugrii2z.cat')
            else:
                print("Incomplete band data provided...")
                sys.exit(1)
            self.bpz_file = self.bpz_catalogue[:-3]+"bpz"
            self.bpz_probs = self.bpz_catalogue[:-3]+"probs"
            self.env_cmd = [". "+self.bpz_lib+"env_set.sh"]
            self.bpz_cmd = ["python "+self.bpz_lib+"bpz.py"+" "+self.bpz_catalogue]
        if self.lens_mask is None:
            self.lens_mask = np.full(self.data.shape, True)
        if self.image_masks is None:
            self.image_masks = [np.full(self.data.shape, False)]
        # update environment
        self.env_update()
        if verbose:
            print(self.__v__)

    @classmethod
    def from_gleamobj(cls, gleam, band='i', lens_mask=None, image_mask=None, model='sersic',
                      **kwargs):
        """
        Initialize from gleamobj

        Args:
            gleam <gleam object> - contains data more than one .fits files

        Kwargs:
            band <str> - the band to use to extract masks
            lens_mask <np.ndarray(bool)> - boolean masks to include in the model fitting
            image_masks <np.ndarray(bool)> - boolean masks to exclude in the model fitting
            model <str> - preferred model to use from the gleam object
            verbose <bool> - verbose mode; print command line statements

        Return:
            <StarSampler object> - initialized with gleam object
        """
        if not isinstance(gleam, (SkyF, SkyPatch, LensObject, MultiLens)):
            return
        if len(gleam.data.shape) < 3:
            return
        data = [g.data for g in gleam]
        gleam = gleam['i']
        if lens_mask:
            lens_mask = lens_mask
        elif hasattr(gleam, 'roi') and gleam.roi:
            lens_mask = np.logical_or.reduce(gleam.roi._masks['circle'])
        else:
            lens_mask = None
        kwargs.setdefault('lens_mask', lens_mask)
        if image_mask:
            image_masks = image_mask
        elif hasattr(gleam, 'roi') and gleam.roi:
            image_masks = gleam.roi._masks['polygon']
        else:
            image_masks = np.full(data.shape, False)
        kwargs.setdefault('image_masks', image_masks)
        if hasattr(gleam, 'mag_formula'):
            kwargs.setdefault('mag_transf', gleam.mag_formula)
        return cls(data, **kwargs)

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.data.shape)

    def __repr__(self):
        return self.__str__()

    @property
    def tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return ['data', 'lens_mask', 'image_masks', 'mag_transf',
                'lens_magnitudes', 'image_magnitudes',
                'root', 'bpz_lib', 'bpz_file', 'bpz_catalogue', 'env_cmd', 'bpz_cmd']

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of MCMCSampler attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in self.tests])

    @property
    def lens_magnitudes(self):
        """
        Lens magnitudes for all bands
        """
        magnitudes = []
        for d in self.data:
            image_mask = np.logical_or.reduce(self.image_masks)
            mask = np.logical_and(self.lens_mask, ~image_mask)
            ADU_lens = Sersic.integrate(d, mask=mask)
            mag_lens = ADU2Mag(ADU_lens)
            magnitudes.append(mag_lens)
        return np.array(magnitudes)

    @property
    def image_magnitudes(self):
        """
        Image magnitudes for all bands
        """
        magnitudes = []
        for d in self.data:
            image_mags = []
            for mask in self.image_masks:
                ADU_image = Sersic.integrate(d, mask=mask)
                mag_image = ADU2Mag(ADU_image)
                image_mags.append(mag_image)
            magnitudes.append(image_mags)
        return np.array(magnitudes)

    def add2bpz_cat(self, mags, errors=0.1):
        """
        Adds a line to the bpz catalogue file content with magnitudes and errors

        Args:
            mags <list(float)> - a list of magnitudes for each band

        Kwargs:
            errors <float> - errors in the magnitude data

        Return:
            TODO
        """
        # load content from catalogue
        with open(self.bpz_catalogue) as f:
            content = f.readlines()
        if not content[-1][0] == '#':
            content[-1] = '#'+content[-1]
        # add line to content
        N = len(mags)
        content.append('')
        content[-1] = (content[-1]
                       + ''.join([str(randint(0, 9)) for i in range(0, 4)])
                       + '   ')
        for i in range(N):
            content[-1] = (content[-1] + '{:.10f}'.format(mags[i])
                           + '   '
                           + '{:.3f}'.format(max(errors, 0.085))
                           + '   ')
        content[-1] = content[-1][:-3]+'\n'
        # write content back to catalogue
        with open(self.bpz_catalogue, 'w') as f:
            f.writelines(content)

    @staticmethod
    def run_command(command, _stdout=True, _stderr=False):
        """
        Run an external script command and return result

        Args:
            command <str> - the command string to run bpz

        Kwargs:
            _stdout <bool> -
            _stderr <bool> -

        Return:
            TODO
        """
        from subprocess import Popen, PIPE
        try:
            from subprocess import DEVNULL    # python3
        except ImportError:
            DEVNULL = open(os.devnull, 'wb')  # <= python2.7
        stdout_pipe = PIPE if _stdout else DEVNULL
        stderr_pipe = PIPE if _stderr else DEVNULL
        proc = Popen(command, stdout=stdout_pipe, stderr=stderr_pipe, shell=True)
        data = proc.communicate()
        return data

    def env_update(self, verbose=False):
        """
        Update the environment to be able to run bpz
        """
        env_data = self.run_command(self.env_cmd)[0]
        os.environ.update(
            dict((line.split("=", 1) for line in env_data.splitlines())))

    def run_bpz(self, verbose=False):
        """
        Run bpz with current state of bpz_file

        Args:
            bpz_file <> -
            command <> -

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            TODO
        """
        self.run_command(self.bpz_cmd, _stdout=False, _stderr=False)
        # read results
        with open(self.bpz_file) as f:
            content = f.readlines()
        bpz = float(content[-1].split()[1])
        bpz_min = float(content[-1].split()[2])
        bpz_max = float(content[-1].split()[3])
        if verbose:
            print("{0:.4f} < {1:.4f} < {2:.4f}".format(bpz_min, bpz, bpz_max))
        return bpz, bpz_min, bpz_max

    def plot_probs(self):
        """
        Plot the probability distribution for the redshift estimation

        Args/Kwargs/Return:
            None
        """
        import matplotlib.pyplot as plt
        z = np.linspace(0.01, 10.01, 1000)
        p = np.loadtxt(self.bpz_probs)
        plt.plot(z, p[1:])
        plt.show()


if __name__ == "__main__":
    pass  # TODO
