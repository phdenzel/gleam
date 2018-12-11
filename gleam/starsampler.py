#!/usr/bin/env python
"""
@author: phdenzel

Mica, Mica, parva stella
Miror quaenam sis tam bella.
"""
###############################################################################
# Imports
###############################################################################
import os
import numpy as np
from scipy import interpolate

from gleam.skyf import SkyF
from gleam.skypatch import SkyPatch
from gleam.lensobject import LensObject
from gleam.multilens import MultiLens
from gleam.megacam import ADU2Mag, MegaCam2SSDS


###############################################################################
class StarSampler(object):
    """
    MCMC sampler class for stellar mass profiles of galaxy lenses
    """
    def __init__(self, data, mask=None, light_model=None, mag_transf=None, zl=None, zs=None,
                 verbose=False):
        """
        Set up parameter space for walkers

        Args:
            data <np.ndarray> - data to be compared to model

        Kwargs:
            mask <np.ndarray(bool)> - boolean mask to exclude in the model fitting
            light_model <gleam.model object> - a light model on the data (for model fitting)
            mag_transf <func> - data (ADU) to magnitude conversion function
            zl <float> - the lens redshift
            zs <float> - the source redshift
            verbose <bool> - verbose mode; print command line statements

        Return:
            <StarSampler object> - standard initializer
        """
        self.data = np.asarray(data)
        self.mask = mask
        self.light_model = light_model
        if self.light_model is not None:
            self.light_model.Ny, self.light_model.Nx = self.data.shape
        self.mag_transf = mag_transf
        self.zl = zl
        self.zs = zs
        if self.mask is None:
            self.mask = np.full(self.data.shape, True)
        if verbose:
            print(self.__v__)

    @classmethod
    def from_gleamobj(cls, gleam, band='i', lens_mask=None, image_mask=None, model='sersic',
                      **kwargs):
        """
        Initialize from gleamobj

        Args:
            gleam <gleam object> - contains data of a or more than one .fits files

        Kwargs:
            band <str/int> - the band to use the model on (if multiple data maps are passed)
            lens_mask <np.ndarray(bool)> - boolean masks to include in the model fitting
            image_masks <np.ndarray(bool)> - boolean masks to exclude in the model fitting
            model <str> - preferred model to use from the gleam object
            verbose <bool> - verbose mode; print command line statements

        Return:
            <StarSampler object> - initialized with gleam object
        """
        if not isinstance(gleam, (SkyF, SkyPatch, LensObject, MultiLens)):
            return
        if len(gleam.data.shape) > 2:
            if band in gleam.bands:
                gleam = gleam[band]
            else:
                gleam = gleam[0]
        data = gleam.data
        if hasattr(gleam, 'light_model'):
            if 'sersic' in gleam.light_model:
                kwargs.setdefault('light_model', gleam.light_model['sersic'])
            else:
                model_name = list(gleam.light_model.keys())[0]
                kwargs.setdefault('light_model', gleam.light_model[model_name])
        if hasattr(gleam, 'mag_formula'):
            kwargs.setdefault('mag_transf', gleam.mag_formula)
        if hasattr(gleam, 'zl'):
            kwargs.setdefault('zl', gleam.zl)
        if hasattr(gleam, 'zs'):
            kwargs.setdefault('zs', gleam.zs)
        if lens_mask:
            lens_mask = lens_mask
        elif hasattr(gleam, 'roi') and gleam.roi:
            lens_mask = np.logical_or.reduce(gleam.roi._masks['circle'])
        else:
            lens_mask = np.full(data.shape, True)
        if image_mask:
            image_mask = image_mask
        elif hasattr(gleam, 'roi') and gleam.roi:
            image_mask = np.logical_or.reduce(gleam.roi._masks['polygon'])
        else:
            image_mask = np.full(data.shape, False)
        kwargs.setdefault('mask', np.logical_and(lens_mask, ~image_mask))
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
        return ['data', 'mask', 'light_model', 'mag_transf', 'zl', 'zs']

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

    def chabrier_estimate(self, band_data=None):
        """
        Get a rough estimate of the stellar mass by looking up the Chabrier IMF table

        Args:
            None

        Kwargs:
            band_data <list> - a list of data from different bands to get a better estimate

        Return:
            M_stel <float,list(float)> - total stellar mass estimate (-1sig, median, +1sig)
        """
        # get the integrated light in units of ADU
        ADU_data = self.light_model.integrate(self.data, mask=self.mask)
        self.light_model.calc_map()
        light_norm = self.light_model.normalize(self.light_model.map2D, mask=self.mask)
        light_model = light_norm*ADU_data
        ADU_model = self.light_model.integrate(light_model)
        # ADU_model = self.light_model.integrate(self.light_model.map2D)
        Mag_model = ADU2Mag(ADU_model)
        # print("ADU/Mag: {}, {}".format(ADU_model, Mag_model))
        if band_data is None:
            ADU_data = self.light_model.integrate(self.data, mask=self.mask)
            Mag_data = ADU2Mag(ADU_data)
        else:
            ADU_data = [self.light_model.integrate(data, mask=self.mask) for data in band_data]
            Mag_data = ADU2Mag(ADU_data)
            # usually doesn't make much of a difference, but why not
            Mag_data = MegaCam2SSDS(Mag_data)
            Mag_data = Mag_data[3]  # only 'i' band necessary
        # load Chabrier table and look up the mass corresponding to the magnitude
        m_stel = self.chabrier_table_mass(Mag_model, self.zl)
        return m_stel

    @staticmethod
    def load_chabrier_table(filepath=None):
        """
        Load the Chabrier table into a dictionary

        Args:
            None

        Kwargs:
            filepath <str> - the filepath to the Chabrier IMF table

        Return:
            table <dict> - the dictionary with the Chabrier IMF table

        Note:
            <redshift>          - redshift
            <distance_modulus>  - distance modulus
            <invH>              - age at the Universe at redshift z
            <mag_halfGy>        - magnitude of 1 Msol of an 0.5 Gy population
            <mass_halfGy>       - actual mass at 0.5 Gy
            <mag_2Gy>           - magnitude of 1 Msol of a 2 Gy population
            <mass_2Gy>          - actual mass at 2 Gy
            <mag_tU>            - magnitude of 1 Msol of a tU population
            <mass_tU>           - actual mass at tU
        """
        if filepath is None:
            filepath = os.path.abspath(os.path.dirname(__file__)) \
                       + "/tables/i_IMF_chabrier.dat"
        table = {}
        keys = ['redshift', 'distance_modulus', 'invH', 'mag_halfGy',
                'mass_halfGy', 'mag_2Gy', 'mass_2Gy', 'mag_tU', 'mass_tU']
        for k in keys:
            table[k] = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i > 12:
                    items = line.split()
                    for i, k in enumerate(keys):
                        table[k].append(float(items[i]))
        # convert to numpy arrays
        for k in keys:
            table[k] = np.array(table[k])
        return table

    @staticmethod
    def chabrier_table_mass(magnitude, redshift, mass_key=None, mag_key=None,
                            table=None):
        """
        Get a rough estimate of the total stellar mass using a Chabrier IMF table

        Args:
            magnitude <float> - total magnitude of the lens's light
            redshift <float> - redshift of the lens

        Kwargs:
            mass_key <str> - the mass key from which to look up the stellar mass
            mag_key <str> - the magnitude key from which to look up the magnitude per Msol factor
            table <dict> - the Chabrier table from which to get the values
                           (if None loaded from default path)

        Return:
            M_stel <float,list(float)> - total stellar mass estimate (default: lo, median, hi)
        """
        if table is None:
            table = StarSampler.load_chabrier_table()
        z = np.abs(table['redshift']-redshift).argmin()
        # actual mass
        if mass_key is None:
            M = np.array([table['mass_halfGy'][z], table['mass_2Gy'][z], table['mass_tU'][z]])
        else:
            M = table[mass_key][z]
        # magnitude per 1 Msol
        if mag_key is None:
            mag = np.array([table['mag_halfGy'][z], table['mag_2Gy'][z], table['mag_tU'][z]])
        else:
            mag = table[mag_key][z]
        return M * np.power(10, 0.4*(mag-magnitude))

    @classmethod
    def resample_map(self, data_map, extent, new_shape, new_extent):
        """
        Resample and interpolate a map to new extent

        Args:
            data_map <np.ndarray> - the map to be extended
            extent <float> - from center to edge

        Kwargs:
            shape <> - None
        """
        x = np.linspace(extent[0], extent[2], data_map.shape[0]),
        y = np.linspace(extent[1], extent[3], data_map.shape[1])
        newx = np.linspace(new_extent[0], new_extent[2], new_shape[0])
        newy = np.linspace(new_extent[1], new_extent[3], new_shape[1])
        rescale = interpolate.interp2d(x, y, data_map)
        stel_map = rescale(newx, newy)
        return stel_map


if __name__ == "__main__":
    pass  # TODO
