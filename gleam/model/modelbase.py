#!/usr/bin/env python
"""
@author: phdenzel

Elliptical 2D base model profile

Base parameters:
    - x, y: center coordinates
    - phi: position angle
    - e: ellipticity
    - I_0: intensity in the center
    - c_0: box parameter
Image parameters:
    - Nx, Ny: image size (in pixels)
    - n_subsamples: number of sub-pixels

Notes:
    - total number of model parameters is 6
"""
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np


class _BaseModel(object):
    __metaclass__ = ABCMeta
    """
    To be inherited by classes specifying model profile

    Note:
       Instantiation won't work until all abstract methods/properties
       are overridden.
    """
    parameter_keys = ['x', 'y', 'phi', 'e', 'I_0', 'c_0']
    map_keys = ['Nx', 'Ny', 'n_subsamples']

    def __init__(self, x=None, y=None, phi=0., e=0., I_0=1., c_0=0.,
                 Nx=100, Ny=100, n_subsamples=1, auto_load=False,
                 verbose=False):
        """
        Initialize base model parameters and set default values

        Args:
            None

        Kwargs:
            x <float> - first center coordinate on x-axis
            y <float> - second center coordinate on y-axis
            phi <float> - position angle (in radians [0, 2\pi])
            e <float> - ellipticity [0, 1]
            I_0 <float> - intensity in the center
            c_0 <float> - box parameter
            Nx <int> - number of pixels along the x-axis
            Ny <int> - number of pixels along the y-axis
            n_subsamples: number of sub-sampling pixels
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        # Base model parameters
        self.x = x or Nx // 2
        self.y = y or Ny // 2
        self.phi = phi
        self.e = e
        self.I_0 = I_0
        self.c_0 = c_0
        # map parameters
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.n_subsamples = int(n_subsamples)
        self.map2D = np.zeros((self.Nx, self.Ny))
        self.indices = np.indices((self.Nx, self.Ny))

        if auto_load:
            self.calc_surface()

        if verbose:
            print(self.__v__)

    @property
    def tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return self.__class__.parameter_keys + ['N'] \
            + self.__class__.map_keys

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of SkyF attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in self.tests])

    @property
    def N(self):
        """
        Number of model parameters

        Args/Kwargs:
            None

        Return:
            N <int> - number of model parameters
        """
        return len(self.parameter_keys)

    @abstractproperty
    def profile_scale(self):
        """
        Abstract: To be specified in the derived class
        The specific scale variable of the model

        Args/Kwargs:
            None

        Return:
            scale <float> - intrinsic scale of the model
        """
        return None

    @property
    def q(self):
        """
        Precompute useful quantity: q = 1 - e

        Args/Kwargs:
            None

        Return:
            q <float> - one minus e
        """
        if hasattr(self, '_q'):
            if self.e in self._q:
                return self._q[self.e]
        else:
            self._q = {}
        self._q[self.e] = 1 - self.e
        return self._q[self.e]

    @q.setter
    def q(self, _q):
        """
        Setter for q = 1 - e

        Args:
            _q <float> - value for q

        Kwargs/Return:
            None
        """
        self.e = 1 - _q
        self._q[self.e] = _q

    @property
    def cosPA(self):
        """
        Precompute useful quantity: cosine of the position angle

        Args/Kwargs:
            None

        Return:
            cosPA <np.float64> - cosine of the position angle
        """
        if hasattr(self, '_cosPA'):
            if self.phi in self._cosPA:
                return self._cosPA[self.phi]
        else:
            self._cosPA = {}
        self._cosPA[self.phi] = np.cos(self.phi)
        return self._cosPA[self.phi]

    @property
    def sinPA(self):
        """
        Precompute useful quantity: sine of the position angle

        Args/Kwargs:
            None

        Return:
            sinPA <np.float64> - sine of the position angle
        """
        if hasattr(self, '_sinPA'):
            if self.phi in self._sinPA:
                return self._sinPA[self.phi]
        else:
            self._sinPA = {}
        self._sinPA[self.phi] = np.sin(self.phi)
        return self._sinPA[self.phi]

    @property
    def model_parameters(self):
        """
        Get model parameter values of the instance in a list

        Args/Kwargs:
            None

        Return:
            model_parameters <list> - list of model parameter values
        """
        return [self.__getattribute__(p) for p in self.parameter_keys]

    @model_parameters.setter
    def model_parameters(self, model_pars):
        """
        Setter for model parameter values

        Args:
            model_pars <list/dict> - new model parameters

        Kwargs/Return:
            None
        """
        if isinstance(model_pars, dict):
            for k in model_pars:
                self.__setattr__(k, model_pars[k])
        elif isinstance(model_pars, list):
            for k, i in zip(self.parameter_keys, model_pars):
                self.__setattr__(k, i)

    @property
    def map_parameters(self):
        """
        Get map parameter values of the instance in a list

        Args/Kwargs:
            None

        Return:
            map_parameters <list> - list of map parameter values
        """
        return [self.__getattribute__(p) for p in self.map_keys]

    def set_model_parameters(self, model_pars, verbose=False):
        """
        Setter function wrapper for model_parameters.setter

        Args:
            model_pars <list/dict> - new model parameters

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        self.model_parameters = model_pars
        if verbose:
            print(model_pars)

    @abstractmethod
    def get_major_profile(self, a):
        """
        Abstract: To be specified in the derived class
        Model observable at radius a along the semi-major axis
        
        Args:
            a <float> - radial distance from the center along the semi-major axis
        Kwargs:
            None

        Return:
            None
        """
        pass

    def calc_pixel(self, x, y, subsampling=False):
        """
        Returns the intensity value for an x, y pixel coordinate
        """
        delx = x-self.x
        dely = y-self.y

        # coordinate transformation to ellipse
        if self.c_0 == 0:
            x0 = delx*self.cosPA+dely*self.sinPA
            # scaled with 1/(axis ratio)
            y0 = (-delx*self.sinPA+dely*self.cosPA)/self.q
            r = np.sqrt(x0*x0+y0*y0)
        else:
            ell_exp = self.c_0+2.
            x0 = np.fabs(delx*self.cosPA+dely*self.sinPA)
            y0 = np.fabs((-delx*self.sinPA+dely*self.cosPA)/self.q)
            ellipse_r = x0**(ell_exp)+y0**(ell_exp)
            r = ellipse_r**(1./ell_exp)

        # subsampling
        n_subsamples = 1
        if subsampling and (r < 10.):
            n_subsamples = self.calc_subsamples(r)
        self.n_subsamples = n_subsamples

        if self.n_subsamples > 1:
            delSubpix = 1./self.n_subsamples
            x_sub_start = x-.5+.5*delSubpix
            y_sub_start = y-.5+.5*delSubpix
            theSum = 0.
            for ii in xrange(self.n_subsamples):
                x_ii = x_sub_start+ii*delSubpix
                delx = x_ii-self.x
                for jj in xrange(self.n_subsamples):
                    y_ii = y_sub_start+jj*delSubpix
                    dely = y_ii-self.y
                    x0 = np.fabs(delx*self.cosPA+dely*self.sinPA)
                    y0 = np.fabs((-delx*self.sinPA+dely*self.cosPA)/self.q)
                    ell_exp = self.c_0+2.
                    ellipse_r = x0**(ell_exp)+y0**(ell_exp)
                    r = ellipse_r**(1./ell_exp)
                    theSum += self.get_major_profile(r)
            totalIntensity = theSum/(self.n_subsamples*self.n_subsamples)
        else:
            totalIntensity = self.get_major_profile(r)
        return totalIntensity

    def get_map(self):
        """
        Returns map as numpy array
        """
        return np.array(self.map2D)

    def calc_map(self, **kwargs):
        """
        Fills the map with given model
        kwargs: subsampling - boolean
        """
        self.map2D = self.calc_pixel(self.indices[0], self.indices[1],
                                     **kwargs)

    def calc_subsamples(self, r, r_sample=10):
        """
        Returns number of subsamples depending on the radius
        kwargs: r_sample    - double; usually around 10
        """
        n_subsamples = 1
        if (self.profile_scale <= 1.) and (r <= 1.):
            n_subsamples = min(100, int(2*r_sample/self.profile_scale))
        else:
            if r <= 4:
                n_subsamples = 2*r_sample
            else:
                n_subsamples = min(100, int(2*r_sample/r))
        return n_subsamples

    def plot_model(self, factor=1, log=False, cbar=True, scalebar=None,
                   title=None, show=False,
                   **kwargs):
        """
        Plots the map once it has been calculated
        """
        # Additional imports
        import warnings
        warnings.filterwarnings("ignore", module="matplotlib")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        # plotting
        fig, ax = plt.subplots()
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'none'
        if log:
            kwargs['norm'] = LogNorm()
        img = ax.imshow(factor*self.map2D, **kwargs)
        if cbar:
            clrbar = fig.colorbar(img)
            clrbar.outline.set_visible(False)
        if title is not None:
            plt.title(title, fontsize=20)
        nx, ny = self.map2D.shape
        ax.set_xlim(xmin=0, xmax=nx)
        ax.set_ylim(ymin=0, ymax=ny)
        if scalebar is not None:
            from matplotlib import patches
            barpos = (0.05*nx, 0.025*ny)
            w = nx*0.15
            h = ny*0.01
            rect = patches.Rectangle(barpos, w, h,
                                     facecolor='white', edgecolor=None,
                                     alpha=0.85)
            ax.add_patch(rect)
            ax.text(barpos[0]+w/4, barpos[1]+ny*0.02,
                    r"$\mathrm{{{:.1f}''}}$".format(scalebar*w),
                    color='white', fontsize=16)
        ax.axis('off')
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_profile(self, factor=1, radius=None, bins=300, log=False,
                     title=None, show=False, **kwargs):
        """
        Plots the model profile along the semi-major axis
        """
        if radius is None:
            radius = self.Nx
        # Additional imports
        import warnings
        warnings.filterwarnings("ignore", module="matplotlib")
        import matplotlib.pyplot as plt
        # plotting
        fig, ax = plt.subplots()
        r = np.linspace(0, radius, bins)
        x_label = r'$\mathrm{radius\,[px]}$'
        y_label = r'$\mathrm{brightness\,[ADU]}$'
        if log:
            prfl = np.log10(self.get_major_profile(r))
            r = np.logspace(0, radius, bins)
            x_label = r'$\mathrm{log}\,$'+x_label
            y_label = r'$\mathrm{log}\,$'+y_label
        else:
            prfl = self.get_major_profile(r)
        ax.plot(r, factor*prfl, **kwargs)
        if title is not None:
            plt.title(title)
        ax.set_xlim(xmin=0, xmax=radius)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if show:
            plt.show()
        return fig

    @classmethod
    def integrate(cls, surface, radius, center=None):
        """
        Returns integrated surface upto radius r
        """
        surface = np.array(surface)  # just to be sure
        if center is None:
            center = [surface.shape[0]//2, surface.shape[1]//2]
        msk = np.indices(surface.shape)
        msk[0] -= center[0]  # shift to center
        msk[1] -= center[1]  # shift to center
        msk = np.square(msk)  # square for distance calculation
        msk = msk[0, :, :] + msk[1, :, :] < radius*radius
        surface[~msk] = 0
        return np.sum(surface[msk])

    @classmethod
    def expand_map(cls, pars, Nx, Ny, relative=False):
        """
        Expand parameters to new dimensions Nx, Ny
        """
        if relative:
            center = np.array([pars['Nx']//2, pars['Ny']//2])
            pos = np.array([pars['x'], pars['y']])
            shift = center - pos
        else:
            shift = np.array([0, 0])
        if 'x' in pars:
            pars['x'] = Nx//2 + shift[0]
        if 'y' in pars:
            pars['y'] = Ny//2 + shift[1]
        if 'Nx' in pars:
            pars['Nx'] = Nx
        if 'Ny' in pars:
            pars['Ny'] = Ny
        return pars

    @classmethod
    def normalize_map(cls, surface, radius=None):
        """
        Return normalized map and the total map
        """
        if radius is None:
            radius = surface.shape[0]//2
        total = cls.integrate(surface, radius)
        mask = None
        return surface/total, total, mask

    @classmethod
    def ensemble_error(cls, minpars, maxpars, Nensemble=200):
        """
        Generates ensembles of models and evaluates errors
        """
        # print minpars.keys()
        # TODO
        pass

    @classmethod
    def print_pars(cls, pars, name=None):
        """
        Prints parameters from a dictionary
        """
        if name is None:
            name = "Pars"
        print(name+":\n" + "".join(
            " {}\t-\t{}\n".format(k, pars[k]) for k in pars.keys()))
