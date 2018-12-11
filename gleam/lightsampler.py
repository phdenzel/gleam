#!/usr/bin/env python
"""
@author: phdenzel

And now, move, move my little walkers...

Note:
    - As of now, only Sersic profiles work, but in general, different models should work as well.
      Requirements for classes:
          o optional: class file should be located in gleam/model/ with class name == file name
          o model class must inherit from _BaseModel (gleam/model/modelbase.py)
          o model class needs 'parameter_keys' (<list> class attribute)
          o model class needs priors property expanded
"""
###############################################################################
# Imports
###############################################################################
import numpy as np
from matplotlib import pyplot as plt
import emcee  # http://dan.iel.fm/emcee/current/

from gleam.skyf import SkyF
from gleam.skypatch import SkyPatch
from gleam.lensobject import LensObject
from gleam.multilens import MultiLens
from gleam.utils import colors as glmc

# import logging

###############################################################################
class LightSampler(object):
    """
    MCMC sampler class for light profiles of galaxy lenses
    """
    def __init__(self, data, model='sersic',
                 center=None, mask=None, gain=1, priors=None,
                 verbose=False):
        """
        Set up parameter space for walkers

        Args:
            data <np.ndarray> - data to be compared to model

        Kwargs:
            model <str/gleam.model.[] object> - name of model to be used or the model directly
            center <float,float> - coordinates where to put the model's center initially
            masks <np.ndarray(bool)> - boolean mask to exclude in the model fitting
            gain <float> - gain with which the data was boosted
            priors <list> - a list of limits to limit the parameter space
            verbose <bool> - verbose mode; print command line statements

        Return:
            <LightSampler object> - standard initializer
        """
        self.data = np.asarray(data)
        self.center = center or (self.data.shape[0]//2, self.data.shape[1]//2)
        self.mask = mask
        self.gain = gain
        self.model = model
        self.mcmc_sampler = None
        self.mcmc_pos = None
        self.mcmc_state = None
        if self.mask is None:
            self.mask = np.full(self.data.shape, True)
        if verbose:
            print(self.__v__)

    @classmethod
    def from_gleamobj(cls, gleam, band='i', lens_mask=None, image_mask=None, **kwargs):
        """
        Initialize from gleamobj

        Args:
            gleam <gleam object> - contains data of a or more than one .fits files

        Kwargs:
            band <str/int> - the band to use the model on (if multiple data maps are passed)
            lens_mask <np.ndarray(bool)> - boolean masks to include in the model fitting
            image_masks <np.ndarray(bool)> - boolean masks to exclude in the model fitting
            center <float,float> - coordinates where to put the model's center initially
            gain <float> - gain with which the data was boosted
            reduce_params <bool> - only use the reduced parameter set
            priors <list> - a list of limits to limit the parameter space
            verbose <bool> - verbose mode; print command line statements

        Return:
            <LightSampler object> - initialized with gleam object
        """
        if not isinstance(gleam, (SkyF, SkyPatch, LensObject, MultiLens)):
            return
        if len(gleam.data.shape) > 2:
            if band in gleam.bands:
                gleam = gleam[band]
            else:
                gleam = gleam[0]
        data = gleam.data
        if hasattr(gleam, 'lens') and gleam.lens:
            kwargs.setdefault('center', gleam.lens.xy)
        elif hasattr(gleam, 'center') and gleam.center:
            kwargs.setdefault('center', gleam.center.xy)
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
        return ['data', 'model_cls', 'model', 'mask', 'parameters', 'fixed', 'priors',
                'par_dim', 'gain',
                'mcmc_sampler', 'mcmc_pos', 'mcmc_state']

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
    def model_cls(self):
        """
        The model with its parameters which are to be fitted to the data

        Args/Kwargs:
            None

        Return:
            model_cls <gleam.model.[] class> - the model class
        """
        if not hasattr(self, '_model_cls'):
            self._model_cls = None
        return self._model_cls

    @model_cls.setter
    def model_cls(self, model):
        """
        Set the model by name or object

        Args:
            model <str/gleam.model.[] class> - the model class by name string or direct input

        Kwargs/Return:
            None
        """
        if isinstance(model, str):
            module_name = 'gleam.model.{}'.format(model)
            cls_name = model.capitalize()
            module = __import__(module_name, fromlist=[model])
            cls = getattr(module, cls_name)
            self._model_cls = cls
        else:
            self._model_cls = model

    @property
    def model(self):
        """
        The model instance which holds the models parameters

        Args/Kwargs:
            None

        Return:
            model <gleam.model.[] object> - the model instance
        """
        if not hasattr(self, '_model'):
            self._model = None
        return self._model

    @model.setter
    def model(self, model):
        """
        The model instance created with the model name

        Args:
            model <str/gleam.model.[] class> - the model class by name string or direct input

        Kwargs/Return:
            None
        """
        if hasattr(model, '__dict__'):
            self._model_cls = model.__class__
            self._model = model
        else:
            self.model_cls = model
            kwargs = {}
            if hasattr(self, 'center'):
                kwargs['x'] = self.center[0]
                kwargs['y'] = self.center[1]
            if hasattr(self, 'data'):
                kwargs['Nx'] = self.data.shape[0]
                kwargs['Ny'] = self.data.shape[1]
            kwargs['auto_load'] = True
            this_model = self.model_cls(**kwargs)
            self._model = this_model

    @property
    def parameters(self):
        """
        The parameters to be fitted

        Args/Kwargs:
            None

        Return:
            parameters <dict> - the parameter dictionary
        """
        return {k: v for k, v in zip(self.model.parameter_keys, self.model.model_parameters)}

    @parameters.setter
    def parameters(self, parameters):
        """
        Set the parameters (only non-fixed parameters if passed as a list)

        Args:
            parameters <list/dict> - list or dictionary of new parameters for model

        Kwargs/Return:
            None
        """
        if isinstance(parameters, (tuple, list, np.ndarray)):
            # when run with python < 3.4 order is not preserved, but shouldn't matter
            pars = {}
            for k in self.model.parameter_keys:
                if k in self.fixed:
                    continue
                else:
                    pars[k] = parameters[0]
                    parameters = parameters[1:]
        elif isinstance(parameters, dict):
            for k in self.fixed:
                if k in parameters:
                    del parameters[k]
            pars = parameters
        self.model.model_parameters = pars
        self.model.calc_map()

    @property
    def fixed(self):
        """
        Fixed model parameters excluded in the parameter search

        Args/Kwargs:
            None

        Return:
            fixed <list(str)> - list of parameter keys excluded from sampling
        """
        if not hasattr(self, '_fixed'):
            self._fixed = []
        return self._fixed

    @fixed.setter
    def fixed(self, parameter_keys):
        """
        Set the list of fixed parameters
        """
        self._fixed = [k for k in parameter_keys]

    @property
    def par_dim(self):
        """
        Args/Kwargs:
            None

        Return:
            par_dim <int> - dimensions of the parameter space
        """
        return len(self.model.parameter_keys) - len(self.fixed)

    @property
    def priors(self):
        """
        Priors of the model (excluding the fixed parameters)

        Args/Kwargs:
            None

        Return:
            priors <list> - the priors of the model
        """
        return [self.model.priors[i] for i, k in enumerate(self.model.parameter_keys)
                if k not in self.fixed]

    @property
    def parspace_min(self):
        """
        Minimal values in parameter space

        Args/Kwargs:
            None

        Return:
            parspace_min <np.ndarray> - lower limits for the parameters
        """
        return np.array([p[0] for p in self.priors])

    @property
    def parspace_max(self):
        """
        Maximal values in parameter space

        Args/Kwargs:
            None

        Return:
            parspace_max <np.ndarray> - upper limits for the parameters
        """
        return np.array([p[1] for p in self.priors])

    def lnprior(self, params):
        """
        A simple prior probability function (logarithmic)

        Args:
            params <list> - point in parameter space

        Kwargs:
            None

        Return:
            lnprior <float> - prior probability for point in parameter space
        """
        # outside_parspace = False
        for i, p in enumerate(params):
            outside_parspace = self.priors[i][0] < p < self.priors[i][1]
            # outside_parspace = outside_parspace and self.priors[i][0] < p < self.priors[i][1]
            if not outside_parspace:
                return -np.inf
        return 0.0

    def lnlike(self, params):
        """
        The likelihood probability function (logarithmic)

        Args:
            params <list> - point in parameter space

        Kwargs:
            None

        Return:
            lnlike <float> - likelihood probability for point in parameter space

        Note:
            - a simple Pearson's chi^2
        """
        self.parameters = params
        return -1 * np.sum((self.model.map2D[self.mask]-self.data[self.mask])**2
                           / (self.gain*self.data[self.mask]))

    def postpdf(self, params):
        """
        The logarithmic post probability distribution function

        Args:
            params <list> - point in parameter space

        Kwargs:
            None

        Return:
            lnlike <float> - pdf probability for point in parameter space
        """
        lp = self.lnprior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(params)

    def run(self, n_walkers=1000, burn_in=50, mcmc_steps=300, threads=1,
            progressbar=True, verbose=False):
        """
        Run the MCMC parameter sampler

        Args:
            None

        Kwargs:
            n_walkers <int> - number of walkers
            burn_in <int> - number of steps for the burn-in phase
            mcmc_steps <int> - number of MCMC simulation steps
            threads <int> - numbers of threads to use during the sampling
            progressbar <bool> - if True, a progressbar is shown on the command-line
            verbose <bool> - verbose mode; print command line statements

        Return:
            sampler <EmceeProfiler object> - the MCMC sampler object
            pos, prob, state <EmceeProfiler output> - the output of the MCMC simulation
        """
        if progressbar:
            # set up progressbar
            try:
                from gleam.utils.widgets import Percentage, Bar, ETA
                from gleam.utils.progressbar import ProgressBar
                burn_in_widget = ['Burn-in...', Percentage(), Bar(marker='#'), ETA()]
                sampler_widget = ['MCMC sampling...', Percentage(), Bar(marker='#'), ETA()]
                burn_in_pbar = ProgressBar(widgets=burn_in_widget, maxval=burn_in)
                sampler_pbar = ProgressBar(widgets=sampler_widget, maxval=mcmc_steps)
            except:
                progressbar = False
                verbose = True
        # start anew or pick up from last runs end positions
        if self.mcmc_sampler is None:
            self.mcmc_sampler = emcee.EnsembleSampler(
                n_walkers, self.par_dim, self.postpdf, threads=threads)
        else:
            self.mcmc_sampler.reset()
        if self.mcmc_pos is None:
            self.mcmc_pos = [self.parspace_min + (self.parspace_max-self.parspace_min)
                             * np.random.rand(self.par_dim)
                             for i in range(n_walkers)]
        if progressbar:
            # burn in to scramble initial parspace positions
            burn_in_pbar.start()
            for i, (pos, prob, state) in enumerate(
                    self.mcmc_sampler.sample(self.mcmc_pos, iterations=burn_in)):
                burn_in_pbar.update(i)
            burn_in_pbar.finish()
        else:
            if verbose:
                print("Burn in...")
            pos, prob, state = self.mcmc_sampler.run_mcmc(self.mcmc_pos, burn_in)
        self.mcmc_sampler.reset()
        # actual MCMC run
        if progressbar:
            sampler_pbar.start()
            for i, (pos, prob, state) in enumerate(
                    self.mcmc_sampler.sample(pos, iterations=mcmc_steps)):
                sampler_pbar.update(i)
            sampler_pbar.finish()
        else:
            if verbose:
                print("MCMC sampling...")
            pos, prob, state = self.mcmc_sampler.run_mcmc(pos, mcmc_steps)
        # save and return the entire chain, end positions, probabilities, and states
        self.mcmc_pos = pos
        self.mcmc_prob = prob
        self.mcmc_state = state
        return self.mcmc_sampler, self.mcmc_pos, self.mcmc_prob, self.mcmc_state

    def ensemble_average(self, verbose=False):
        """
        Evaluate the samplers chain and extract the model parameters

        Args:
            None

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        if self.mcmc_sampler is None:
            return
        samples = self.mcmc_sampler.flatchain

        from scipy.stats import norm
        mus = []
        sigmas = []
        for i in range(samples.shape[-1]):
            mus.append(norm.fit([q for q in samples[:, i]])[0])
            sigmas.append(norm.fit([q for q in samples[:, i]])[1])
        self.parameters = mus
        self.sigmas = sigmas
        if verbose:
            print(self.parameters)
            print(self.sigmas)

    def renormalize(self, verbose=False):
        """
        Renormalize the model based on the integrated data

        Args:
            None

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        tot_model = np.sum(self.model.map2D[self.mask])
        tot_data = np.sum(self.data[self.mask])
        model_norm = self.model.map2D / tot_model
        self.model.map2D = model_norm * tot_data

    def parspace_plot(self, show=False, **kwargs):
        """
        Plots the positions of MCMC walkers in parameter space

        Args:
            None

        Kwargs:
            show <bool> - show the plot immediately

        Return:
            fig <mpl.figure.Figure object> - the figure on which the corner plot was made
        """
        import corner  # https://github.com/dfm/corner.py
        # logger = logging.getLogger()
        # logger.disabled = True  # hide annoying warning log
        labels = [k for k in self.model.parameter_keys if k not in self.fixed]
        fig = corner.corner(self.mcmc_pos, color=glmc.purpleblue, labels=labels, **kwargs)
        # logger.disabled = False
        if show:
            plt.show()
        return fig

    def plot_residuals(self, log=False, colorbar=True, scalebar=None,
                       mask=None, contours=None, show=False, **kwargs):
        """
        Plot the residuals of the model and data map once the MCMC sampler was run

        Args:
            None

        Kwargs:
            log <bool> - plot in log scale
            colorbar <bool> - plot the colorbar
            scalebar <float> - if not None, the scalebar is plotted
                               with scalebar being the pixel scale
            mask <np.ndarray(bool)> - a boolean mask where False values are set
                                      to -np.inf resulting in a transparent map
            contours <bool> - plot contours on the map
            show <bool> - display the plot
            **kwargs <dict> - matplotlib.pyplot.imshow keywords

        Return:
            fig <mpl.figure.Figure object> - the figure on which the residual plot was made
        """
        # data
        resid = self.data - self.model.get_map()
        if log:
            np.log10(resid)
        if mask:
            np.place(resid, ~self.mask, -np.inf)
        # keywords
        fig, ax = plt.subplots()
        kwargs.setdefault('alpha', 0.4)
        kwargs.setdefault('interpolation', 'none')
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('aspect', 'equal')
        kwargs.setdefault('linestyles', 'solid')
        kwargs.setdefault('clevels', 30)
        kwargs.setdefault('levels', np.linspace(np.min(resid), np.max(resid), kwargs['clevels']))
        # kwargs.setdefault('extent', [(-self.data.shape[0]-1)//2, self.data.shape[0]//2,
        #                              (-self.data.shape[1]-1)//2, self.data.shape[1]//2])
        # plotting
        if contours:
            img = ax.contourf(resid, **kwargs)
            kwargs.pop('alpha')
            ax.contour(img, **kwargs)
        else:
            kwargs.pop('levels')
            kwargs.pop('clevels')
            kwargs.pop('linestyles')
            img = ax.imshow(resid, **kwargs)
        if colorbar:
            clrbar = fig.colorbar(img)
            clrbar.outline.set_visible(False)
        if scalebar is not None:
            from matplotlib import patches
            barpos = (0.05*self.data.shape[0], 0.025*self.data.shape[1])
            w = self.data.shape[0]*0.15  # 15% of the range in x
            h = self.data.shape[1]*0.01
            rect = patches.Rectangle(barpos, w, h,
                                     facecolor='white', edgecolor=None,
                                     alpha=0.85)
            ax.add_patch(rect)
            ax.text(barpos[0]+w/4, barpos[1]+self.data.shape[1]*0.02,
                    r"$\mathrm{{{:.1f}''}}$".format(scalebar*w),
                    color='white', fontsize=16)
        ax.axis('off')
        ax.set_aspect('equal')
        if show:
            plt.show()
        return fig


if __name__ == "__main__":
    pass  # TODO
