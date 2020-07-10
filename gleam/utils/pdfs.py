import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, newton


def lorentzian(x, mu, gamma, I):
    """
    Lorentzian/Cauchy probability distribution function (longer wings compared to Gaussian)
    """
    return I/(np.pi*gamma) * gamma*gamma/((x - mu)**2 + gamma*gamma)

def gaussian(x, mu, sigma, I):
    """
    Gaussian probability distribution function (no explanation needed)
    """
    return I/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(mu - x)**2/sigma**2)

def pseudovoigt(x, x0, sigma, gamma, eta, I):
    """
    Pseudo-Voigt distribution function (weighting between Lorentzian and Gaussian)
    """
    g = np.exp(0.5*(x-x0)**2/sigma**2)
    l = gamma*gamma / ((x-x0)**2 + gamma*gamma)
    return I*(eta*g + (1-eta)*l)

def tukeygh(x, x0, g, h, I):
    """
    Tukey g-h distribution function (provides flexibility and asymmetric wings)

    Note:
        - g=h=0: Gaussian
        - positive g lifts right wing
        - negative g lifts left wing
        - positive h lifts both wings
    """
    lo, hi = -8, 8  # -np.max(np.abs(x)), np.max(np.abs(x))
    if not isinstance(x, (tuple, list, np.ndarray)):
        x = [x]
    u = []
    for xd in x:
        def f(xi):
            res = (np.exp(abs(g)*xi)-1) * np.exp(0.5*h*xi*xi) - g*(xd-x0)
            # if np.isnan(res):
            #     return 0
            return res
        ui = brentq(f, lo, hi)
        u.append(ui)
    u = np.asarray(u)
    return np.exp(-0.5*u*u)* I



class Measure(object):
    def __init__(self, name, mean, std=None):
        self.name = name
        self.mean = mean
        if isinstance(std, (int, float)):
            std = (-std, std)
        self.std = std
        
    def __call__(self, y=0):
        measure = [self.sig_lims[0], self.mean, self.sig_lims[1]]
        return measure, [y for i in measure]
    
    @property
    def sig_lims(self):
        return self.mean+self.std[0], self.mean+self.std[1]
    
    def plot(self, y=0, transform=None, color=None, msize=4, xshift=0, yshift=0, lw=3, fontsize=18):
        if transform is None:
            transform = lambda x: x
        estimate, yvals = self(y)
        estimate = transform(estimate)
        plt.scatter(transform(self.mean), y, s=msize, color=color)
        plt.plot(estimate, yvals, lw=lw, color=color)
        plt.text(transform(self.mean+xshift), y+yshift, self.name, color='black',
                 fontsize=fontsize, horizontalalignment='center')
        
    def span_monocolor(self, ax=None, Nbins=50, transform=None, color='black', alpha=1.0):
        if ax is None:
            ax = plt.gca()
        if transform is None:
            transform = lambda x: x
        ax.axvline(transform(self.mean), color=color, alpha=0.5*alpha, lw=0.5, zorder=0)
        lims = transform(np.array(self.sig_lims))
        for i in range(Nbins):
            step = np.diff(lims) * i/float(2*Nbins)
            xlims = (lims[0] + step), (lims[1] - step)
            ax.axvspan(*xlims, color=color, alpha=alpha/Nbins, zorder=0, lw=0)
        ax.axvspan(-1, -0.99, color=color, label=self.name, alpha=0.75, zorder=0, lw=0)
        ax.legend(loc='upper right', fontsize=14, numpoints=1, borderpad=0.3)
        
    def span_multicolor(self, ax=None, Nbins=50, transform=None,
                        cmap=None, alpha=1.0, alpha_grad=True):
        if ax is None:
            ax = plt.gca()
        if transform is None:
            transform = lambda x: x
        if cmap is None:
            cmap = plt.cm.get_cmap('phoenix')
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        ax.axvline(transform(self.mean), color='black', alpha=0.5*alpha, lw=0.5, zorder=0)
        mu = transform(self.mean)
        sigs = transform(np.array(self.sig_lims))
        sigma = 0.5 * np.sum(np.abs(mu - sigs))
        xbins = np.linspace(sigs[0], sigs[1], Nbins)
        dist = np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(mu-(0.5*(xbins[:-1]+xbins[1:])))**2/sigma**2)
        dist = (dist-dist.min())
        dist = dist / dist.max()
        for i, (xl, xr) in enumerate(zip(xbins, xbins[1:])):
            color = cmap(dist[i])
            if alpha_grad:
                alpha = dist[i]
            ax.axvspan(xl, xr, color=color, alpha=alpha, zorder=0, lw=0)
        # ax.legend(loc='upper right', fontsize=14, numpoints=1, borderpad=0.3)



if __name__ == "__main__":

    x = np.linspace(1, 10, 100)
    y = 0*x

    for n in range(5):
        g = 2*np.random.random() - 1
        h = np.random.random()
        x0 = 9*np.random.random() + 1
        I = 10*np.random.random() + 1
        y = tukeygh(x, x0, g, h, I)
        plt.plot(x, y)[0].set_label('x0 = %5.2f g = %5.2f h = %4.2f I = %5.2f' % (x0, g, h, I))
    plt.legend()
    plt.show()
