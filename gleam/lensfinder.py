#!/usr/bin/env python
"""
@author: phdenzel

Climb every peak in search for lens and source candidates

TODO:
    - add a main method
    - complete tests
"""
###############################################################################
# Imports
###############################################################################
import sys
import os
import numpy as np
import scipy.ndimage as ndimg

from gleam.skycoords import SkyCoords


__all__ = ['LensFinder']


###############################################################################
class LensFinder(object):
    """
    Framework for finding peaks in the sky (.fits files)
    """
    def __init__(self, lensobject, n=5, min_q=0.1, sigma=(4, 4), centroid=5, verbose=False):
        """
        Initialize peak finding in a lensobject

        Args:
            lensobject <LensObject object> - a LensObject instance with the .fits file's data

        Kwargs:
            n <int> - number of peak candidates allowed
            min_q <float> - a percentage quotient for the minimal peak separation
            sigma <int(,int)> - lower/upper sigma factor for signal-to-noise estimate
            centroid <int> - use COM positions around a pixel slice of size of centroid
                             around peak center if centroid > 1
            verbose <bool> - verbose mode; print command line statements

        Return:
            <LensFinder object> - standard initializer for LensFinder
        """
        self.lensobject = lensobject
        self.n = n
        self.min_q = min_q  # peak separation % of the whole image
        self.min_d = (self.min_q*self.lensobject.naxis1, self.min_q*self.lensobject.naxis2)
        self.sigma = sigma
        self.threshold = None
        self.peak_positions, self.peak_values = [], []
        self.peaks = [SkyCoords.empty()]
        self.lens_candidate, self.lens_value, self.lens_index = None, None, None
        self.source_candidates, self.source_values, self.source_indices = [], [], []
        if self.lensobject.data is not None:
            # estimate threshold
            self.threshold = self.threshold_estimate(self.lensobject.data, sigma=self.sigma)
            # find peaks
            self.peak_positions, self.peak_values = self.peak_candidates(
                self.lensobject.data, self.threshold, n=self.n, min_d=self.min_d,
                centroid=centroid)
            self.peaks = [self.lensobject.p2skycoords(p, unit='pixel', relative=False) for p
                          in self.peak_positions]
            # choose lens in peaks
            self.lens_candidate, self.lens_value, self.lens_index = self.detect_lens(
                self.peaks, self.peak_values)
            # sources are all the rest
            self.source_candidates, self.source_values, self.source_indices = (
                [self.peaks[i] for i in range(len(self.peaks)) if i != self.lens_index],
                [self.peak_values[i] for i in range(len(self.peak_values))
                 if i != self.lens_index],
                [i for i in range(len(self.peak_positions)) if i != self.lens_index])
            # if self.source_candidates is not None and self.lens_candidate is not None:
            #     self.order_by_distance(self.source_candidates, self.lens_candidate)
        # some verbosity
        if verbose:
            print(self.__v__)

    def __str__(self):
        try:
            s = u"\u03C3".encode("utf-8") + ""
        except TypeError:
            s = u"\u03C3"
        return "LensFinder(peaks({}/{}):".format(len(self.peaks), self.n) \
            + "{}".format(self.sigma) + s + ")"

    def __repr__(self):
        return self.__str__()

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of LensFinder attributes
        """
        tests = ['n', 'min_q', 'min_d', 'sigma',  'threshold',
                 'peaks', 'peak_positions', 'peak_values']
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in tests])

    @staticmethod
    def threshold_estimate(data, snr=None, centf=np.ma.median, stdf=np.std, sigma=(3, 3),
                           iterations=None, verbose=False):
        """
        Automated threshold estimator for the peak finding algorithm; estimates pixel-wise
        thresholds by filtering out values above a few sigmas above the centroid point

        Args:
            data <np.ndarray> - data to find peaks in

        Kwargs:
            snr <float> - signal-to-noise ratio
            centf <func> - centroid function (default: np.median)
            stdf <func> - standard deviation function (default: np.std)
            sigma <int(,int)> - lower and upper sigma factor for sigma clipping
            verbose <bool> - verbose mode; print command line statements

        Return:
            threshold <float> - threshold ready to use in a peak finding algorithm
        """
        def clip(filtered_data, lower_sig, upper_sig):
            # get stats
            median = centf(filtered_data)
            std = stdf(filtered_data)
            min_value = median - std*lower_sig
            max_value = median + std*upper_sig
            if max_value is np.ma.masked:
                max_value = np.ma.MaskedArray(np.nan, mask=True)
                min_value = np.ma.MaskedArray(np.nan, mask=True)
            # ... and clip above and below sigma limits
            filtered_data.mask |= filtered_data > max_value
            filtered_data.mask |= filtered_data < min_value
            return filtered_data

        # copy data for filtering
        filtered_data = np.ma.array(data, copy=True)
        if filtered_data.size == 0:
            return filtered_data
        # a few input checks
        if snr is None:
            snr = np.max(data)/np.std(data)
        lastrej = 0
        if iterations is None:
            lastrej = filtered_data.count()+1
        if isinstance(sigma, (tuple, list)):
            lower_sig = sigma[0]
            upper_sig = sigma[1]
        elif isinstance(sigma, (int, float)):
            lower_sig = sigma
            upper_sig = sigma
        else:
            lower_sig = upper_sig = 3
        # filter data, i.e. perform sigma clip
        if lastrej > 0:
            while filtered_data.count() != lastrej:
                lastrej = filtered_data.count()
                filtered_data = clip(filtered_data, lower_sig, upper_sig)
        else:
            for i in range(iterations):
                filtered_data = clip(filtered_data, lower_sig, upper_sig)
        # prevent mask False (scalar) if no values are clipped
        if filtered_data.mask.shape == ():
            filtered_data.mask = False  # make .mask shape match .data shape
        # get stats from filtered data
        mean = np.ma.mean(filtered_data)
        # median = np.ma.median(filtered_data)
        std = np.ma.std(filtered_data)
        bkgrd = np.zeros_like(data) + mean
        error = np.zeros_like(data) + std
        if verbose:
            print(snr)
            print(bkgrd + (error*snr))
        return bkgrd + (error*snr)

    @staticmethod
    def peak_candidates(data, threshold, min_d=3, n=1e12, mask_region=None, ignore_border=True,
                        centroid=0, verbose=False):
        """
        Recover peak candidate positions (in pixels) and their flux values above a threshold

        Args:
            data <np.ndarray> - data to find peaks in
            threshold <float> - threshold above which peaks are valid

        Kwargs:
            min_d <float>/<float,float> - minimal distance to next peak for each coordinate
            n <int> - number of peak candidates allowed
            mask_region <np.ndarray(bool)> - masked region in which peaks are ignored
            ignore_border <bool> - ignore candidates closer than 10% of the whole image to edge
            centroid <int> - use COM positions around a slice of size of centroid if centroid > 1
            verbose <bool> - verbose mode; print command line statements

        Return:
            peaks, peak_vals <list(int,int)>, <list(float)> - peak positions and values
        """
        maxima = ndimg.maximum_filter(data, size=min_d, mode='constant', cval=0.0)
        candidate_mask = (data == maxima)
        # if there is a mask
        if mask_region is not None:
            mask = np.asanyarray(mask_region)
            if data.shape != mask.shape:
                raise ValueError('Region mask must have same dimensions as data')
            candidate_mask = np.logical_and(candidate_mask, ~mask)
        # ignore peaks close to the image's border
        if ignore_border:
            border = int(0.1*data.shape[0])
            for i in range(candidate_mask.ndim):
                candidate_mask = candidate_mask.swapaxes(0, i)
                candidate_mask[:border] = False
                candidate_mask[-border:] = False
                candidate_mask = candidate_mask.swapaxes(0, i)
        # get candidates above threshold
        candidate_mask = np.logical_and(candidate_mask, (data > threshold))
        y_peaks, x_peaks = candidate_mask.nonzero()
        peak_vals = data[y_peaks, x_peaks]
        if len(x_peaks) > n:
            # sort by values and only select highest
            highest = np.argsort(peak_vals)[::-1][:n]
            x_peaks, y_peaks = x_peaks[highest], y_peaks[highest]
            peak_vals = peak_vals[highest]
            # re-sort in y again
            y_sort = np.argsort(y_peaks)
            x_peaks, y_peaks = x_peaks[y_sort], y_peaks[y_sort]
            peak_vals = peak_vals[y_sort]
        peaks, vals = [(x, y) for x, y in zip(x_peaks, y_peaks)], [v for v in peak_vals]
        # centroiding
        if centroid > 1:
            c = int(centroid//2)
            for i, (px, py) in enumerate(peaks):
                cindex = ndimg.measurements.center_of_mass(data[py-c:py+c+1, px-c:px+c+1])
                peaks[i] = [px+(cindex[1]-c), py+(cindex[0]-c)]
        # some verbosity
        if verbose:
            print(peaks)
            print(vals)
        return peaks, vals

    @staticmethod
    def detect_lens(peaks, peak_values, method='distances', verbose=False):
        """
        Find lens by comparing distances between peaks
        (i.e. if distances to a point are similar it might be in the center around a ring)

        Args:
            peaks <list(SkyCoords object)> - the list of the peak coordinates
            peak_values <list(float)> - the list of values at the peak coordinates

        Kwargs:
            method <str> - method of lens detection (distances)
            verbose <bool> - verbose mode; print command line statements

        Return:
            lens_candidate <SkyCoords object> - lens candidate's sky coordinate
            lens_position <int,int> - pixel position of the lens candidate
            lens_index <int> - index in the list of peaks

        TODO:
            - method is too crude, maybe use
                o peak value comparison
                o some kind of peak number analysis
                o some manual input like dual/quad/...
        """
        if method == 'distances':
            deviations = []
            for p in peaks:
                dist = [p.deg2arcsec(p.distance(s))[0] for s in peaks]
                if len(dist) <= 1:
                    return None, None, None
                if len(dist) < 2:
                    mean = dist[0]
                else:
                    mean = sum(dist)/(len(dist)-1)
                devs = [abs(d-mean) if d != 0.0 else d for d in dist]
                deviations.append(sum(devs))
            lens_index = min(range(len(deviations)), key=deviations.__getitem__)
        if verbose:
            print(peaks[lens_index], peak_values[lens_index], lens_index)
        return peaks[lens_index], peak_values[lens_index], lens_index

    @staticmethod
    def order_by_distance(sources, lens, verbose=False):
        """
        Order source image coordinates by radial distance to lens

        Args:
            sources <list> - list of source coordinates
            lens <SkyCoord> - coordinate of the lens

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            ordered <list> - ordered source coordinate list
        """
        dist = [s.distance(lens, from_pixels=True) for s in sources]
        order = np.argsort(dist)[::-1]
        if verbose:
            print([sources[i] for i in order], order)
        return [sources[i] for i in order], order

    @staticmethod
    def relative_positions(sources, lens, unit='arcsec', verbose=False):
        """
        Recover relative positions from peaks to the lens

        Args:
            sources <list(SkyCoords object)> - a list of peak coordinates
            lens <SkyCoords object> - the coordinate to which the positions relate

        Kwargs:
            unit <str> - unit of the shift (arcsec, degree, pixel)
            verbose <bool> - verbose mode; print command line statements

        Return:
            shift <list> - the rectangular shift (in arcsec by default)
        """
        positions = [s.get_shift_to(lens, unit=unit) for s in sources]
        if verbose:
            print(positions)
        return positions


# MAIN FUNCTION ###############################################################
def main(*args):
    pass


def parse_arguments():
    """
    Parse command line arguments
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    parser.add_argument("case", nargs='?',
                        help="Path input to .fits file for lensfinder",
                        default=os.path.abspath(os.path.dirname(__file__)) \
                        + '/test' + '/W3+3-2.I.12907_13034_7446_7573.fits')
    parser.add_argument("-n", "--npeaks", dest="n", metavar="<N>", type=int,
                        help="Number of peaks the automatic image recognition is supposed to find",
                        default=5)
    parser.add_argument("-q", "--min-q", dest="min_q", metavar="<min_q>", type=float,
                        help="A percentage quotient for the minimal peak separation",
                        default=0.1)
    parser.add_argument("--sigma", metavar=("<sx", "sy>"), dest="sigma", nargs=2, type=float,
                        help="Lower/upper sigma factor for signal-to-noise estimation",
                        default=(2, 2))

    # mode args
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Run program in verbose mode",
                        default=False)
    parser.add_argument("-t", "--test", "--test-mode", dest="test_mode", action="store_true",
                        help="Run program in testing mode",
                        default=False)

    args = parser.parse_args()
    case = args.case
    delattr(args, 'case')
    return parser, case, args


###############################################################################
if __name__ == '__main__':
    parser, case, args = parse_arguments()
    no_input = len(sys.argv) <= 1 and os.path.abspath(os.path.dirname(__file__))+'/test/' in case
    if no_input:
        parser.print_help()
    elif args.test_mode:
        sys.argv = sys.argv[:1]
        from gleam.test.test_lensfinder import TestLensFinder
        TestLensFinder.main()
    else:
        main(case, args)
