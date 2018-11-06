#!/usr/bin/env python
"""
@author: phdenzel

Feed the glass config factory with information

Note:
   - GLSCFactory is very selective and not very advanced/flexible. To make it
     work use append_keywords/append_filter or change GLSCFactory.keywords/
     GLSCFactory.key_filter directly
   - GLSCFactory uses templates to write glass config files... an important
     part are the separator lines used to navigate inbetween sections
"""
###############################################################################
# Imports
###############################################################################
import gleam  # avoids cyclic imports with gleam.lensobject this way
from gleam.utils.makedir import mkdir_p

import sys
import os
import re
# import copy


__all__ = ['GLSCFactory']


###############################################################################
class GLSCFactory(object):
    """
    Generate glass config file's with this class
    """
    # default search parameters; extend with append GLSCFactory.append_keywords
    keywords = {'px2arcsec': ['pixelsize', 'pixel-size', 'pixel size', 'scale'],
                'zl': ['redshift', 'lens redshift', 'lens/source redshift',
                       'lens/src redshift', 'l/s redshift', 'source/lens redshift',
                       'src/lens redshift', 's/l redshift'],
                'zs': ['redshift', 'source redshift', 'lens/source redshift',
                       'lens/src redshift', 'l/s redshift', 'source/lens redshift',
                       'src/lens redshift', 's/l redshift'],
                'photzp': ['zeropoint', 'zero-point', 'zero point', 'zp'],
                'tdelay': ['time delay', 'delay', 'BCD - A', 'B - A'],
                'tderr': ['time delay', 'delay', 'BCD - A', 'B - A'],
                'double': ['B - A'],
                'quad': ['BCD - A'],
                'kext': ['external convergence', 'ext. convergence'],
                'kext_error': ['external convergence', 'ext. convergence'],
                'v_disp': ['velocity', 'dispersion'],
                'v_disp_error': ['velocity', 'dispersion']}
    # additional filter to distinguish between values and errors on the same line
    # extend with GLSCFactory.append_filter
    key_filter = {'px2arcsec': lambda lst: [min(lst), min(lst)],
                  'zl': min,
                  'zs': max,
                  'photzp': lambda lst: lst[0],
                  'tdelay': lambda lst: lst[:int(len(lst)/2)]+[0],
                  'tderr': lambda lst: (
                      lst[int(len(lst)/2):] + [sum(lst[int(len(lst)/2):])
                                               / max(1, len(lst[int(len(lst)/2):]))]),
                  'quad': lambda arg: True if arg else False,
                  'double': lambda arg: True if arg else False,
                  'kext': lambda lst: lst[:int(len(lst)/2)],
                  'kext_error': lambda lst: lst[int(len(lst)/2):],
                  'v_disp': lambda lst: lst[:int(len(lst)/2)][0],
                  'v_disp_error': lambda lst: lst[int(len(lst)/2):][0]}
    # default index labeling of the positions
    labeling = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}

    def __init__(self, parameter=None, text_file=None, text=None, filter_=True, sync=True,
                 lens_object=None, fits_file=None,
                 template_single=os.path.abspath(os.path.dirname(__file__)) \
                 + '/' + 'template.single.gls',
                 template_multi=os.path.abspath(os.path.dirname(__file__)) \
                 + '/' + 'template.multi.gls',
                 output=None, name=None, reorder=None, verbose=False, **kwargs):
        """
        Initialize a Glass Config Factory with information from .fits and/or .txt file

        Args:
            None

        Kwargs:
            parameter <dict> - parameter directly for the GLASS config generation
            text_file <str> - path to .txt file (shortcuts are automatically resolved)
            text <list(str)> - alternatively to text_file the text can be input directly
            lens_object <LensObject object> - a lens object for basic information about lens
            fits_file <str> - alternatively to a lens object, a path to .fits file
                              (shortcuts are automatically resolved) can be input
            override <bool> - change values in lens_object obtained from text
            output <str> - output name of the .gls file
            name <str> - object name in the .gls file (extracted from output by default)
            reorder <str> - reorder the image positions relative to ABCD ordered bottom-up
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        # name/path parameters
        if output is None:
            output = 'autogen'
        if output.endswith('.gls'):
            output = os.path.splitext(output)[0]
        if output.endswith('/'):
            output = output + 'autogen'
        self.directory, self.fname = os.path.split(os.path.abspath(output))
        if name is not None:
            self.name = name
        else:
            self.name = self.fname
        self._parameter = parameter
        self.parameter = {'name': "'{}'".format(self.name),
                          'fname': "'{}'".format(self.fname),
                          'dpath': "'{}'".format(""),
                          'reorder': reorder}

        # read text input
        if text is None:
            self.text = self.read(text_file)
        else:
            self.text = text

        # read lens input
        if lens_object is None:
            self.lens_object = None
            if fits_file is not None:
                self.lens_object = gleam.lensobject.LensObject(fits_file, auto=True, **kwargs)
        else:
            self.lens_object = lens_object
        if sync:
                self.sync_lens_params()

        # glass config
        self.template = {}
        self.template['single'] = self.read(template_single)
        self.template['multi'] = self.read(template_multi)
        self._config = dict(self.template)  # copy template by default

        # some verbosity
        if verbose:
            print(self.__v__)

    def __str__(self):
        return "GLSCFactory({}, {}, {}, {}...)".format(*list(self.parameter.keys())[:4])

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of GLSCFactory attributes
        """
        tests = ['lens_object', 'text', 'directory', 'name', 'parameter',
                 'template', 'config']
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t)) for t in tests])

    @property
    def parameter(self):
        """
        Parameter gathered from text and lens matching search parameters

        Args/Kwargs:
            None

        Return:
            parameter <dict> - a dictionary of information from text and lens
        """
        if self._parameter is None:
            self._parameter = dict()
        text_info = self.text_extract(self.text, filter_=True)
        lens_info = self.lens_extract(self.lens_object)
        self._parameter.update(text_info)
        self._parameter.update(lens_info)
        return self._parameter

    @parameter.setter
    def parameter(self, parameter):
        """
        Update parameter extracted from both text and lens by search parameters

        Args:
            parameter <dict> - a dictionary of information intended for glass config file

        Kwargs/Return:
            None
        """
        if self._parameter is None:
            self._parameter = dict()
        self._parameter.update(parameter)

    @property
    def config(self):
        """
        Configurations based on text and lens parameter for the glass config file

        Args/Kwargs:
            None

        Return:
           config <list(str)> - configurations ready to be written to a file
        """
        if self._config is None:  # only copy if necessary
            self._config = dict(self.template)
        parameter = self.parameter  # update parameter by calling property
        # fill configs with parameter
        parameter_reordered = False
        for k in self._config.keys():
            for i, line in enumerate(self._config[k]):
                param = line[1:].split("=")[0].strip()
                if line[0] == "_" and param in parameter:
                    if param == 'ABCD' and parameter['reorder'] is not None and not parameter_reordered:
                        topbottom = list(parameter['ABCD'])  # not yet ABCD
                        parameter_reordered = True
                        for j, x in enumerate(parameter['reorder']):
                            parameter['ABCD'][GLSCFactory.labeling[x]] = topbottom[j]
                    self._config[k][i] = param+" = "+str(parameter[param])+"\n"
        return self._config

    @staticmethod
    def append_keywords(keywords, verbose=False):
        """
        Append key/words to search parameters of GLSCFactory
        (one to many mapping of key and words)

        Args:
            keywords <dict> - additional search parameters to append to defaults

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            GLSCFactory.keywords <dict> - search parameters of the class
        """
        if keywords is not None and isinstance(keywords, dict):
            for k, w in keywords.items():
                GLSCFactory.keywords[k] = w
        if verbose:
            print(GLSCFactory.keywords)
        return GLSCFactory.keywords

    @staticmethod
    def append_filter(filters, verbose=False):
        """
        Append keywords to search parameters of GLSCFactory (one to one mapping of key and func)

        Args:
            key <dict(str, callable)> - variable key of search matches on which
                                        the filter is applied

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            GLSCfactory.key_filter <dict> - filter functions to apply to search matches
        """
        if filters is not None and isinstance(filters, dict):
            for k, f in filters.items():
                print(k, f)
                GLSCFactory.key_filter[k] = f
        if verbose:
            print(GLSCFactory.key_filter)
        return GLSCFactory.key_filter

    @staticmethod
    def read(filepath, check_ext=False, verbose=False):
        """
        Read a text file and return text

        Args:
            filepath <str> - path to .txt file (shortcuts are automatically resolved)

        Kwargs:
            check_ext <bool> - check path for .txt extension
            verbose <bool> - verbose mode; print command line statements

        Return:
            text <list(str)> - text from the .txt file
        """
        # validate input
        if filepath is None:
            return None
        if not isinstance(filepath, str):
            raise TypeError("Input path needs to be string")
        # expand shortcuts
        if '~' in filepath:
            filepath = os.path.expanduser(filepath)
        dirname = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(filepath):
            filepath = "/".join([dirname, filepath])
        # check if path exists
        if not os.path.exists(filepath):
            try:  # python 3
                FileNotFoundError
            except NameError:  # python 2
                FileNotFoundError = IOError
            raise FileNotFoundError("'{}' does not exist".format(filepath))
        # optionally check extension
        if check_ext and True not in [filepath.endswith(ext) for ext in ('.text', '.txt')]:
            raise ValueError('Input file path need to be a .txt file')
        f = open(filepath, 'r')
        text = f.readlines()
        f.close()
        if verbose:
            print(text)
        return text

    @staticmethod
    def text_extract(text, keywords=None, filter_=False, key_filter=None, verbose=False):
        """
        Extract GLSCFactory's search parameters from text

        Args:
            text <list(str)> - list of strings in which the search parameters take effect

        Kwargs:
            keywords <dict> - additional strings to append to search parameters
            key_filter <dict> - additional functions to append to key filters
            filter_ <bool> - apply filters from GLSCFactory.key_filter
            verbose <bool> - verbose mode; print command line statements

        Return:
            info <dict> - dictionary of extracted information from the text
        """
        info = {}
        if text is None:
            return {}
        # get all search parameters
        GLSCFactory.append_keywords(keywords)
        keys = list(GLSCFactory.keywords.keys())
        key_groups = list(GLSCFactory.keywords.values())
        words = sum(GLSCFactory.keywords.values(), [])
        params = [re.compile(s, re.I) for s in words]  # case-insensitive search parameters
        # search for parameters
        for i, line in enumerate(text):
            hits = [p.search(line).group() if p.search(line) else None for p in params]
            idx_word = [n for n, h in enumerate(hits) if h]
            idx_keyg = [idx for w in idx_word for idx, g in enumerate(key_groups) if words[w] in g]
            if line.strip()[-1] == ":":
                numbers = [float(n) for n in re.findall(
                    "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", text[i+1])]
            else:
                numbers = [float(n) for n in re.findall(
                    "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)]
            for i in idx_keyg:
                info[keys[i]] = numbers
        # run filter through finds
        if filter_:
            GLSCFactory.append_filter(key_filter)
            for k in info:
                if GLSCFactory.key_filter[k]:
                    info[k] = GLSCFactory.key_filter[k](info[k])
        if verbose:
            print(info)
        return info

    def sync_lens_params(self, verbose=False):
        """
        Override attributes of the lens object with info from text parameters

        Args:
            None

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        for k in self.parameter.keys():
            if k in dir(self.lens_object):
                self.lens_object.__setattr__(k, self.parameter[k])
                if verbose:
                    print(self.lens_object.__getattribute__(k))

    @staticmethod
    def lens_extract(lo, directory=None, quad=True, double=False, output=None, verbose=False):
        """
        Extract GLSCFactory's search parameters from a LensObject object

        Args:
            lo <LensObject object> - holding information about the .fits file

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            info <dict> - dictionary of extracted information from SkyF object
        """
        info = dict()
        if directory is not None:
            info.update({'dpath': "'{}'".format(directory)})
        # positions
        ABCD = lo.src_shifts(unit='arcsec')
        # parity
        if len(lo.srcimgs) == 4:
            parity = ['min', 'min', 'sad', 'sad']
        elif len(lo.srcimgs) == 2:
            parity = ['min', 'sad']
        else:
            ABCD = [ABCD[i] if len(ABCD) > i else [] for i in range(4)]
            parity = ['min', 'min', 'sad', 'sad']
        # gather info and return it
        info.update({'ABCD': ABCD, 'parity': parity})
        if verbose:
            print(info)
        return info

    def write(self, filename=None, verbose=False):
        """
        Write glass configs to a new file

        Args:
            None

        Kwargs:
            filename <str> - save in a different location as given in output
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        if filename is not None:
            if '~' in filename:
                filename = os.path.expanduser(filename)
            self.directory, self.fname = os.path.split(os.path.abspath(filename))
        mkdir_p(self.directory)
        output = self.directory+"/"+self.fname
        if not self.fname.endswith(".gls"):
            output = output + ".gls"
        with open(output, 'w') as f:
            f.writelines(self.config['single'])
        if verbose:
            print('Writing configs to {}'.format(output))

    def append(self, filename=None, multi=True, last=False, verbose=False):
        """
        Append only lens specific configs to an already existing file

        Args:
            None

        Kwargs:
            filename <str> - save in a different location as given in output
            multi <bool> - use multi template instead of single template
            last <bool> - complete file by attaching the last 4 lines
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        from gleam.utils.makedir import mkdir_p
        if filename is not None:
            if '~' in filename:
                filename = os.path.expanduser(filename)
            self.directory, self.fname = os.path.split(os.path.abspath(filename))
        mkdir_p(self.directory)
        output = self.directory+"/"+self.fname
        if not self.fname.endswith(".gls"):
            output = output + ".gls"
        lens_config = self.config['multi'] if multi else self.config['single']
        config_sections = [(i, line) for i, line in enumerate(lens_config)
                           if line.startswith('###')]
        if os.path.exists(output):
            lens_section = [config_sections[j-1][0] if l.startswith('### Lens')
                            else config_sections[j+1][0] if l.startswith('### Source')
                            else None
                            for j, (i, l) in enumerate(config_sections)]
            start, end = [i for i in lens_section if isinstance(i, int)]
        else:
            start, end = 0, config_sections[-1][0]
        if os.path.exists(output) and verbose:
            print("Appending configs to {}".format(output))
        elif verbose:
            print("Writing configs to {}".format(output))
        if last:
            print("Completing configs at {}".format(output))
        with open(output, 'a+') as f:
            f.writelines(lens_config[start:end])
            if last:
                f.writelines(lens_config[end:])

    @staticmethod
    def config_diff(*glsc):
        """
        Determine diff lines in gls configurations
        """
        NotImplemented  # yet


# MAIN FUNCTION ###############################################################
def main(case, args):
    pass


def parse_arguments():
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    # main args
    parser.add_argument("case", nargs='*',
                        help="Path input to .fits file for skyf to use",
                        default=os.path.abspath(os.path.dirname(__file__)) + '/test' \
                        + '/W3+3-2.U.12907_13034_7446_7573.fits')
    parser.add_argument("-z", "--redshifts", metavar=("<zl", "zs>"), nargs=2, type=float,
                        help="Redshifts for lens and source")

    # gls config factory args
    parser.add_argument("--single-config", dest="config_single", metavar="<output-name>", type=str,
                        help="Generate a single-lens glass config file")
    parser.add_argument("--multi-config", dest="config_multi", metavar="<output-name>", type=str,
                        help="Generate a glass config file in append-mode")
    parser.add_argument("--name", dest="name", metavar="<name>", type=str,
                        help="Name of the lens object in the glass config file")
    parser.add_argument("--finish", dest="finish_config", action="store_true",
                        help="Append and complete the config file with these configs"
                        + " in multi-config mode",
                        default=False)
    parser.add_argument("--text-file", dest="text_file", metavar="<path-to-file>", type=str,
                        help="Path to text file with additional info for glass config generation",
                        default=os.path.abspath(os.path.dirname(__file__)) + "/test" \
                        + "/test_lensinfo.txt")
    parser.add_argument("--filter", dest="filter_", action="store_true",
                        help="Use GLSCFactory's additional filter for extracted text info",
                        default=False)
    parser.add_argument("--reorder", dest="reorder", metavar="<abcd-order>", type=str.upper,
                        help="Reorder the image positions relative to ABCD ordered bottom-up")

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


if __name__ == '__main__':
    parser, case, args = parse_arguments()
    no_input = len(sys.argv) <= 1 and os.path.abspath(os.path.dirname(__file__))+'/test/' in case
    if no_input:
        parser.print_help()
    elif args.test_mode:
        sys.argv = sys.argv[:1]
        from gleam.test.test_glscfactory import TestGLSCFactory
        TestGLSCFactory.main()
    else:
        main(case, args)
