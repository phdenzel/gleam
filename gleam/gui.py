#!/usr/bin/env python
"""
@author: phdenzel

An app for some gleamy interactive activity
"""
###############################################################################
# Imports
###############################################################################
import sys
import os
import traceback
import matplotlib
matplotlib.use('TkAgg')  # needs to come before any other matplotlib imports
if sys.version_info.major < 3:
    import Tkinter as tk
    # import ttk
elif sys.version_info.major == 3:
    import tkinter as tk
    # import tkinter.ttk as ttk
else:
    raise ImportError("Could not import Tkinter")
import __init__
from gleam.app.prototype import FramePrototype
from gleam.app.menubar import Menubar
from gleam.app.navbar import Navbar
from gleam.app.toolbar import Toolbar
from gleam.app.window import Window
from gleam.app.statusbar import Statusbar
from gleam.multilens import MultiLens


# MAIN APP CLASS ##############################################################
class App(FramePrototype):
    """
    Main frame for the app
    """
    def __init__(self, master, filepath, cell_size=None, display_off=False,
                 verbose=False, *args, **kwargs):
        """
        Initialize all components of the app

        Args:
            master <Tk object> - master root of the frame
            filepath <str> - file path to the directory where all .fits files reside

        Kwargs:
            cell_size <>
            verbose <bool> - verbose mode; print command line statements

        Return:
            <App object> - standard initializer
        """
        FramePrototype.__init__(self, master, root=master, *args, **kwargs)
        self.master.title('App')
        self.master.protocol('WM_DELETE_WINDOW', self._on_close)
        # add lens info
        self.lens_patch = MultiLens(filepath, auto=False)
        self.env.add(self.env, self.lens_patch, self, itemname='lens_patch')

        # choose good window default size
        if cell_size is None:
            # default configuration
            cols = 3
            rows = self.lens_patch.N//3
            # 67% of the screen width and 89% of the screen height
            cw = int(0.6666*self.master.winfo_screenwidth()//cols // 100 * 100)
            ch = int(0.8888*self.master.winfo_screenheight()//rows // 100 * 100)
            cell_size = (min(cw, ch), min(cw, ch))

        # initialize app components
        self.menubar = Menubar(self)
        self.navbar = Navbar(self)
        self.toolbar = Toolbar(self)
        self.statusbar = Statusbar(self)
        self.window = Window(self, cell_size=cell_size)

        # pack all frames with the grid geometry manager
        self.menubar.grid(columnspan=2, row=0, sticky=tk.NE)
        self.navbar.grid(columnspan=2, row=1, sticky=tk.NE)
        self.toolbar.grid(column=0, row=2, columnspan=1, sticky=tk.W)
        self.window.grid(column=1, row=2, columnspan=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.statusbar.grid(columnspan=2, row=3, sticky=tk.E+tk.S+tk.W)
        # adding resize configs for window at (1, 2)
        self.grid(sticky=tk.N+tk.S+tk.E+tk.W)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        # TODO
        self.statusbar.log('In development...')

        if not display_off:
            self.project(self.lens_patch.image_patch(cmap='gnuplot2'))

        if verbose:
            print(self.__v__)

    @classmethod
    def init(cls, files=[], verbose=False, **kwargs):
        """
        From files initialize an App instance together with it's tk root

        Args:
            None

        Kwargs:
            files <list(str)> - .fits files for the app to read
            verbose <bool> -  verbose mode; print command line statements

        Return:
            root <Tk object> - the master Tk object
            app <App object> - the app frame instance packed with GLEAM features
        """
        root = tk.Tk()  # init master root
        app = App(root, files, verbose=False, **kwargs)
        return root, app

    @property
    def tests(self):
        """
        A list of attributes being tested when calling __v__

        Args/Kwargs:
            None

        Return:
            tests <list(str)> - a list of test variable strings
        """
        return ['master', 'lens_patch',
                'navbar', 'statusbar', 'menubar', 'toolbar', 'window']

    def project(self, images, verbose=False):
        """
        Project images directly onto the canvas

        Args:
            images <list(PIL.Image object)> - images to be added to the buffer

        Kwargs:
            size <int,int> - change the images' size while projecting
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        for i, img in enumerate(images):
            self.window.canvas.add_image(img, i)

    def display(self, term_mode=False, verbose=False):
        """
        Wrapper for mainloop

        Args:
            None

        Kwargs:
            term_mode <bool> - display the app in terminal mode
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        if term_mode:
            self.after(100, self.term_shell)
        if verbose:
            print(self.__v__)
        self.mainloop()        

    def term_shell(self, verbose=False):
        """
        Use as callback during mainloop to exit via command line

        Args:
            None

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        if sys.version_info.major < 3:
            user_input = raw_input(">> ")
        else:
            user_input = input(">>> ")
        if user_input in ["exit", "exit()", "q", "quit", "quit()"]:
            if verbose:
                print("Quitting the app")
            self.master.quit()
        else:
            print(self.statusbar.log_history)
            self.after(100, self.term_shell)

    def _on_close(self):
        """
        Execute when window is closed
        """
        self.master.quit()
        sys.exit(1)


# MAIN FUNCTION ###############################################################
def main(case, args):
    """
    Start the app
    """
    root, app = App.init(case)
    app.display(term_mode=args.term_mode, verbose=args.verbose)


def parse_arguments():
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    # main args
    parser.add_argument("case", nargs='?',
                        help="Path input to .fits file for the app to use",
                        default=os.path.abspath(os.path.dirname(__file__))+'/test')

    # mode args
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Run program in verbose mode",
                        default=False)
    parser.add_argument("--term", dest="term_mode", action="store_true",
                        help="Run program in terminal mode",
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
    no_input = len(sys.argv) <= 1 and os.path.abspath(os.path.dirname(__file__))+'/test' in case
    if no_input:
        parser.print_help()
    elif args.test_mode:
        sys.argv = sys.argv[:1]
        from gleam.test.test_app import TestApp
        TestApp.main()
    else:
        main(case, args)
