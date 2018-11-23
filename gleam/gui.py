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
import matplotlib as mpl
mpl.use('TkAgg')  # needs to come before any other matplotlib imports
if sys.version_info.major < 3:
    import Tkinter as tk
    # import ttk
    import tkFileDialog as filedialog
elif sys.version_info.major == 3:
    import tkinter as tk
    # import tkinter.ttk as ttk
    import tkinter.filedialog as filedialog
else:
    raise ImportError("Could not import Tkinter")
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
        # self.master.wm_iconbitmap()

        # add lens framework
        self.lens_patch = MultiLens(filepath, auto=False)
        self.env.add(self.env, self.lens_patch, self, itemname='lens_patch')

        # initialize app components
        self.menubar = Menubar(self)
        self.navbar = Navbar(self)
        self.toolbar = Toolbar(self)
        self.statusbar = Statusbar(self)
        self.window = Window(self)

        # state attributes
        self.selection_mode = None
        self.colormap = 'gnuplot2'

        # menubar
        self.menubar.main_labels = ['file', 'edit', 'view', 'help']
        self.menubar.labels = [['open...', 'save as...', 'exit'],
                               ['colormap'],
                               ['navbar', 'toolbar', 'statusbar'],
                               ['help', 'about']]
        self.menubar.mk_checklist('colormap',
                                  sub_labels=list(mpl.cm.datad.keys()),
                                  variable=self._colormap, command=self._on_colormap)
        self.menubar.shortcuts = [[u'\u2303F', u'\u2303S', u'\u2303Q'],
                                  [''],
                                  [u'\u2303\u2325N', u'\u2303\u2325T', u'\u2303\u2325S'],
                                  [u'\u2303H', '']]
        self.menubar.bindings = [[self._on_open, self._on_save_as, self._on_close],
                                 [],
                                 [self.menubar.dummy_func, self.menubar.dummy_func, self.menubar.dummy_func],
                                 [self._on_help, self._on_about], ]
        self.menubar.rebuild()

        # navbar

        # toolbar
        self.toolbar.sections = ['data', 'selection', ' ']
        self.toolbar.add_labels('data', [['XY', self.window.canvas._cursor_position],
                                         ['ADU', self.window.canvas._cursor_value],
                                         ['Mag', self.window.canvas.cursor_value_transf(
                                             self.lens_patch[0].mag_formula)], ])
        self.toolbar.add_buttons(
            'selection', [['/assets/circle.png'],
                          ['/assets/rect.png'],
                          ['/assets/polygon.png'], ])
        self.toolbar.add_buttons(
            ' ', [['save as json', 'save as png'], ])
        self.toolbar.bindings = [[],
                                 [self._on_circle, self._on_rect, self._on_polygon],
                                 [self._on_save_as_json, self.toolbar.dummy_func], ]
        self.toolbar.rebuild()

        # statusbar
        self.statusbar.log('In development...')

        # window and canvas
        if cell_size is None:
            # default configuration
            cols = 3
            rows = max(self.lens_patch.N//3, 1)
            # 67% of the screen width and 89% of the screen height
            cw = int(0.6666*self.master.winfo_screenwidth()//cols // 100 * 100)
            ch = int(0.8888*self.master.winfo_screenheight()//rows // 100 * 100)
            cell_size = (min(cw, ch), min(cw, ch))
        self.window.canvas.cell_size = cell_size
        self.window.canvas.scroll_event_offset_widget = self.toolbar

        # pack all frames with the grid geometry manager
        self.menubar.grid(columnspan=2, row=0, sticky=tk.NE)
        self.navbar.grid(columnspan=2, row=1, sticky=tk.NE)
        self.toolbar.grid(column=0, row=2, columnspan=1, sticky=tk.N+tk.W+tk.S)
        self.window.grid(column=1, row=2, columnspan=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.statusbar.grid(columnspan=2, row=3, sticky=tk.E+tk.S+tk.W)
        # adding resize configs for window at (1, 2)
        self.grid(sticky=tk.N+tk.S+tk.E+tk.W)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        # test projecting entire image patch
        if not display_off:
            self.project(self.lens_patch.image_patch(cmap=self.colormap),
                         image_data=[f.data for f in self.lens_patch.lens_objects])

        # test reducing image number and column number
        # self.window.canvas.N = 1
        # self.window.canvas.ncols = 1
        # self.window.canvas.matrix_size = 500, 500
        # self.project([self.lens_patch.composite_image])

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

    def project(self, images, image_data=None, verbose=False):
        """
        Project images directly onto the canvas

        Args:
            images <list(PIL.Image object)> - images to be added to the buffer

        Kwargs:
            image_data <list(np.ndarray)> - the raw data represented in the images
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        for i, img in enumerate(images):
            if hasattr(image_data, '__len__'):
                self.window.canvas.add_image(img, i, image_data=image_data[i])
            else:
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

# Properties ##################################################################
    @property
    def selection_mode(self):
        """
        Variable to indicate whether selection mode is on or off

        Args/Kwargs:
            None

        Return:
            mode <str> - mode string ('circle'/'rect'/'polygon' or None)
        """
        if not hasattr(self, '_selection_mode'):
            self._selection_mode = None
        return self._selection_mode

    @selection_mode.setter
    def selection_mode(self, mode):
        """
        Set the selection mode. While selection mode is off, all widgets are enabled.

        Args:
            mode <str> - mode string ('circle'/'rect'/'polygon' or None)

        Kwargs/Return:
            None
        """
        self._selection_mode = mode
        if self._selection_mode is None:
            self.enable(self.toolbar)
        else:
            button_name = "_on_{}".format(self._selection_mode)
            self.disable(self.toolbar, exceptions=[button_name])

    @property
    def colormap(self):
        """
        The colormap used for data representation

        Args/Kwargs:
            None

        Return:
            cmap <str> - matplotlib.cm strings
        """
        if not hasattr(self, '_colormap'):
            self._colormap = tk.StringVar()
            self._colormap.set("gnuplot2")
        return self._colormap.get()

    @colormap.setter
    def colormap(self, cmap):
        """
        Setter for the colormap used for data representation

        Args:
            cmap <str> - new matplotlib.cm string

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_colormap'):
            self._colormap = tk.StringVar()
        self._colormap.set(cmap)

# Bindings ####################################################################
    def _on_open(self, event=None):
        """
        Execute when 'open' event is triggered
        """
        fin = filedialog.askopenfilenames(
            parent=self.master, defaultextension=".fits", multiple=True)
        if fin:
            print(fin)
            # self.lenspatch = MultiLens(fin, auto=False)

    def _on_save_as(self, event=None):
        """
        Execute when 'save as' event is triggered
        """
        fout = filedialog.asksaveasfilename(
            parent=self.master, defaultextension=".json",
            initialfile=self.lens_patch.json_filename())
        if not fout:
            fout = self.lens_patch.json_filename()
        if fout.endswith('json'):
            self.lens_patch.jsonify(name=fout, with_hash=False)

    def _on_close(self, event=None):
        """
        Execute when window is closed
        """
        self.master.quit()
        sys.exit(1)

    def _on_colormap(self):
        """
        Change the colormap and reproject
        """
        # print(self.colormap)
        self.project(self.lens_patch.image_patch(cmap=self.colormap),
                     image_data=[f.data for f in self.lens_patch.lens_objects])

    def _on_help(self, event=None):
        """
        Execute when 'help' event is triggered
        """
        import webbrowser
        helpurl = "https://github.com/phdenzel/gleam/blob/master/README.org"
        webbrowser.open(helpurl, new=0)

    def _on_about(self, event=None):
        """
        Execute when 'about' event is triggered
        """
        pass

    def _on_circle(self, event=None):
        """
        Execute when 'circle selection' event is triggdered
        """
        # if 'circle selection' event already has been triggered
        if self.selection_mode == 'circle':
            self.selection_mode = None
        else:
            self.selection_mode = 'circle'

    def _on_rect(self, event=None):
        """
        Execute when 'rectangle selection' event is triggdered
        """
        # if 'rect selection' event already has been triggered
        if self.selection_mode == 'rect':
            self.selection_mode = None
        else:
            self.selection_mode = 'rect'

    def _on_polygon(self, event=None):
        """
        Execute when 'polygon selection' event is triggdered
        """
        # if 'polygon selection' event already has been triggered
        if self.selection_mode == 'polygon':
            self.selection_mode = None
        else:
            self.selection_mode = 'polygon'
        # bind key press to create polygon, every following click adds to it

    def _on_save_as_json(self, event=None):
        """
        Execute when 'save as json' event is triggered
        """
        fout = self.lens_patch.json_filename()
        if fout.endswith('json'):
            self.lens_patch.jsonify(name=fout, with_hash=False)

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
            print("\n APP")
            print(self.__v__)
            print("\n MENUBAR")
            print(self.menubar.__v__)
            print("\n TOOLBAR")
            print(self.toolbar.__v__)
            print("\n STATUSBAR")
            print(self.statusbar.__v__)
            self.after(100, self.term_shell)


# MAIN FUNCTION ###############################################################
def main(case, args):
    """
    Start the app
    """
    # root, app = App.init(case+"/W3+3-2.U.12907_13034_7446_7573.fits")
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
