#!/usr/bin/env python
"""
@author: phdenzel

Prototypes for mightier app components
"""
###############################################################################
# Imports
###############################################################################
import sys
import re
import traceback
from PIL import ImageTk
if sys.version_info.major < 3:
    import Tkinter as tk
elif sys.version_info.major == 3:
    import tkinter as tk
else:
    raise ImportError("Could not import Tkinter")


# TODO:
#     - make environment iterable... could be useful to just do 'for e in self.env...'

# ENV CLASS ###################################################################
class Environment(object):
    """
    An environment where all tk app components are registered and easily accessible
    """
    def __init__(self, item, root=None):
        """
        Initialize an environment with elements directly accessible as object or in a list
        and a tree describing their hierarchy

        Args:
            item <tk object> - a tk widget object such as tk.Canvas or tk.Frame

        Kwargs:
            root <tk.Tk object> - root of the tk app

        Return:
            <Environment object> - non-standard initializer; use Environment.from_tk to init
        """
        self.elements = []
        self.tree = Tree()
        if root is not None:
            self += root
        self += item

    def __add__(self, other):
        """
        Use '+' operator to add tk widget objects to the environment

        Args:
            other <tk object> - a tk widget object to be added to the environment

        Kwargs:
            None

        Return:
            env <Environment object> - the environment with the other object added
        """
        if isinstance(other, (tk.Frame, FramePrototype, tk.Canvas, CanvasPrototype)):
            self.elements.append(other._name)
            separator = '>'
            path = self.tk_hierarchy(other, separator=separator)
            self.tree.add(self.tree, path, separator=separator)
            self.__setattr__(other._name, other)
            return self
        elif isinstance(other, tk.Tk):
            if 'root' not in self.elements:
                self.elements.append('root')
                self.tree['root']
            self.__setattr__('root', other)
            return self
        else:
            NotImplemented

    def __str__(self):
        return 'Env::{:s}'.format(':'.join(self.elements))

    @staticmethod
    def tk_hierarchy(other, separator='>'):
        """
        Get the hierarchy string from the other tk object

        Args:
            other <tk >
        """
        hierarchy = []
        obj = other
        while obj is not None:
            if hasattr(obj, '_name'):
                hierarchy.append(obj._name)
            elif isinstance(obj, tk.Tk):
                hierarchy.append('root')
            else:
                hierarchy.append(obj.__class__.__name__.lower())
            obj = obj.master
        path = separator.join(hierarchy[::-1])
        return path

    @classmethod
    def from_tk(cls, obj):
        """
        Either initialize an environment and add the tk-object and its tk-root to the environment,
        or link to the environment of the tk-object's master if there is any.

        Args:
            TODO

        Kwargs:
            None

        Return:
            env <Environment object> - the environment
        """
        if hasattr(obj, 'master') and hasattr(obj.master, 'env'):
            obj.env = obj.master.env
            obj.env += obj
        elif hasattr(obj, 'master'):
            obj.env = cls(obj, root=obj.master)
        else:
            obj.env = cls(obj)

    @staticmethod
    def add(env, item, master, itemname=None, verbose=False):
        """
        Add a generic object to the environment

        Args:
            env <Environment object> - the environment to which the item is added
            item <Object object> - arbitary object
            node <str> - tk node from which the object comes from

        Kwargs:
           verbose <bool> - verbose mode; print command line statements

        Return:
            env <Environment object> - the environment in its updated state
        """
        if itemname is None:
            itemname = 'generic'
        count = 1
        while itemname in env.elements:
            itemname = itemname+'_{}'.format(count)
            count += 1
        # add env attributes
        env.__setattr__(itemname, item)
        item.__setattr__('env', env)
        # add to elements
        env.elements.append(itemname)
        # add to tree
        path = env.tk_hierarchy(master, separator='>')
        env.tree.add(env.tree, path, separator='>')
        return env


class Tree(dict):
    """
    A recusively initializing Tree dictionary
    """
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

    @staticmethod
    def add(tree, path, separator='>'):
        """
        Add/initialize an entire tree path

        Args:
            tree <Tree object> - a tree to be expanded
            path <str> - path through the tree, e.g. 'root>navbar>button>label>...'

        Kwargs:
            separator <str> - the separator of nodes in the path string; default '>'

        Return:
        """
        path = path.split(separator)
        for node in path:
            tree = tree[node]


# FRAME PROTOTYPE CLASS #######################################################
class FramePrototype(tk.Frame, object):
    """
    To be inherited by other Frame classes for the app

    Note:
       - Base tk.Frame class not really useful on its own
    """
    def __init__(self, master, root=None, add_env=True, *args, **kwargs):
        """
        Initialize with reference to master Tk for Frame

        Args:
            master <Tk object> - master root of the frame

        Kwargs:
            None

        Return:
            <FramePrototype object> - standard initializer
        """
        # default naming convention
        name = kwargs.pop('name', self.__class__.__name__.lower())
        # standard initialization
        tk.Frame.__init__(self, master, name=name, *args, **kwargs)
        # add to environment
        if add_env:
            Environment.from_tk(self)
        # default tests
        self._tests = ['env']

    def __str__(self):
        return "{0:}.{1:}({2:}x{3:})".format(self.master, self.__class__.__name__,
                                             self.winfo_width(), self.winfo_height())

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
        return []

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of the classes attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t))
                          for t in self._tests+self.tests])

    @staticmethod
    def enable(widget):
        """
        Enable all children of a widget

        Args:
            widget <tk.Widget> - a widget with components to enable

        Kwargs/Return:
            None
        """
        for child in widget.winfo_children():
            child.configure(state=tk.NORMAL)

    @staticmethod
    def disable(widget, exceptions=[]):
        """
        Disable all children of a widget

        Args:
            widget <tk.Widget> - a widget with components to enable

        Kwargs:
            exceptions <list(str)> - a list of widget names

        Return:
            None
        """
        for child in widget.winfo_children():
            if hasattr(child, 'name'):
                if child.name not in exceptions:
                    child.configure(state=tk.DISABLED)
            else:
                child.configure(state=tk.DISABLED)

    @staticmethod
    def is_displaying():
        """
        Detects whether Tkinter mainloop is locking the script

        Args:
            None

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            isdisplaying <bool> - True if script in mainloop lock

        Note:
            Tkinter/tkinter module has be imported as tk
        """
        stack = traceback.extract_stack()
        tk_file = tk.__file__
        if tk_file.endswith('pyc'):
            tk_file = tk_file[:-1]
        for (file_name, lineno, function_name, text) in stack:
            if (file_name, function_name) == (tk_file, 'mainloop'):
                return True
        return False


# CANVAS PROTOTYPE CLASS ######################################################
class CanvasPrototype(tk.Canvas, object):
    """
    A resizeable and zoomable Canvas class with imposed 'matrix' structure
    """
    def __init__(self, master, N, cell_size=(300, 300), ncols=3,
                 verbose=False, *args, **kwargs):
        """
        Initialize a gridded canvas with reference to tk.Frame

        Args:
            master <FramePrototype object> - root of the canvas

        Kwargs:
            None

        Return:
            <CanvasPrototype object> - standard initializer
        """
        # default naming convention
        name = kwargs.pop('name', self.__class__.__name__.lower())
        # standard initialization
        tk.Canvas.__init__(self, master, name=name, *args, **kwargs)
        # add to environment
        Environment.from_tk(self)

        # matrix initialization
        self.N = N or 1
        self.cell_size = cell_size
        self.ncols = ncols

        self.cursor_index = 0
        self.cursor_position = 0, 0
        self.cursor_value = 0

        # configure canvas to be resizeable
        self.configure(width=self.matrix_size[0], height=self.matrix_size[1])
        self.grid(sticky=tk.N+tk.S+tk.E+tk.W)

        # binding actions on the canvas
        self.bind("<Configure>", self._resize_on_resize)
        self.bind("<Button-4>", self._zoom_on_scroll, add="+")
        self.bind("<Button-5>", self._zoom_on_scroll, add="+")
        # careful! binding MouseWheel causes UnicodeDecodeError on MacOS Tcl/Tk < 8.6
        self.bind("<MouseWheel>", self._zoom_on_scroll, add="+")
        self.bind("<Motion>", self._track_on_move, add="+")

        self._tests = ['env']
        if verbose:
            print(self.__v__)

    def __str__(self):
        if str(self.master) == '.':
            return "{0:}{1:}({2:}x{3:})".format(self.master, self.__class__.__name__,
                                                self.winfo_width(), self.winfo_height())
        return "{0:}.{1:}({2:}x{3:})".format(self.master, self.__class__.__name__,
                                             self.winfo_width(), self.winfo_height())

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
        return ['N', 'cell_size', 'ncols',
                'cursor_index', 'cursor_position', 'cursor_value', 'zoom',
                'img_buffer', '_img_copy']

    @property
    def __v__(self):
        """
        Info string for test printing

        Args/Kwargs:
            None

        Return:
            <str> - test of the classes attributes
        """
        return "\n".join([t.ljust(20)+"\t{}".format(self.__getattribute__(t))
                          for t in self._tests+self.tests])

    @property
    def N(self):
        """
        Number of images in the buffer

        Args/Kwargs:
            None

        Return:
            N <int> - number of images in buffer, resp. length if buffer list
        """
        if not hasattr(self, '_img_buffer'):
            return 0
        else:
            return len(self.img_buffer)

    @N.setter
    def N(self, N):
        """
        Setter for the length of the image buffer

        Args:
            N <int> - reduce/raise length of buffer list

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_img_buffer'):
            self.img_buffer = (None, N-1)
        elif N > len(self.img_buffer):
            self.img_buffer = (None, N-1)
        elif N < len(self.img_buffer):
            self._img_buffer = self._img_buffer[:N]

    @property
    def ncols(self):
        """
        Number of columns of the canvas matrix

        Args/Kwargs:
            None

        Return:
            ncols <int> - number of columns of the canvas matrix
        """
        return self._ncols

    @ncols.setter
    def ncols(self, ncols):
        """
        Setter for the number of columns of the canvas matrix

        Args:
            ncols <int> - number of columns of the canvas matrix

        Kwargs/Return:
            None
        """
        if self.N <= ncols:
            self._ncols = self.N
        else:
            self._ncols = ncols

    @property
    def matrix_size(self):
        """
        Determine matrix size from image buffer length and cell size

        Args/Kwargs:
            None

        Return:
            width <int>, height <int> - dimensions of the window
        """
        # deduce matrix size from cell size
        w = self.ncols*self.cell_size[0]
        h = (self.N//self.ncols+(0 if self.N % self.ncols == 0 else 1))*self.cell_size[1]
        return w, h

    @matrix_size.setter
    def matrix_size(self, dimensions):
        """
        Setter for the canvas size, effectively changing matrix cell size

        Args:
            dimensions <int,int> - dimensions of the entire canvas

        Kwargs/Return:
            None
        """
        # set the size of canvas and its master
        self.master.configure(width=dimensions[0], height=dimensions[1])
        self.configure(width=dimensions[0], height=dimensions[1])
        # matrix size determined by cell size for fixed number of columns
        cw = dimensions[0]//self.ncols
        ch = dimensions[1]//(self.N//self.ncols+(0 if self.N % self.ncols == 0 else 1))
        self.cell_size = (cw, ch)

    @property
    def img_buffer(self):
        """
        Image cache buffer for persistent/efficient window projection

        Args/Kwargs:
            None

        Return:
            img_buffer <list(PIL.Image object)> - a list of temporary image objects
        """
        return self._img_buffer

    @img_buffer.setter
    def img_buffer(self, img_n_idx):
        """
        Setter of image cache buffer for persistent/efficient window projection

        Args:
            img_n_idx <tuple(image, index)> - tuple-packed input with image and index

        Kwargs/Return:
            None

        Note:
            - Set a list element of img_buffer by assigning it an image and index e.g.
              'self.img_buffer = (img, index)'
            - if the assigned index doesn't exist, the img_buffer expands and initializes
              elements with 'None'
        """
        # lazy load
        if not hasattr(self, '_img_buffer'):
            self._img_buffer = []
            self._img_copy = []
            self._img_data = []
        img, idx = img_n_idx
        # expand buffer list if it's too short
        if idx > len(self._img_buffer)-1:
            tail = [None]*(idx-len(self._img_buffer)+1)
            self._img_buffer = self._img_buffer+tail
            self._img_copy = self._img_copy+tail
            self._img_data = self._img_data+tail
        self._img_buffer[idx] = img

    def matrix_anchor(self, index, loc='NW', verbose=False):
        """
        Calculate the matrix position at north-west corner/center from image index

        Args:
            index <int> - index of the image in the Multilens object <lens_patch>

        Kwargs:
            loc <str> - location of the anchor within the cell (NW//SW/SE/CENTER)
            verbose <bool> - verbose mode; print command line statements

        Return:
            matrix_anchor <int,int> - north-west cell position on the matrix at specified index
        """
        # nrows = (self.N//self.ncols+(0 if self.N % self.ncols == 0 else 1))
        nw = (index % self.ncols)*self.cell_size[0], (index // self.ncols)*self.cell_size[1]
        if loc.upper() == 'NW':
            return nw
        elif loc.upper() == 'NE':
            return nw[0]+self.cell_size[0], nw[1]
        elif loc.upper() == 'CENTER':
            return nw[0]+self.cell_size[0]//2, nw[1]+self.cell_size[1]//2
        elif loc.upper() == 'SE':
            return nw[0]+self.cell_size[0], nw[1]+self.cell_size[1]
        elif loc.upper() == 'SW':
            return nw[0], nw[1]+self.cell_size[1]

    def matrix_index(self, position, verbose=False):
        """
        Get the index of the image in the buffer list from a position on the matrix

        Args:
            position <int,int> - pixel coordinates on the canvas matrix

        Kwargs:
            verbose <bool> - verbose mode; print command line statements

        Return:
            index <int> - index of the image buffer list
        """
        idx = position[0]//self.cell_size[0] + position[1]//self.cell_size[1] * self.ncols
        if idx in range(self.N):
            return idx
        else:
            return 0

    def matrix_position(self, position, relative=False, verbose=False):
        """
        Convert canvas position to a position in the cell

        Args:
            position <int,int> - position on the canvas with origin at north-west corner

        Kwargs:
            rel <bool>
            verbose <bool> - verbose mode; print command line statements

        Return:
            p_actual <int,int> - transformed positions corrected for scale and zoom
        """
        idx = self.matrix_index(position)
        nw_anc = self.matrix_anchor(idx, loc='NW')
        if self._img_copy[idx]:
            w, h = self._img_copy[idx].size
        else:
            return 0, 0
        # relative to cell corner
        cell_pos = position[0]-nw_anc[0], position[1]-nw_anc[1]
        # correct for scale
        scaled_pos = int(cell_pos[0]*w/self.cell_size[0]), int(cell_pos[1]*h/self.cell_size[1])
        # correct for zoom
        xmin, ymin, xmax, ymax = self.zoom_image(idx, return_img=False)
        p_actual = (int(xmin+float(scaled_pos[0])/w*(xmax-xmin)),
                    int(ymin+float(scaled_pos[1])/h*(ymax-ymin)))
        if relative:
            return float(p_actual[0])/w, float(p_actual[1])/h
        else:
            return p_actual

    def add_image(self, image, index, image_data=None, verbose=False):
        """
        Insert image at specified index in buffer and project onto canvas

        Args:
            image <PIL.Image object> - image to be added to buffer
            index <int> - index of the image in the buffer

        Kwargs:
            image_data <np.ndarray> - image data for correct positional data tracking
            verbose <bool> - verbose mode; print command line statements

        Return:
            None
        """
        # only expand buffer if necessary
        if index > len(self.img_buffer)-1:
            self.img_buffer = (None, index)
        # copy original
        self._img_copy[index] = image
        self._img_data[index] = image_data
        # apply image zoom
        if self.zoom_dict[index] > 1:
            image = self.zoom_image(index)
        # move image to buffer and show on canvas
        if image is not None and index < self.N:
            pos = self.matrix_anchor(index, loc='NW')
            image = image.resize(self.cell_size)
            self.img_buffer = (ImageTk.PhotoImage(image=image), index)
            self.create_image(*pos, image=self.img_buffer[index], anchor=tk.NW)

    @property
    def cursor_index(self):
        """
        Index over which cell the cursor is hovering

        Args/Kwargs:
            None

        Return:
            idx <int> - the index of the cell
        """
        if not hasattr(self, '_cursor_index'):
            self._cursor_index = tk.StringVar()
            self._cursor_index.set("{{:=2d}}".format(0))
        return int(self._cursor_index.get())

    @cursor_index.setter
    def cursor_index(self, idx):
        """
        Setter of the cursor index

        Args:
            idx <int> - the new cursor index

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_cursor_index'):
            self._cursor_index = tk.StringVar()
        self._cursor_index.set("{:=2d}".format(idx))

    @property
    def cursor_position(self):
        """
        Position of the cursor within a cell

        Args/Kwargs:
            None

        Return:
            pos <float,float> - the position within the cell
        """
        if not hasattr(self, '_cursor_position'):
            self._cursor_position = tk.StringVar()
            self._cursor_position.set("({:4}, {:4})".format('', ''))
        return [int(s) for s in re.findall(r'\d+', self._cursor_position.get())]

    @cursor_position.setter
    def cursor_position(self, pos):
        """
        Setter of the cursor position

        Args:
            pos <int> - the new cursor position

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_cursor_position'):
            self._cursor_position = tk.StringVar()
        self._cursor_position.set("({:=4d}, {:=4d})".format(pos[0], pos[1]))

    @property
    def cursor_value(self):
        """
        Value of the data over which the cursor is hovering

        Args/Kwargs:
            None

        Return:
            val <float> - the data point value
        """
        if not hasattr(self, '_cursor_value'):
            self._cursor_value = tk.StringVar()
            self._cursor_value.set("{:.4f}".format(0))
        return float(self._cursor_value.get())

    @cursor_value.setter
    def cursor_value(self, val):
        """
        Setter of the cursor value

        Args:
            val <int> - the new cursor value

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_cursor_value'):
            self._cursor_value = tk.StringVar()
        self._cursor_value.set("{:.4f}".format(val))

    def cursor_value_transf(self, transform):
        """
        Add trace to cursor_value and use transform function on it

        Args/Kwargs:
            None

        Return:
            transform <tk.StringVar objects> - transform of the cursor value
        """
        if not hasattr(self, '_cursor_value_transforms'):
            self._cursor_value_transf = []
            self._cursor_value_transforms = []
        transformed = tk.StringVar()

        def transform_trace(*args):
            t = transform(self.cursor_value)
            transformed.set("{:.4f}".format(t))

        self._cursor_value_transf.append(transformed)
        self._cursor_value_transforms.append(transform_trace)
        self._cursor_value.trace("w", self._cursor_value_transforms[-1])
        return self._cursor_value_transf[-1]

    @property
    def zoom_dict(self):
        """
        Zoom settings for images

        Args/Kwargs:
            None

        Return:
            zoom <dict> - dictionary of zoom settings
        """
        if not hasattr(self, '_zoom_dict'):
            zoom_keys = list(range(self.N))                        # zoom factor keys
            zoom_keys += ['c{}'.format(i) for i in range(self.N)]  # zoom position keys
            zoom_init = [1]*self.N
            zoom_init += [(0.5, 0.5)]*self.N
            self._zoom_dict = {k: i for k, i in zip(zoom_keys, zoom_init)}
        if not hasattr(self, '_zoom'):
            self._zoom = tk.StringVar()
            self._zoom.set("x{:=5.2f}".format(1))
            self._zoom_center = tk.StringVar()
            self._zoom_center.set("({:4}, {:4})".format('', ''))
        return self._zoom_dict

    @zoom_dict.setter
    def zoom_dict(self, zoom):
        """
        Update zoom settings for images

        Args:
            zoom <dict> - dictionary of zoom settings

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_zoom_dict'):
            zoom_keys = list(range(self.N))                        # zoom factor keys
            zoom_keys += ['c{}'.format(i) for i in range(self.N)]  # zoom position keys
            zoom_init = [1]*self.N
            zoom_init += [(0.5, 0.5)]*self.N
            self._zoom_dict = {k: i for k, i in zip(zoom_keys, zoom_init)}
        if not hasattr(self, '_zoom'):
            self._zoom = tk.StringVar()
        if not hasattr(self, '_zoom_center'):
            self._zoom_center = tk.StringVar()
        for k in zoom:
            if isinstance(k, int):
                self._zoom.set("x{:=5.2f}".format(zoom[k]))
            if isinstance(k, str):
                pos = zoom[k]
                self._zoom_center.set("({:.2f}, {:.2f})".format(pos[0], pos[1]))
        self._zoom_dict.update(zoom)

    def zoom_image(self, index, warp=True, return_img=True, verbose=False):
        """
        Change zoom specifications for image at specified index in buffer by cropping

        Args:
            index <int> - index of the image in the buffer

        Kwargs:
            return_img <bool> - return crop-zoomed image; if False return cropping limits
            verbose <bool> - verbose mode; print command line statements

        Return:
            crop <PIL.Image object> - cropped PIL image from buffer
        """
        # get zoom image properties
        img = self._img_copy[index]
        z_key = "c{}".format(index)
        center = (int(img.size[0]*self.zoom_dict[z_key][0]),
                  int(img.size[1]*self.zoom_dict[z_key][1]))
        window = (int(img.size[0]/self.zoom_dict[index]),
                  int(img.size[1]/self.zoom_dict[index]))
        if window[0] < 1:
            window[0] = 1
        if window[1] < 1:
            window[1] = 1
        # produce zoomed image by cropping
        z_n = int(center[0]-window[0]//2)
        z_w = int(center[1]-window[1]//2)
        z_s = int(center[0]+window[0]//2)
        z_e = int(center[1]+window[1]//2)
        # either warp image or keep aspect but show background
        if warp:
            if z_n < 0:
                z_n = 0
            if z_w < 0:
                z_w = 0
            if z_s > img.size[0]:
                z_s = img.size[0]
            if z_e > img.size[1]:
                z_e = img.size[1]
        if return_img:
            return img.crop((z_n, z_w, z_s, z_e))
        else:
            return (z_n, z_w, z_s, z_e)

    def _track_on_move(self, event, origin='SW'):
        """
        Tracking actions applied when mouse moves on canvas

        Args:
            event <str> - event to be bound to this function; format <[modifier-]type[-detail]>

        Kwargs/Return:
            None
        """
        # get index
        self.cursor_index = self.matrix_index((event.x, event.y))
        # get position relative to origin (matrix_position anchored to NW by default)
        pos = self.matrix_position((event.x, event.y))
        if hasattr(self, '_img_copy') and self._img_copy[self.cursor_index]:
            w, h = self._img_copy[self.cursor_index].size
        else:
            w, h = 0, 0
        if origin == 'SW':
            self.cursor_position = min(pos[0], w-1), min(h-1-pos[1], h-1)
        if origin == 'NW':
            self.cursor_position = min(pos[0], w-1), min(pos[1], h-1)
        if origin == 'NE':
            self.cursor_position = min(w-1-pos[0], w-1), min(pos[1], h-1)
        if origin == 'SE':
            self.cursor_position = min(w-1-pos[0], w-1), min(h-1-pos[1], h-1)
        # get data
        if self._img_data[self.cursor_index] is not None:
            p = self.cursor_position
            val = self._img_data[self.cursor_index][p[1], p[0]]
        else:
            val = 0
        self.cursor_value = val

    def _track_on_click(self, event):
        """
        Tracking actions applied when mouse clicks on canvas

        Args:
            event <str> - event to be bound to this function; format <[modifier-]type[-detail]>

        Kwargs/Return:
            None
        """
        self._track_on_move(self, event)

    def _resize_on_resize(self, event):
        """
        Resize actions applied when master is resized

        Args:
            event <str> - event to be bound to this function; format <[modifier-]type[-detail]>

        Kwargs/Return:
            None
        """
        # get current size of canvas
        self.matrix_size = (self.master.winfo_width(), self.master.winfo_height())
        # redraw canvas
        for i in range(self.N):
            self.add_image(self._img_copy[i], i, image_data=self._img_data[i])

    def _zoom_on_scroll(self, event):
        """
        Zoom actions applied when canvas is scrolled

        Args:
            event <str> - event to be bound to this function; format <[modifier-]type[-detail]>

        Kwargs/Return:
            None
        """
        # get zoom parameters
        if event.delta > 0:
            dzoom = 1.05
        else:
            dzoom = 0.95
        if event.type == '38':  # event coordinates -> BUG?
            offset = self.scroll_event_offset
            z_x, z_y = event.x-offset[0], event.y-offset[1]
        else:
            z_x, z_y = event.x, event.y
        self.cursor_index = self.matrix_index((z_x, z_y))
        zoom = self.zoom_dict[self.cursor_index]*dzoom
        # back to original settings if the limit is reached
        if zoom < 1:
            zoom = 1
            z_cen = (0.5, 0.5)
        # save zoom settings
        elif self.zoom_dict[self.cursor_index] == 1:  # change zoom center from original settings
            z_anc = self.matrix_anchor(self.cursor_index, loc='NW')
            z_cen = (float(z_x-z_anc[0])/self.cell_size[0],
                     float(z_y-z_anc[1])/self.cell_size[1])
        else:
            z_cen = self.zoom_dict["c{}".format(self.cursor_index)]
        self.zoom_dict = {self.cursor_index: zoom, "c{}".format(self.cursor_index): z_cen}
        # apply zoom
        for i in range(self.N):
            self.add_image(self._img_copy[i], i, image_data=self._img_data[i])

    @property
    def scroll_event_offset(self):
        """
        Helper due to BUG?

        Args/Kwargs:
            None

        Return:
            offset <int,int> - offset of the x and y coordinates of a scroll event
        """
        if not hasattr(self, '_scroll_event_offset'):
            self._scroll_event_offset = tk.StringVar()
            self._scroll_event_offset.set('0x0')
        if hasattr(self, '_scroll_event_offset_widget'):
            offset = self.scroll_event_offset_widget.winfo_width(), 0
            self.scroll_event_offset = offset
        return [int(c) for c in self._scroll_event_offset.get().split('x')]

    @scroll_event_offset.setter
    def scroll_event_offset(self, offset):
        """
        Setter for the scroll_event_offset

        Args:
            offset_func <int,int> - new offset in x and y

        Kwargs/Return:
            None
        """
        if not hasattr(self, '_scroll_event_offset'):
            self._scroll_event_offset = tk.StringVar()
        self._scroll_event_offset.set("{}x{}".format(offset[0], offset[1]))

    @property
    def scroll_event_offset_widget(self):
        """
        Offsetting widget

        Args/Kwargs:
            None

        Return:
            widget <tk.Widget object> - some widget with .winfo_width()
        """
        if not hasattr(self, '_scroll_event_offset_widget'):
            self._scroll_event_offset_widget = self.env.root
        return self._scroll_event_offset_widget

    @scroll_event_offset_widget.setter
    def scroll_event_offset_widget(self, widget):
        """
        Setter for the Offsetting widget

        Args/Kwargs:
            None

        Return:
            widget <tk.Widget object> - some widget with .winfo_width()
        """
        self._scroll_event_offset_widget = widget
