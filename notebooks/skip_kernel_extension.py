"""
@author: phdenzel

Some IPython magitricks:

Load with 
`%load_ext skip_kernel_extension`
and use with
`%%skip True  # skips cell`
`%%skip False # won't skip`
"""

def skip(line, cell=None):
    """
    Skips execution of the current line/cell if line evaluates to True
    """
    if eval(line):
        return

    get_ipython().ex(cell)

def load_ipython_extension(shell):
    """
    Registers the skip magic when the extension loads
    """
    shell.register_magic_function(skip, 'line_cell')

def unload_ipython_extension(shell):
    """
    Unregisters the skip magic when the extension unloads
    """
    del shell.magics_manager.magics['cell']['skip']
