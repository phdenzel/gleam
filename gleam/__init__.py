import sys
import os.path

# Add root to PYTHONPATH
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if os.path.exists(root) and root not in sys.path:
    sys.path.insert(2, root)
    # print("Adding {} to PYTHONPATH".format(root))

# Add glass to PYTHONPATH
libspath = os.path.join(root, 'libs')
if os.path.exists(libspath):
    libs = os.listdir(libspath)[::-1]
    for l in libs:
        lib = os.path.join(libspath, l)
        if lib not in sys.path or not any(['glass' in p for p in sys.path]):
            sys.path.insert(3, lib)
