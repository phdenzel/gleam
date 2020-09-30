import sys
import os.path
# import gleam.app
# import gleam.model
# import gleam.test
# import gleam.utils
# import gleam.glass_interface
# import gleam.skycoords
# import gleam.skyf
# import gleam.lensobject
# import gleam.skypatch
# import gleam.multilens
# import gleam.glscfactory
# import gleam.lensfinder
# import gleam.roiselector
# import gleam.lightsampler
# import gleam.redshiftsampler
# import gleam.starsampler
# import gleam.reconsrc

# import gleam.gui
# import gleam.megacam


# Add root to PYTHONPATH
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# if os.path.exists(root) and root not in sys.path:
#     sys.path.insert(2, root)
#     # print("Adding {} to PYTHONPATH".format(root))

# # Add glass to PYTHONPATH
# libspath = os.path.join(root, 'libs')
# if os.path.exists(libspath):
#     libs = os.listdir(libspath)[::-1]
#     for l in libs:
#         lib = os.path.join(libspath, l)
#         if lib not in sys.path or not any(['glass' in p for p in sys.path]):
#             sys.path.insert(3, lib)
