import sys
import os.path

# Add root to PYTHONPATH (root is two levels down)
root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    os.path.pardir))

if os.path.exists(root) and root not in sys.path:
    sys.path.insert(2, root)
    # print("Adding {} to PYTHONPATH".format(root))
