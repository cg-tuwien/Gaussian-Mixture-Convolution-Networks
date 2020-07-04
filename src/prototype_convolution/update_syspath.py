import os
import sys
source_dir = os.path.dirname(__file__)
p = source_dir + '/..'
if p not in sys.path:
    sys.path.append(p)
