import sys
if sys.version_info >= (3, 0):
    sys.stderr.write("metric-learn works with python 2.6+, but not python 3.*. Exiting.")
    sys.exit(1)
from itml import ITML
from lmnn import LMNN
from lsml import LSML
from sdml import SDML
from nca import NCA
from lfda import LFDA
from rca import RCA
