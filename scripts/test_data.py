

import os, sys
sys.path.insert(0, os.path.abspath('..'))
from puia.tests import run_tests


if __name__ == "__main__":
    run_tests('data')