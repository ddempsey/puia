
# general imports
import os, ast
from inspect import getfile, currentframe
from .utilities import DummyClass

def test_data():
    from data import TremorData
    wd=os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data'])
    td=TremorData(station='TEST', data_dir=wd)
    rs=td.df['rsam']
    td.parent=DummyClass(data_streams=['inv_rsam', 'diff_dsar'])
    td._compute_transforms()

def run_tests(testname='all'):
    # runs selected tests
    testfile = getfile(currentframe())
    with open(testfile,'r') as fp:
        tree = ast.parse(fp.read(), filename=testfile)
        
    fs = [f.name for f in tree.body if isinstance(f, ast.FunctionDef)]
    fs.pop(fs.index('run_tests'))

    if testname is 'all':
        for f in fs:
            eval('{:s}()'.format(f))
        return
    
    testname = 'test_'+testname
    if testname not in fs:
        raise ValueError('test function \'{:s}\' not found'.format(testname))

    eval('{:s}()'.format(testname))

