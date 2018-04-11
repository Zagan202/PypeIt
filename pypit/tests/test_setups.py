# Module to run tests on scripts
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib
matplotlib.use('agg')  # For Travis

# TEST_UNICODE_LITERALS

import sys, os
import pytest
import glob

from pypit import pyputils

msgs = pyputils.get_dummy_logger()


def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), 'files')
    return os.path.join(data_dir, filename)


def test_run_setup():
    """ Test the setup script
    """
    from pypit.scripts import setup
    import yaml
    # Remove .setup if needed
    sfiles = glob.glob('*.setups')
    for sfile in sfiles:
        os.remove(sfile)
    #
    droot = data_path('b')
    pargs = setup.parser([droot, 'shane_kast_blue', '-d', '-c',
                          '--extension=fits.gz', '--redux_path={:s}'.format(data_path(''))])
    setup.main(pargs)
    setup_file = glob.glob(data_path('setup_files/shane_kast_blue*.setups'))[0]
    # Load
    with open(setup_file, 'r') as infile:
        setup_dict = yaml.load(infile)
    # Test
    assert '01' in setup_dict['A'].keys()
    assert setup_dict['A']['--']['disperser']['name'] == '600/4310'
    # Failures
    pargs2 = setup.parser([droot, 'shane_kast_blu', '-d', '-c',
                              '--extension=fits.gz', '--redux_path={:s}'.format(data_path(''))])
    with pytest.raises(IOError):
        setup.main(pargs2)


def test_setup_made_pypit_file():
    """ Test the .pypit file(s) made by pypit_setup
    """
    from pypit.pypit import load_input
    pyp_file = data_path('shane_kast_blue_setup_A/shane_kast_blue_setup_A.pypit')
    pyp_dict = load_input(pyp_file, msgs)
    # Test
    assert len(pyp_dict['dat']) == 2
    assert pyp_dict['ftype']['b1.fits.gz'] == 'arc'
    assert pyp_dict['setup']['name'][0] == 'A'

'''
def test_setup_pfile():
    """ This won't run because of msgs issue..
    """
    from pypit.scripts import setup
    # Try from .pypit file
    sfiles = glob.glob('*.setup')
    pypit_file = sfiles[0].replace('.setup', '.pypit')
    for sfile in sfiles:
        os.remove(sfile)
    pargs2 = setup.parser([pypit_file, 'shane_kast_blue', '--pypit_file', '-d'])
    setup.main(pargs2)
    pytest.set_trace()
'''
