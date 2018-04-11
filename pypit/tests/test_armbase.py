# Module to run tests on armbase module
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# TEST_UNICODE_LITERALS

import numpy as np
import pytest

from pypit import pyputils
msgs = pyputils.get_dummy_logger()
from pypit import arutils as arut
from pypit import armbase as armb

#def data_path(filename):
#    data_dir = os.path.join(os.path.dirname(__file__), 'files')
#    return os.path.join(data_dir, filename)


def test_update_masters():
    # Dummy self
    arut.dummy_settings(spectrograph='shane_kast_blue', set_idx=True)
    slf1 = arut.dummy_self()
    slf1._idx_arcs = np.array([0,1])
    slf2 = arut.dummy_self()
    sciexp = [slf1, slf2]
    #  Not actually filling anything
    armb.UpdateMasters(sciexp, 0, 1, 'arc')

