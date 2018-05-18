# Module for guiding Slit/Order tracing
from __future__ import absolute_import, division, print_function

import inspect
import numpy as np

#from importlib import reload


from pypit import msgs
from pypit import ardebug as debugger
from pypit.core import arflux
from pypit import arload
from pypit import masterframe


# For out of PYPIT running
if msgs._debug is None:
    debug = debugger.init()
    debug['develop'] = True
    msgs.reset(debug=debug, verbosity=2)

frametype = 'sensfunc'

# Default settings, if any


class FluxSpec(masterframe.MasterFrame):
    """Class to guide fluxing

    Parameters
    ----------

    Attributes
    ----------
    """
    def __init__(self, std_spec1d_file=None, sens_file=None, setup=None, settings=None):

        # Required parameters (but can be None)
        self.std_spec1d_file = std_spec1d_file
        self.sens_file = sens_file

        # Optional parameters
        self.settings = settings

        # Attributes
        self.frametype = frametype

        # Main outputs
        self.sens_func = None  # narray

        # Key Internals
        self.std_specobjs = []
        self.std_header = None

        # Load
        if self.std_spec1d_file is not None:
            self.std_specobjs, self.std_header = arload.load_specobj(self.std_spec1d_file)

        # MasterFrame
        masterframe.MasterFrame.__init__(self, self.frametype, setup, self.settings)

    def generate_sens_func(self):
        if len(self.std_specobjs) == 0:
            msgs.warn("You need to load in the Standard spectra first")
            return None
        # Parse the header
        self.sens_dict = arflux.generate_sensfunc(self.std_specobjs, self.std_header['RA'],
                                            self.std_header['DEC'], self.std_header['AIRMASS'],
                                            self.std_header['EXPTIME'])

    def __repr__(self):
        # Generate sets string
        txt = '<{:s}: >'.format(self.__class__.__name__)
        return txt


