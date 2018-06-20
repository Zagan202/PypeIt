# Module for guiding construction of the Wavelength Image
from __future__ import absolute_import, division, print_function

import inspect
import numpy as np
import os

#from importlib import reload

from pypit import msgs
from pypit import ardebug as debugger
from pypit import arutils
from pypit import masterframe
from pypit import ginga

# For out of PYPIT running
if msgs._debug is None:
    debug = debugger.init()
    debug['develop'] = True
    msgs.reset(debug=debug, verbosity=2)

frametype = 'wave'

default_settings = dict()


class WaveImage(masterframe.MasterFrame):
    """Class to generate the Wavelength Image

    Parameters
    ----------
    tilts : ndarray
      Tilt image
    wv_calib : dict
      1D wavelength solutions
    settings : dict
    setup : str
    maskslits : ndarray
      True = skip this slit
    slitpix : ndarray
      Specifies locations of pixels in the slits

    Attributes
    ----------
    frametype : str
      Hard-coded to 'wave'
    wave : ndarray
      Wavelength image

    steps : list
      List of the processing steps performed
    """
    def __init__(self, tilts=None, wv_calib=None, settings=None,
                 setup=None, maskslits=None, slitpix=None):

        # Required parameters (but can be None)
        self.tilts = tilts
        self.wv_calib = wv_calib
        self.maskslits = maskslits
        self.slitpix = slitpix

        # Optional parameters
        self.setup = setup
        self.settings = settings

        # Attributes
        self.frametype = frametype
        self.steps = []

        # Main outputs
        self.wave = None

        # Key Internals

        # MasterFrame
        masterframe.MasterFrame.__init__(self, self.frametype, setup, self.settings)

    def _build_wave(self):
        """
        Main algorithm to build the wavelength image

        Returns
        -------
        self.wave : ndarray
          Wavelength image

        """
        # Loop on slits
        ok_slits = np.where(~self.maskslits)[0]
        self.wave = np.zeros_like(self.tilts)
        for slit in ok_slits:
            iwv_calib = self.wv_calib[str(slit)]
            tmpwv = arutils.func_val(iwv_calib['fitc'], self.tilts, iwv_calib['function'],
                                     minv=iwv_calib['fmin'], maxv=iwv_calib['fmax'])
            word = np.where(self.slitpix == slit+1)
            self.wave[word] = tmpwv[word]
        # Step
        self.steps.append(inspect.stack()[0][3])
        # Return
        return self.wave

    def show(self, item):
        """
        Show the image

        Parameters
        ----------
        item

        Returns
        -------

        """
        if item == 'wave':
            if self.wave is not None:
                ginga.show_image(self.wave)
        else:
            msgs.warn("Not able to show this type of image")

    def __repr__(self):
        # Generate sets string
        txt = '<{:s}: >'.format(self.__class__.__name__)
        return txt


def get_mswave(setup, tslits_dict, wvimg_settings, mstilts, wv_calib, maskslits):
    """
    Load/Generate the wavelength image


    Parameters
    ----------
    setup : str
      Required for MasterFrame loading
    tslits_dict : dict
      Slits dict; required for processing
    wvimg_settings : dict
      Settings for wavelength image loading or generation
    mstilts : ndarray
      Tilts image; required for processing
    wv_calib : dict
      1D wavelength fits
    maskslits : ndarray (bool)
      Indicates which slits are masked

    Returns
    -------
    mswave : ndarray
    waveImage : WaveImage object

    """
    # Instantiate
    waveImage = WaveImage(mstilts, wv_calib, settings=wvimg_settings,
                                    setup=setup, maskslits=maskslits,
                                    slitpix=tslits_dict['slitpix'])
    # Attempt to load master
    mswave = waveImage.master()
    if mswave is None:
        mswave = waveImage._build_wave()
    # Save to hard-drive
    waveImage.save_master(mswave, steps=waveImage.steps)
    # Return
    return mswave, waveImage
