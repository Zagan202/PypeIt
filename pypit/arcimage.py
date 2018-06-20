# Module for generating the Arc image
from __future__ import absolute_import, division, print_function

import inspect
import numpy as np
import os

from pypit import msgs
from pypit import processimages
from pypit import masterframe
from pypit.core import arsort

from pypit import ardebug as debugger

# For out of PYPIT running
if msgs._debug is None:
    debug = debugger.init()
    debug['develop'] = True
    msgs.reset(debug=debug, verbosity=2)

# Does not need to be global, but I prefer it
frametype = 'arc'


class ArcImage(processimages.ProcessImages, masterframe.MasterFrame):
    """
    This class is primarily designed to generate an Arc Image from one or more arc frames
      The master() method returns the image (loaded or built)

    Parameters
    ----------
    file_list : list (optional)
      List of filenames
    spectrograph : str (optional)
       Used to specify properties of the detector (for processing)
       Passed to ProcessImages
       Attempts to set with settings['run']['spectrograph'] if not input
    settings : dict (optional)
       Passed to ProcessImages
       Settings for image combining+detector
    setup : str (optional)
      Setup tag;  required for MasterFrame functionality
    det : int, optional
      Detector index, starts at 1
    sci_ID : int (optional)
      Science ID value
      used to match bias frames to the current science exposure
    msbias : ndarray or str
      Guides bias subtraction
    fitstbl : Table (optional)
      FITS info (mainly for filenames)

    Attributes
    ----------
    frametype : str
      Set to 'arc'

    Inherited Attributes
    --------------------
    stack : ndarray
      Final output image
    """
    # Keep order same as processimages (or else!)
    def __init__(self, file_list=[], spectrograph=None, settings=None, det=1, setup=None, sci_ID=None,
                 msbias=None, fitstbl=None):

        # Parameters unique to this Object
        self.sci_ID = sci_ID
        self.msbias = msbias
        self.fitstbl = fitstbl
        self.setup = setup

        # Start us up
        processimages.ProcessImages.__init__(self, file_list, spectrograph=spectrograph, settings=settings, det=det)

        # Attributes (set after init)
        self.frametype = frametype

        # Settings
        # The copy allows up to update settings with user settings without changing the original
        if settings is None:
            # Defaults
            self.settings = processimages.default_settings()
        else:
            self.settings = settings.copy()
            # The following is somewhat kludgy and the current way we do settings may
            #   not touch all the options (not sure .update() would help)
            if 'combine' not in settings.keys():
                if self.frametype in settings.keys():
                    self.settings['combine'] = settings[self.frametype]['combine']

        # Child-specific Internals
        #    See ProcessImages for the rest

        # MasterFrames
        masterframe.MasterFrame.__init__(self, self.frametype, self.setup, self.settings)

    def build_image(self):
        """
        Build the arc image from one or more arc files

        Returns
        -------

        """
        # Get list of arc frames for this science frame
        #  unless one was input already
        if self.nfiles == 0:
            self.file_list = arsort.list_of_files(self.fitstbl, self.frametype, self.sci_ID)
        # Combine
        self.stack = self.process(bias_subtract=self.msbias)
        #
        return self.stack


def get_msarc(det, setup, sci_ID, spectrograph, fitstbl, tsettings, msbias):
    """
    Grab/generate an Arc image

    Parameters
    ----------
    det : int
      Required for processing
    setup : str
      Required for MasterFrame loading
    sci_ID : int
      Required to choose the right arc
    spectrograph : str
      Required if processing
    fitstbl : Table
      Required to choose the right arc
    tsettings : dict
      Required if processing or loading MasterFrame
    msbias : ndarray or str
      Required if processing

    Returns
    -------
    msarc : ndarray
    arcImage : ArcImage object

    """
    # Instantiate with everything needed to generate the image (in case we do)
    arcImage = ArcImage([], spectrograph=spectrograph,
                                 settings=tsettings, det=det, setup=setup,
                                 sci_ID=sci_ID, msbias=msbias, fitstbl=fitstbl)
    # Load the MasterFrame (if it exists and is desired)?
    msarc = arcImage.master()
    if msarc is None:  # Otherwise build it
        msgs.info("Preparing a master {0:s} frame".format(arcImage.frametype))
        msarc = arcImage.build_image()
        # Save to Masters
        arcImage.save_master(msarc, raw_files=arcImage.file_list, steps=arcImage.steps)
    # Return
    return msarc, arcImage
