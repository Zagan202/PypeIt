""" Abstract class for Master Image frames
      This could go inside armasters.py
"""
from __future__ import absolute_import, division, print_function

import warnings

from pypit import msgs
from pypit.core import armasters

from pypit import ardebug as debugger

from abc import ABCMeta

default_settings = dict(masters={
    'directory': 'MF/',  # local to the run directory
    'reuse': False,
    'force': False,
    'loaded': [],
    'setup': None,
})


class MasterFrame(object):
    """
    This class is designed to gather a set of like methods
    for Master calibration frames

    Parameters
    ----------
    frametype : str
    setup : str
    settings : dict

    Attributes
    ----------
    """
    __metaclass__ = ABCMeta

    def __init__(self, frametype, setup, settings):
        # Start us up

        # Parameters
        self.frametype = frametype
        self.setup = setup
        if settings is None:
            self.settings = {}
        else:
            self.settings = settings
        self.msframe = None

        # Kludge settings a bit for now
        if 'masters' not in self.settings.keys():
            self.settings['masters'] = {}
            try:
                self.settings['masters']['directory'] = self.settings['run']['directory']['master']+'_'+self.settings['run']['spectrograph']
            except:
                msgs.warn("MasterFrame class not fully loaded (e.g. no masters in settings).  Avoid using Master methods")
                self.settings['masters'] = default_settings['masters'].copy()
            else:
                for key in ['loaded', 'reuse', 'force']:
                    self.settings['masters'][key] = settings['reduce']['masters'][key]

    @property
    def ms_name(self):
        ms_name = armasters.master_name(self.frametype, self.setup,
                                             self.settings['masters']['directory'])
        return ms_name

    @property
    def mdir(self):
        return self.settings['masters']['directory']

    def _masters_load_chk(self):
        # Logic on whether to load the masters frame
        if (self.settings['masters']['reuse']) or (self.settings['masters']['force']):
            return True

    def load_master_frame(self, force=False):
        """
        Load a MasterFrame

        Parameters
        ----------

        Returns
        -------
        master_frame : ndarray or dict or None
        head0 : Header or None
        file_list : list or None
        """
        if self._masters_load_chk() or force:
            return armasters.load_master_frame(self.frametype, self.setup, self.mdir, force=force)
        else:
            return None, None, None

    def master(self):
        """
        Load the master frame from disk, as settings allows

        Returns
        -------
        msframe : ndarray or None
          arc image

        """
        # Load the MasterFrame if it exists and user requested one to load it
        msframe, header, raw_files = self.load_master_frame()
        if msframe is None:
            return None
        else:
            # Prevent over-writing the master frame when it is time to save
            self.settings['masters']['loaded'].append(self.frametype+self.setup)
            # Hold it
            self.msframe = msframe
        # Return
        return msframe.copy()

    def save_master(self, data, outfile=None, raw_files=None, steps=None):
        """
        Save the input data as a MasterFrame file
          Primarily a wrapper to armasters.core_save_master

        Intended for simple images only; more complex objects need their own method

        Parameters
        ----------
        data : ndarray or dict
        outfile : str (optional)
        raw_files : list (optional)
        steps : list (optional)
        """
        if outfile is None:
            outfile = self.ms_name
        # Steps
        if steps is not None:
            jsteps = ','
            jsteps.join(steps)
            keywds=dict(steps=jsteps)
        else:
            keywds = None
        # Finish
        armasters.save_master(data, filename=outfile,
                                   raw_files=raw_files, keywds=keywds,
                                   frametype=self.frametype)

